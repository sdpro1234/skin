from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from transformers import BartModel, BartTokenizer
from torchvision import models
import pandas as pd
import numpy as np
import g4f
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction_type = db.Column(db.String(20), nullable=False)  # 'text', 'image', 'both'
    prediction_result = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(200))
    text_input = db.Column(db.Text)

# Create tables
with app.app_context():
    db.create_all()

# --- Model Definition ---
class MultiModalClassifier(nn.Module):
    def __init__(self, text_model, image_model, text_feat_dim, image_feat_dim, hidden_dim, num_classes):
        super(MultiModalClassifier, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.text_fc = nn.Linear(text_feat_dim, hidden_dim)
        self.image_fc = nn.Linear(image_feat_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text_input=None, image_input=None):
        features = None
        if text_input is not None:
            text_input_filtered = {k: v for k, v in text_input.items() if k != "labels"}
            text_outputs = self.text_model(**text_input_filtered)
            pooled_text = text_outputs.last_hidden_state.mean(dim=1)
            text_features = self.text_fc(pooled_text)
            features = text_features if features is None else features + text_features

        if image_input is not None:
            image_features = self.image_model(image_input)
            image_features = self.image_fc(image_features)
            features = image_features if features is None else features + image_features

        if (text_input is not None) and (image_input is not None):
            features = features / 2

        logits = self.classifier(features)
        return logits

# --- Initialize Model ---
def init_model():
    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    # Load label list
    text_df = pd.read_csv("dataset.csv")
    label_list = text_df['Label'].unique().tolist()
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load image model
    image_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    image_model.fc = nn.Identity()
    
    # Create multimodal model
    model = MultiModalClassifier(
        text_model=BartModel.from_pretrained(model_name),
        image_model=image_model,
        text_feat_dim=768,
        image_feat_dim=512,
        hidden_dim=512,
        num_classes=len(label_list)
    )
    
    # Load weights
    model.load_state_dict(torch.load("multimodal_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer, id2label, device

model, tokenizer, id2label, device = init_model()

# --- Image Transformations ---
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Inference Functions ---
def inference_text(text, max_length=128):
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    for key in encoding:
        encoding[key] = encoding[key].to(device)
    with torch.no_grad():
        logits = model(text_input=encoding, image_input=None)
    probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
    pred_id = int(np.argmax(probs))
    return id2label[pred_id], probs

def inference_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image_transforms(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        logits = model(text_input=None, image_input=image)
    probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
    pred_id = int(np.argmax(probs))
    return id2label[pred_id], probs

def inference_both(text, image_path, max_length=128):
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    for key in encoding:
        encoding[key] = encoding[key].to(device)
    image = Image.open(image_path).convert("RGB")
    image = image_transforms(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        logits = model(text_input=encoding, image_input=image)
    probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
    pred_id = int(np.argmax(probs))
    return id2label[pred_id], probs

# --- GPT-4 Functions ---
def generate_response(user_input):
    try:
        response = g4f.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_input}],
            temperature=0.6,
            top_p=0.9
        )
        return response.strip() if response else "Sorry, I didn't understand that."
    except Exception as e:
        return f"Error: {e}"

def get_remedy_suggestions(pred_label):
    prompt = (
        f"Provide some home remedy suggestions and potential over-the-counter medicine recommendations for managing the skin condition '{pred_label}'. "
        "Include natural remedies, dietary or lifestyle changes, and common medications that might help alleviate symptoms. "
        "Also add a disclaimer that these suggestions are for informational purposes only and do not replace professional medical advice."
    )
    return generate_response(prompt)

# --- Authentication Routes ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))

        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Username or email already exists!', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

# --- Dashboard Route ---
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to access the dashboard.', 'warning')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    predictions = Prediction.query.filter_by(user_id=user.id).order_by(Prediction.created_at.desc()).limit(10).all()
    return render_template('dashboard.html', user=user, predictions=predictions)

# --- Protected Routes ---
@app.route('/home')
def home():
    if 'user_id' in session:
        return render_template('index.html', logged_in=True)
    return render_template('index.html', logged_in=False)

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.form
    text_input = data.get('text_input', '')
    image_file = request.files.get('image_file')
    
    image_path = None
    if image_file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session['user_id']}_{datetime.now().timestamp()}_{image_file.filename}")
        image_file.save(image_path)
    
    mode = data.get('mode')
    
    try:
        if mode == 'text' and text_input:
            pred_label, probs = inference_text(text_input)
        elif mode == 'image' and image_path:
            pred_label, probs = inference_image(image_path)
        elif mode == 'both' and text_input and image_path:
            pred_label, probs = inference_both(text_input, image_path)
        else:
            return jsonify({'error': 'Invalid input for selected mode'}), 400
        
        suggestions = get_remedy_suggestions(pred_label)
        
        # Save prediction to database
        new_prediction = Prediction(
            user_id=session['user_id'],
            prediction_type=mode,
            prediction_result=pred_label,
            confidence=float(np.max(probs)),
            image_path=image_path if image_path else None,
            text_input=text_input if text_input else None
        )
        db.session.add(new_prediction)
        db.session.commit()
        
        # Format probabilities for chart
        probabilities = [{'label': id2label[i], 'value': float(probs[i])} for i in range(len(probs))]
        
        return jsonify({
            'prediction': pred_label,
            'probabilities': probabilities,
            'suggestions': suggestions,
            'hospital_link': 'https://www.google.com/maps/search/Tamil+Nadu+Hospital/'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    response = generate_response(user_input)
    return jsonify({'response': response})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)