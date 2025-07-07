# app.py
#
# This is the main Flask web application for the skin cancer detection system.
# It handles web requests, image uploads, model inference, Grad-CAM explainability, and fuzzy logic recommendations.
#
# The app integrates deep learning, explainable AI, and fuzzy logic for clinical decision support.

import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from explainable_ai import GradCAM
import numpy as np
import cv2

# Add import for fuzzy logic
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# -----------------------------
# FLASK APP SETUP & CONFIGURATION
# -----------------------------
# Initialize Flask app, configure upload folder, secret key, and allowed file types
app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
           static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'))
app.secret_key = 'skin_cancer_detection_app'
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -----------------------------
# DEVICE CONFIGURATION
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# -----------------------------
# MODEL DEFINITION & LOADING
# -----------------------------
# Define the model architecture (ResNet50-based)
class SkinLesionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(SkinLesionModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = models.resnet50(weights=None)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)
        
    def to(self, device):
        self.device = device
        return super().to(device)

# Load the trained model weights
print("Loading model...")
model = SkinLesionModel().to(device)
checkpoint = torch.load('best_skin_cancer_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully!")

# -----------------------------
# EXPLAINABLE AI (Grad-CAM) SETUP
# -----------------------------
# Initialize Grad-CAM explainer for model interpretability
explainer = GradCAM(model)

# -----------------------------
# FUZZY LOGIC SYSTEM SETUP
# -----------------------------
# Define fuzzy logic system for medical attention recommendation
# Membership functions and rules are based on clinical logic

def setup_fuzzy_system():
    # Define input and output variables
    confidence = ctrl.Antecedent(np.arange(0, 101, 1), 'confidence')
    risk_level = ctrl.Antecedent(np.arange(0, 11, 1), 'risk_level')
    medical_attention = ctrl.Consequent(np.arange(0, 101, 1), 'medical_attention')
    
    # Define membership functions for confidence
    confidence['low'] = fuzz.trimf(confidence.universe, [0, 0, 50])
    confidence['medium'] = fuzz.trimf(confidence.universe, [30, 60, 90])
    confidence['high'] = fuzz.trimf(confidence.universe, [70, 100, 100])
    
    # Define membership functions for risk level (0-10 scale)
    risk_level['low'] = fuzz.trimf(risk_level.universe, [0, 0, 4])
    risk_level['medium'] = fuzz.trimf(risk_level.universe, [3, 5, 7])
    risk_level['high'] = fuzz.trimf(risk_level.universe, [6, 10, 10])
    
    # Define membership functions for medical attention
    medical_attention['unnecessary'] = fuzz.trimf(medical_attention.universe, [0, 0, 40])
    medical_attention['recommended'] = fuzz.trimf(medical_attention.universe, [20, 50, 80])
    medical_attention['urgent'] = fuzz.trimf(medical_attention.universe, [60, 100, 100])
    
    # Define rules
    rule1 = ctrl.Rule(confidence['high'] & risk_level['high'], medical_attention['urgent'])
    rule2 = ctrl.Rule(confidence['high'] & risk_level['medium'], medical_attention['recommended'])
    rule3 = ctrl.Rule(confidence['high'] & risk_level['low'], medical_attention['unnecessary'])
    rule4 = ctrl.Rule(confidence['medium'] & risk_level['high'], medical_attention['urgent'])
    rule5 = ctrl.Rule(confidence['medium'] & risk_level['medium'], medical_attention['recommended'])
    rule6 = ctrl.Rule(confidence['medium'] & risk_level['low'], medical_attention['recommended'])
    rule7 = ctrl.Rule(confidence['low'] & risk_level['high'], medical_attention['recommended'])
    rule8 = ctrl.Rule(confidence['low'] & risk_level['medium'], medical_attention['recommended'])
    rule9 = ctrl.Rule(confidence['low'] & risk_level['low'], medical_attention['unnecessary'])
    
    # Create control system
    medical_attention_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    
    return ctrl.ControlSystemSimulation(medical_attention_ctrl)

# Initialize fuzzy system
fuzzy_system = setup_fuzzy_system()

# Function to get fuzzy medical attention recommendation
# Uses fuzzy logic to combine model confidence and risk level into actionable advice
# Returns a dictionary with recommendation type, urgency, message, and score

def get_fuzzy_recommendation(confidence_value, risk_type):
    # Convert risk type to numerical value
    risk_values = {
        'Low': 2,
        'Medium': 5, 
        'High': 8
    }
    
    risk_value = risk_values.get(risk_type, 5)  # Default to medium if unknown
    
    # Remove % sign and convert to float
    if isinstance(confidence_value, str) and '%' in confidence_value:
        confidence_value = float(confidence_value.replace('%', ''))
    
    # Input to fuzzy system
    fuzzy_system.input['confidence'] = confidence_value
    fuzzy_system.input['risk_level'] = risk_value
    
    try:
        # Compute
        fuzzy_system.compute()
        attention_score = fuzzy_system.output['medical_attention']
        
        # Determine recommendation based on score
        if attention_score < 30:
            return {
                'attention_type': 'Self-monitoring',
                'urgency': 'Low',
                'message': 'Regular self-monitoring is recommended.',
                'score': attention_score
            }
        elif attention_score < 70:
            return {
                'attention_type': 'Medical Consultation',
                'urgency': 'Medium',
                'message': 'Consultation with a healthcare provider is recommended.',
                'score': attention_score
            }
        else:
            return {
                'attention_type': 'Urgent Medical Attention',
                'urgency': 'High',
                'message': 'Prompt medical attention is strongly advised.',
                'score': attention_score
            }
    except:
        # Fallback if fuzzy computation fails
        if risk_type == 'High':
            return {
                'attention_type': 'Medical Attention',
                'urgency': 'High',
                'message': 'Due to high risk classification, medical attention is advised.',
                'score': 80
            }
        elif risk_type == 'Medium':
            return {
                'attention_type': 'Medical Consultation',
                'urgency': 'Medium',
                'message': 'Regular monitoring by a healthcare provider is recommended.',
                'score': 50
            }
        else:
            return {
                'attention_type': 'Self-monitoring',
                'urgency': 'Low',
                'message': 'Regular self-monitoring is sufficient.',
                'score': 20
            }

# -----------------------------
# CLASS MAPPING & LABEL INFO
# -----------------------------
# Maps class indices to human-readable names, risk, features, and explanations
CLASS_MAPPING = {
    0: {
        'code': 'akiec',
        'name': 'Actinic Keratoses',
        'risk': 'Medium Risk',
        'description': 'Pre-cancerous growths caused by sun damage. Regular monitoring needed.',
        'features': 'Rough, scaly patches with irregular borders',
        'explanation': 'The model has detected rough, scaly patches characteristic of sun damage. These lesions require regular monitoring by a healthcare provider.'
    },
    1: {
        'code': 'bcc',
        'name': 'Basal Cell Carcinoma',
        'risk': 'High Risk',
        'description': 'The most common form of skin cancer. Requires medical attention.',
        'features': 'Pearly, waxy appearance or possible ulceration',
        'explanation': 'The model has identified features typical of basal cell carcinoma, such as a pearly appearance or possible ulceration. Medical evaluation is recommended.'
    },
    2: {
        'code': 'bkl',
        'name': 'Benign Keratosis',
        'risk': 'Low Risk',
        'description': 'Harmless growth that appears with age.',
        'features': 'Well-defined borders with a "stuck-on" appearance',
        'explanation': 'The analysis shows features consistent with a benign growth, including well-defined borders and a characteristic appearance.'
    },
    3: {
        'code': 'df',
        'name': 'Dermatofibroma',
        'risk': 'Low Risk',
        'description': 'Benign skin growth, common and harmless.',
        'features': 'Firm, raised appearance with regular borders',
        'explanation': 'The model has detected the typical appearance of a dermatofibroma, which is a harmless skin growth.'
    },
    4: {
        'code': 'mel',
        'name': 'Melanoma',
        'risk': 'High Risk',
        'description': 'Serious form of skin cancer. Immediate medical evaluation needed.',
        'features': 'Irregular borders, color variations, or asymmetry',
        'explanation': 'The analysis has identified concerning features such as irregular borders or color variations that warrant immediate medical evaluation.'
    },
    5: {
        'code': 'nv',
        'name': 'Melanocytic Nevi',
        'risk': 'Low Risk',
        'description': 'Common mole, usually harmless but monitor for changes.',
        'features': 'Regular borders, consistent coloring, symmetrical',
        'explanation': 'The model has identified features typical of a common mole, including regular borders and consistent coloring.'
    },
    6: {
        'code': 'vasc',
        'name': 'Vascular Lesions',
        'risk': 'Medium Risk',
        'description': 'Blood vessel-related skin condition.',
        'features': 'Red or purple coloration with typical patterns',
        'explanation': 'The analysis shows patterns typical of vascular lesions, with characteristic coloration and distribution.'
    }
}

# -----------------------------
# IMAGE PROCESSING & PREDICTION
# -----------------------------
# allowed_file: Checks if uploaded file has an allowed extension
# process_image: Handles image preprocessing, model inference, Grad-CAM, and fuzzy logic recommendation
# Returns a dictionary with all prediction and explanation info for the UI

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path):
    """Process image and return prediction with explanation"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_prob, pred_class = torch.max(probabilities, 1)
    
    # Get class information
    class_info = CLASS_MAPPING[pred_class.item()]
    
    # Get top 3 predictions
    top_probs, top_indices = torch.topk(probabilities[0], 3)
    top_predictions = []
    for prob, idx in zip(top_probs, top_indices):
        top_predictions.append({
            'name': CLASS_MAPPING[idx.item()]['name'],
            'probability': f"{prob.item()*100:.2f}%"
        })
    
    # Generate analysis area text based on features
    analysis_area = f"The model has analyzed the following key features:\n"
    analysis_area += f"• {class_info['features']}\n"
    analysis_area += f"• The lesion shows characteristics typical of {class_info['name']}\n"
    if 'High' in class_info['risk']:
        analysis_area += "• Urgent medical attention is recommended"
    elif 'Medium' in class_info['risk']:
        analysis_area += "• Regular medical monitoring is advised"
    else:
        analysis_area += "• Regular self-monitoring is recommended"
    
    # Get Grad-CAM explanation
    try:
        grad_cam_result = explainer.explain(image_path)
        # Fix the path to match the correct directory structure
        heatmap_filename = os.path.basename(grad_cam_result['heatmap_path'])
        overlay_filename = os.path.basename(grad_cam_result['overlay_path'])
        
        explanation = {
            'heatmap_path': heatmap_filename,
            'overlay_path': overlay_filename,
            'region_explanation': analysis_area,
            'text': class_info['explanation']
        }
    except Exception as e:
        print(f"Grad-CAM generation failed: {str(e)}")
        explanation = {
            'heatmap_path': None,
            'overlay_path': None,
            'region_explanation': analysis_area,
            'text': class_info['explanation']
        }
    
    # Prepare confidence text and value
    confidence_value = float(pred_prob.item() * 100)
    confidence_text = f"{confidence_value:.2f}%"
    
    # Get fuzzy logic recommendation
    risk_level = class_info['risk'].split()[0]  # Get "High", "Medium", or "Low"
    fuzzy_recommendation = get_fuzzy_recommendation(confidence_value, risk_level)
    
    return {
        'code': class_info['code'],
        'name': class_info['name'],
        'probability': confidence_text,
        'probability_value': confidence_value,
        'risk': risk_level,
        'description': class_info['description'],
        'feature_explanation': class_info['features'],
        'explanation': explanation,
        'top_predictions': top_predictions,
        'fuzzy_recommendation': fuzzy_recommendation
    }

# -----------------------------
# FLASK ROUTES
# -----------------------------
# '/' route: Handles GET and POST requests for main page
# - GET: Shows upload form
# - POST: Processes uploaded image, runs prediction, and renders results
# '/about' route: Shows about page

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                prediction = process_image(filepath)
                prediction['image'] = filename  # Add filename for display
                
                # Move Grad-CAM generated files to static/uploads if they exist
                if prediction['explanation']['heatmap_path']:
                    src_heatmap = os.path.join(os.path.dirname(filepath), prediction['explanation']['heatmap_path'])
                    dst_heatmap = os.path.join(app.config['UPLOAD_FOLDER'], prediction['explanation']['heatmap_path'])
                    if os.path.exists(src_heatmap):
                        os.replace(src_heatmap, dst_heatmap)
                
                if prediction['explanation']['overlay_path']:
                    src_overlay = os.path.join(os.path.dirname(filepath), prediction['explanation']['overlay_path'])
                    dst_overlay = os.path.join(app.config['UPLOAD_FOLDER'], prediction['explanation']['overlay_path'])
                    if os.path.exists(src_overlay):
                        os.replace(src_overlay, dst_overlay)
                
            except Exception as e:
                flash(f'Error processing image: {str(e)}')
                return redirect(request.url)
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)
    
    return render_template('index.html', 
                         prediction=prediction,
                         filename=filename)

@app.route('/about')
def about():
    return render_template('about.html', class_mapping=CLASS_MAPPING)

if __name__ == '__main__':
    app.run(debug=True)
