import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
from explainable_ai import GradCAM

app = Flask(__name__)
app.secret_key = 'skin_cancer_detection_app'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Define the model architecture
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

# Load the model
print("Loading model...")
model = SkinLesionModel().to(device)
checkpoint = torch.load('best_skin_cancer_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully!")

# Initialize Grad-CAM
explainer = GradCAM(model)

# Class information
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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path):
    """Process image and return prediction with explanation"""
    try:
        # Debug information
        print(f"\nProcessing image: {image_path}")
        print(f"File exists: {os.path.exists(image_path)}")
        if os.path.exists(image_path):
            print(f"File size: {os.path.getsize(image_path)} bytes")
        
        # Verify file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Verify file is not empty
        if os.path.getsize(image_path) == 0:
            raise ValueError("Image file is empty")
            
        # Try to open and verify image
        try:
            # First try to open with PIL
            image = Image.open(image_path)
            print(f"Image format: {image.format}")
            print(f"Image size: {image.size}")
            print(f"Image mode: {image.mode}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Force image loading to verify it's valid
            image.verify()
            
            # Reopen the image after verification
            image = Image.open(image_path).convert('RGB')
            
        except Exception as e:
            print(f"Error opening image: {str(e)}")
            # Try alternative method using OpenCV
            try:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError("OpenCV could not read the image")
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img)
            except Exception as cv2_error:
                raise ValueError(f"Both PIL and OpenCV failed to read the image. PIL error: {str(e)}, OpenCV error: {str(cv2_error)}")
        
        # Image preprocessing
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
        
        # Prepare confidence text
        confidence_value = float(pred_prob.item() * 100)
        
        return {
            'code': class_info['code'],
            'name': class_info['name'],
            'probability': f"{confidence_value:.2f}%",
            'risk': class_info['risk'].split()[0],
            'description': class_info['description'],
            'feature_explanation': class_info['features'],
            'explanation': explanation,
            'top_predictions': top_predictions
        }
        
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        raise Exception(f"Failed to process image: {str(e)}")

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
            try:
                # Read the file content first
                file_content = file.read()
                if len(file_content) < 1000:  # Basic size check
                    raise ValueError("File appears to be too small or corrupted")
                
                # Reset file pointer
                file.seek(0)
                
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Ensure the upload directory exists
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                # Save the file with explicit binary mode
                with open(filepath, 'wb') as f:
                    f.write(file_content)
                
                print(f"\nSaved file to: {filepath}")
                print(f"File size after save: {os.path.getsize(filepath)} bytes")
                
                # Verify the file was saved correctly
                if not os.path.exists(filepath):
                    raise Exception("Failed to save uploaded file")
                
                # Try to open the image immediately after saving
                try:
                    test_image = Image.open(filepath)
                    test_image.verify()  # Verify it's a valid image
                    print(f"Successfully verified image: {filename}")
                except Exception as e:
                    raise ValueError(f"Saved file is not a valid image: {str(e)}")
                
                prediction = process_image(filepath)
                prediction['image'] = filename
                
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
                # Clean up the file if it exists
                if filename and os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                return redirect(request.url)
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)
    
    return render_template('index.html', 
                         prediction=prediction,
                         filename=filename)

@app.route('/about')
def about():
    """Render the about page"""
    class_mapping = {
        0: {
            'name': 'Melanoma',
            'code': 'MEL',
            'risk': 'High',
            'description': 'The most dangerous form of skin cancer, can spread to other organs.'
        },
        1: {
            'name': 'Melanocytic nevus',
            'code': 'NV',
            'risk': 'Low',
            'description': 'Common mole, usually benign.'
        },
        2: {
            'name': 'Basal cell carcinoma',
            'code': 'BCC',
            'risk': 'Medium',
            'description': 'Most common type of skin cancer, rarely spreads.'
        },
        3: {
            'name': 'Actinic keratosis / Bowen\'s disease',
            'code': 'AKIEC',
            'risk': 'Medium',
            'description': 'Precancerous skin condition that can develop into squamous cell carcinoma.'
        },
        4: {
            'name': 'Benign keratosis',
            'code': 'BKL',
            'risk': 'Low',
            'description': 'Non-cancerous skin growths including seborrheic keratosis and solar lentigo.'
        },
        5: {
            'name': 'Dermatofibroma',
            'code': 'DF',
            'risk': 'Low',
            'description': 'Benign skin tumor, usually appears as a hard, raised growth.'
        },
        6: {
            'name': 'Vascular lesion',
            'code': 'VASC',
            'risk': 'Low',
            'description': 'Blood vessel abnormalities including angiomas and pyogenic granulomas.'
        }
    }
    return render_template('about.html', class_mapping=class_mapping)

if __name__ == '__main__':
    app.run(debug=True)
