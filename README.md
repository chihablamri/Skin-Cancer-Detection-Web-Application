# Skin Cancer Detection Web Application

A web-based application for skin cancer detection using deep learning, featuring explainable AI and fuzzy logic-based medical recommendations.

## Features

- **Deep Learning Model**: ResNet50-based model for skin lesion classification
- **Explainable AI**: Grad-CAM visualization for model interpretability
- **Fuzzy Logic System**: Intelligent medical attention recommendations
- **Web Interface**: User-friendly interface for image upload and analysis
- **Comprehensive Analysis**: Detailed predictions with confidence scores and medical recommendations

## Supported Lesion Types

The model can classify 7 different types of skin lesions:
- Actinic Keratoses (akiec)
- Basal Cell Carcinoma (bcc)
- Benign Keratosis (bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic Nevi (nv)
- Vascular Lesions (vasc)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Flask
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd skin-cancer-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the model file:
- Place `best_skin_cancer_model.pth` in the root directory

## Project Structure

```
├── app.py                 # Main Flask application
├── best_skin_cancer_model.pth  # Trained model weights
├── explainable_ai.py      # Grad-CAM implementation
├── model_pytorch.py       # Model architecture
├── utils_pytorch.py       # PyTorch utilities
├── utils.py              # General utilities
├── requirements.txt       # Project dependencies
├── static/               # Static files (CSS, JS, uploads)
└── templates/            # HTML templates
```

## Usage

1. Start the web application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload a skin lesion image through the web interface

4. View the results:
- Classification prediction
- Confidence score
- Risk level assessment
- Medical attention recommendation
- Grad-CAM visualization
- Detailed analysis

## Features in Detail

### 1. Deep Learning Model
- Based on ResNet50 architecture
- Fine-tuned for skin lesion classification
- Achieves high accuracy on the HAM10000 dataset

### 2. Explainable AI (Grad-CAM)
- Visualizes the regions the model focuses on
- Helps understand model decisions
- Provides transparency in predictions

### 3. Fuzzy Logic System
- Analyzes confidence scores and risk levels
- Provides intelligent medical recommendations
- Three levels of urgency:
  - Self-monitoring
  - Medical Consultation
  - Urgent Medical Attention

### 4. Web Interface
- Clean and intuitive design
- Real-time image processing
- Comprehensive result display
- Mobile-responsive layout

## Medical Disclaimer

This application is designed to assist healthcare professionals and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
