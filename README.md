# Skin Cancer Detection Web App

This is a web application that uses deep learning (PyTorch) to detect and classify skin cancer from images. The application allows users to upload images of skin lesions and get predictions on the type of lesion and its malignancy potential.

## Features

- Upload skin lesion images for analysis
- Get predictions with confidence scores
- View detailed information about different types of skin lesions
- Modern, responsive user interface

## Prerequisites

- Python 3.6+
- PyTorch
- Flask
- OpenCV
- A trained model file (`skin_cancer_model.pth`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/skin-cancer-detection.git
cd skin-cancer-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv skin_cancer_env
skin_cancer_env\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Make sure you have the trained model file `skin_cancer_model.pth` in the project directory.

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and go to `http://127.0.0.1:5000`

3. Upload an image of a skin lesion using the web interface and click "Analyze Image"

4. View the prediction results and information about the detected skin lesion type

## Training Your Own Model

If you want to train your own model, you can use the provided scripts:

1. Make sure you have the HAM10000 dataset in the `dataverse_files` directory
2. Run the training script:
```bash
python main_pytorch.py
```

## Project Structure

- `app.py` - Main Flask application
- `model_pytorch.py` - PyTorch model definition
- `utils_pytorch.py` - Utilities for data loading and preprocessing
- `main_pytorch.py` - Script for training the model
- `skin_cancer_model.pth` - Trained model file
- `static/` - Static assets (CSS, JS, uploaded images)
- `templates/` - HTML templates

## Disclaimer

This tool is intended for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
