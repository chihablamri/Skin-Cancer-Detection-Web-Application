import os
import numpy as np
from utils import DataLoader
from model import SkinCancerModel
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def train_model(data_dir, metadata_path):
    """
    Train the skin cancer detection model
    
    Args:
        data_dir (str): Directory containing the image data
        metadata_path (str): Path to the metadata CSV file
    """
    # Initialize data loader
    data_loader = DataLoader(data_dir, metadata_path)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.prepare_data()
    
    # Initialize and train model
    model = SkinCancerModel()
    model.compile_model()
    
    # Train the model
    history = model.train(X_train, y_train, X_val, y_val)
    
    # Save the model
    model.model.save('model.h5')
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, data_loader.class_mapping)

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def evaluate_model(model, X_test, y_test, class_mapping):
    """
    Evaluate the model and print metrics
    
    Args:
        model (SkinCancerModel): Trained model
        X_test (numpy.ndarray): Test images
        y_test (numpy.ndarray): Test labels
        class_mapping (dict): Mapping of class names to indices
    """
    # Get predictions
    predictions = model.model.predict(X_test)
    predictions = np.argmax(predictions, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Create reverse class mapping
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test_classes,
        predictions,
        target_names=[reverse_mapping[i] for i in range(len(class_mapping))]
    ))

if __name__ == "__main__":
    # Set paths using relative paths from the project root
    data_dir = "dataverse_files"  # Directory containing image subfolders and metadata
    metadata_path = os.path.join(data_dir, "HAM10000_metadata") # Path to the metadata file (no .csv extension needed by pandas)
    # ground_truth_path = os.path.join(data_dir, "ISIC2018_Task3_Test_GroundTruth.csv") # Path to test ground truth if needed later
    
    # Print debug info
    print(f"Current working directory: {os.getcwd()}")
    print(f"Checking if data directory exists: {os.path.exists(data_dir)}")
    if os.path.exists(data_dir):
        print(f"Files in data directory: {os.listdir(data_dir)}")
    
    # Check if data exists
    part1_dir = os.path.join(data_dir, "HAM10000_images_part_1")
    part2_dir = os.path.join(data_dir, "HAM10000_images_part_2")
    if not os.path.exists(part1_dir) and not os.path.exists(part2_dir):
        print(f"Error: Image directories not found!")
        print(f"Looking for: {part1_dir} or {part2_dir}")
        print("Please ensure the HAM10000 image folders exist inside 'dataverse_files'.")
        exit(1)
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file {metadata_path} not found!")
        print("Please ensure the HAM10000_metadata file exists inside 'dataverse_files'.")
        exit(1)
    
    print("Starting model training...")
    
    try:
        train_model(data_dir, metadata_path)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
