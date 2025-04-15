import os
import torch
import numpy as np
from utils_pytorch import PyTorchDataLoader
from model_pytorch import SkinCancerModelPyTorch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import time
from tqdm import tqdm
import platform
import psutil
import multiprocessing

def print_system_info():
    """Print detailed system information"""
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    
    # Python version
    print(f"Python version: {platform.python_version()}")
    
    # Operating system
    print(f"OS: {platform.system()} {platform.version()}")
    
    # CPU info
    print(f"CPU: {platform.processor()}")
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    print(f"CPU cores: {cpu_count} physical, {cpu_count_logical} logical")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.total / (1024**3):.2f} GB total, {memory.available / (1024**3):.2f} GB available")
    
    # GPU info
    if torch.cuda.is_available():
        print("\nGPU INFORMATION:")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
        # Additional GPU properties
        props = torch.cuda.get_device_properties(0)
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Multi-processors: {props.multi_processor_count}")
    else:
        print("\nNo CUDA-compatible GPU detected.")
    
    print("="*50 + "\n")

def check_dataset(data_dir):
    """
    Check dataset for completeness and print detailed info
    
    Returns:
        tuple: (is_valid, message)
    """
    print("\n" + "="*50)
    print("DATASET VALIDATION")
    print("="*50)
    
    # Check if directories exist
    part1_dir = os.path.join(data_dir, "HAM10000_images_part_1")
    part2_dir = os.path.join(data_dir, "HAM10000_images_part_2")
    metadata_path = os.path.join(data_dir, "HAM10000_metadata")
    
    if not os.path.exists(data_dir):
        return False, f"Error: Data directory '{data_dir}' does not exist!"
    
    print(f"✓ Data directory found: {data_dir}")
    
    if not os.path.exists(metadata_path):
        return False, f"Error: Metadata file '{metadata_path}' not found!"
    
    print(f"✓ Metadata file found: {metadata_path}")
    
    # Count images in each directory
    image_count_part1 = 0
    image_count_part2 = 0
    
    if os.path.exists(part1_dir):
        files_part1 = [f for f in os.listdir(part1_dir) if f.endswith('.jpg')]
        image_count_part1 = len(files_part1)
        print(f"✓ Image directory 1 found: {part1_dir} ({image_count_part1} images)")
    else:
        print(f"✗ Image directory not found: {part1_dir}")
    
    if os.path.exists(part2_dir):
        files_part2 = [f for f in os.listdir(part2_dir) if f.endswith('.jpg')]
        image_count_part2 = len(files_part2)
        print(f"✓ Image directory 2 found: {part2_dir} ({image_count_part2} images)")
    else:
        print(f"✗ Image directory not found: {part2_dir}")
    
    total_images = image_count_part1 + image_count_part2
    
    if total_images == 0:
        return False, "Error: No images found in the dataset directories!"
    
    print(f"Total images available: {total_images}")
    
    # Read first few lines of metadata to confirm format
    try:
        with open(metadata_path, 'r') as f:
            first_line = f.readline().strip()
            expected_header = "lesion_id,image_id,dx,dx_type,age,sex,localization,dataset"
            if first_line != expected_header:
                return False, f"Error: Metadata file has unexpected format. Header should be: {expected_header}"
            
            print(f"✓ Metadata file has correct format")
            
            # Count lines in metadata
            f.seek(0)
            line_count = sum(1 for line in f)
            print(f"✓ Metadata contains information for {line_count-1} images")
    except Exception as e:
        return False, f"Error reading metadata file: {str(e)}"
    
    print("="*50 + "\n")
    return True, "Dataset validation successful"

def train_model(data_dir, metadata_path, num_epochs=10, batch_size=32):
    """
    Train the skin cancer detection model with PyTorch
    
    Args:
        data_dir (str): Directory containing the image data
        metadata_path (str): Path to the metadata CSV file
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    """
    start_time = time.time()
    
    # Initialize data loader
    print("\n" + "="*20 + " INITIALIZING DATA LOADER " + "="*20)
    data_loader = PyTorchDataLoader(data_dir, metadata_path, batch_size=batch_size)
    
    # Prepare data
    print("\n" + "="*20 + " PREPARING DATASET " + "="*20)
    print("Loading and preprocessing images... This may take several minutes.")
    print("Splitting data into training, validation, and test sets...")
    
    train_loader, val_loader, test_loader, num_classes = data_loader.prepare_data()
    
    class_names = list(data_loader.class_mapping.keys())
    class_counts = data_loader.get_class_distribution()
    
    print("\nClass distribution in dataset:")
    for class_name in class_names:
        idx = data_loader.class_mapping[class_name]
        count = class_counts.get(idx, 0)
        print(f"  - {class_name}: {count} images")
    
    print(f"\nTotal classes: {num_classes}")
    print(f"Training set: {len(train_loader.dataset)} images")
    print(f"Validation set: {len(val_loader.dataset)} images")
    print(f"Test set: {len(test_loader.dataset)} images")
    print(f"Batch size: {batch_size}")
    data_prep_time = time.time() - start_time
    print(f"Data preparation completed in {data_prep_time:.2f} seconds")
    
    # Initialize model
    print("\n" + "="*20 + " MODEL INITIALIZATION " + "="*20)
    model = SkinCancerModelPyTorch(num_classes=num_classes)
    model.compile()
    
    # Train the model
    print("\n" + "="*20 + " STARTING MODEL TRAINING " + "="*20)
    history = model.fit(train_loader, val_loader, num_epochs=num_epochs)
    
    # Save the model
    print("\n" + "="*20 + " SAVING MODEL " + "="*20)
    model_save_path = 'skin_cancer_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_mapping': data_loader.class_mapping,
        'training_history': history,
        'epochs_trained': num_epochs,
        'date_trained': time.strftime("%Y-%m-%d %H:%M:%S"),
    }, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot training history
    print("\n" + "="*20 + " GENERATING TRAINING PLOTS " + "="*20)
    plot_training_history(history)
    
    # Evaluate model
    print("\n" + "="*20 + " MODEL EVALUATION " + "="*20)
    print(f"Evaluating model on test set ({len(test_loader.dataset)} images)...")
    evaluate_model(model, test_loader, data_loader.class_mapping)
    
    # Print overall training summary
    total_time = time.time() - start_time
    print("\n" + "="*20 + " TRAINING SUMMARY " + "="*20)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('training_history_pytorch.png')
    print(f"Training history plot saved to 'training_history_pytorch.png'")
    plt.close()

def evaluate_model(model, test_loader, class_mapping):
    """
    Evaluate the model and print metrics
    
    Args:
        model (SkinCancerModelPyTorch): Trained model
        test_loader (DataLoader): Test data loader
        class_mapping (dict): Mapping of class names to indices
    """
    # Get predictions
    model.eval()
    all_preds = []
    all_targets = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating", unit="batch"):
            inputs = inputs.to(model.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Create reverse class mapping
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    class_names = [reverse_mapping[i] for i in range(len(class_mapping))]
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names
    )
    print(report)
    
    # Save report to file
    with open('classification_report.txt', 'w') as f:
        f.write("SKIN CANCER CLASSIFICATION REPORT\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(report)
    
    print("Classification report saved to 'classification_report.txt'")

if __name__ == "__main__":
    # Set proper multiprocessing start method for Windows
    if platform.system() == 'Windows':
        # This fixes issues with CUDA multiprocessing on Windows
        multiprocessing.set_start_method('spawn', force=True)
    
    # Set paths using relative paths from the project root
    data_dir = "dataverse_files"  # Directory containing image subfolders and metadata
    metadata_path = os.path.join(data_dir, "HAM10000_metadata") # Path to the metadata file (no .csv extension needed by pandas)
    
    # Print ASCII art header
    print("\n" + "="*70)
    print("""
    ███████╗██╗  ██╗██╗███╗   ██╗     ██████╗ █████╗ ███╗   ██╗ ██████╗███████╗██████╗ 
    ██╔════╝██║ ██╔╝██║████╗  ██║    ██╔════╝██╔══██╗████╗  ██║██╔════╝██╔════╝██╔══██╗
    ███████╗█████╔╝ ██║██╔██╗ ██║    ██║     ███████║██╔██╗ ██║██║     █████╗  ██████╔╝
    ╚════██║██╔═██╗ ██║██║╚██╗██║    ██║     ██╔══██║██║╚██╗██║██║     ██╔══╝  ██╔══██╗
    ███████║██║  ██╗██║██║ ╚████║    ╚██████╗██║  ██║██║ ╚████║╚██████╗███████╗██║  ██║
    ╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝     ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝╚═╝  ╚═╝
                                                                                      
    ██████╗ ██╗   ██╗████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗                        
    ██╔══██╗╚██╗ ██╔╝╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║                        
    ██████╔╝ ╚████╔╝    ██║   ██║   ██║██████╔╝██║     ███████║                        
    ██╔═══╝   ╚██╔╝     ██║   ██║   ██║██╔══██╗██║     ██╔══██║                        
    ██║        ██║      ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║                        
    ╚═╝        ╚═╝      ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝                        
    """)
    print("="*70)
    print("\nSkin Cancer Detection Model Training with PyTorch")
    print("This program will train a deep learning model to classify skin cancer images")
    print("="*70)
    
    try:
        # Install psutil if not available
        try:
            import psutil
        except ImportError:
            print("Installing required package: psutil")
            import subprocess
            subprocess.check_call(["python", "-m", "pip", "install", "psutil"])
            import psutil
        
        # Print system info
        print_system_info()
        
        # Validate dataset
        is_valid, message = check_dataset(data_dir)
        if not is_valid:
            print(f"Dataset error: {message}")
            exit(1)
        
        # Configure training parameters
        print("\nTRAINING CONFIGURATION")
        print("="*30)
        
        # Default values
        num_epochs = 10
        batch_size = 32
        
        try:
            epochs_input = input(f"Number of training epochs (default: {num_epochs}): ").strip()
            if epochs_input:
                num_epochs = int(epochs_input)
            
            batch_input = input(f"Batch size (default: {batch_size}): ").strip()
            if batch_input:
                batch_size = int(batch_input)
        except ValueError:
            print("Invalid input, using default values")
        
        print(f"Training for {num_epochs} epochs with batch size {batch_size}")
        
        # Get user confirmation before starting
        confirm = input("\nDo you want to start training now? (yes/no): ")
        if confirm.lower() not in ['yes', 'y']:
            print("Training cancelled by user.")
            exit(0)
        
        print("\nStarting model training...")
        train_model(data_dir, metadata_path, num_epochs=num_epochs, batch_size=batch_size)
        
        print("\n" + "="*30)
        print("Training completed successfully!")
        print("="*30)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc() 