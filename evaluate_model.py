import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
from PIL import Image
import os
from model_pytorch import SkinCancerModelPyTorch
from explainable_ai import GradCAM
import pandas as pd

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SkinCancerModelPyTorch()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device

def evaluate_model(model, test_loader, device):
    all_preds = []
    all_labels = []
    total_time = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_samples += images.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average inference time
    avg_inference_time = total_time / total_samples
    
    return np.array(all_preds), np.array(all_labels), avg_inference_time

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_metrics(metrics):
    # Plot accuracy and loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['accuracy'], label='Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['loss'], label='Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_metrics.png')
    plt.close()

def main():
    # Load model
    model_path = 'best_skin_cancer_model.pth'
    model, device = load_model(model_path)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load test data
    test_dir = 'dataverse_files/val'
    test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate model
    predictions, true_labels, avg_inference_time = evaluate_model(model, test_loader, device)
    
    # Get class names
    classes = test_dataset.classes
    
    # Generate classification report
    report = classification_report(true_labels, predictions, target_names=classes, output_dict=True)
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, classes)
    
    # Calculate and print metrics
    accuracy = np.mean(predictions == true_labels)
    
    print("\nModel Evaluation Results:")
    print("=" * 50)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Average Inference Time: {avg_inference_time*1000:.2f} ms per image")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=classes))
    
    # Save results to CSV
    results_df = pd.DataFrame(report).transpose()
    results_df.to_csv('model_evaluation_results.csv')
    
    # Save inference time
    with open('inference_time.txt', 'w') as f:
        f.write(f"Average inference time: {avg_inference_time*1000:.2f} ms per image")

if __name__ == "__main__":
    main() 