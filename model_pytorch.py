import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet50_Weights
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class SkinCancerModelPyTorch(nn.Module):
    def __init__(self, num_classes=7, force_cpu=False):
        """
        Initialize the skin cancer detection model using PyTorch
        
        Args:
            num_classes (int): Number of output classes
            force_cpu (bool): If True, force the model to use CPU regardless of CUDA availability
        """
        super(SkinCancerModelPyTorch, self).__init__()
        
        # Load a pre-trained ResNet50 model with updated weights parameter
        self.base_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze early layers
        for param in list(self.base_model.parameters())[:-20]:  # Keep last few layers trainable
            param.requires_grad = False
            
        # Replace the final fully connected layer with a more robust classifier
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
        
        # Set device
        if force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU Model: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
        self.to(self.device)
        
        # Initialize loss function with class weights
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.base_model(x)
        
    def compile(self):
        """Compile the model with optimizer and learning rate scheduler"""
        self.optimizer = optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2
        )
        
    def fit(self, train_loader, val_loader, num_epochs=20):  # Increased epochs
        """
        Train the model with improved training loop
        """
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for inputs, labels in train_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                train_bar.set_postfix({
                    'loss': f'{train_loss/train_total:.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
                for inputs, labels in val_bar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    val_bar.set_postfix({
                        'loss': f'{val_loss/val_total:.4f}',
                        'acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            # Calculate metrics
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                }, 'best_skin_cancer_model.pth')
            
            # Plot training progress
            self._plot_learning_curves(history, epoch + 1)
            
        return history
    
    def _plot_learning_curves(self, history, current_epoch):
        """Plot learning curves during training"""
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='Training')
        plt.plot(history['val_acc'], label='Validation')
        plt.title(f'Model Accuracy (Epoch {current_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='Training')
        plt.plot(history['val_loss'], label='Validation')
        plt.title(f'Model Loss (Epoch {current_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(f'training_progress_epoch_{current_epoch}.png')
        plt.close() 