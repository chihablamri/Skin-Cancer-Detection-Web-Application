import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
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
        
        # Load a pre-trained ResNet50 model
        self.base_model = models.resnet50(pretrained=True)
        
        # Freeze early layers
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Replace the final fully connected layer
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
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
        
        # Initialize loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.base_model.fc.parameters(), lr=0.001)
        
    def forward(self, x):
        # Ensure input is on same device as model
        x = x.to(self.device)
        return self.base_model(x)
    
    def compile(self):
        """Compile the model (PyTorch doesn't need compilation, but we keep this for consistency)"""
        print("Model ready for training")
        print(f"Total trainable parameters: {sum(p.numel() for p in self.base_model.fc.parameters())}")
    
    def fit(self, train_loader, val_loader, num_epochs=10):
        """
        Train the model with enhanced progress reporting
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Number of training epochs
            
        Returns:
            dict: Training history
        """
        print(f"\n{'='*20} TRAINING INFORMATION {'='*20}")
        print(f"Training on {len(train_loader.dataset)} images")
        print(f"Validating on {len(val_loader.dataset)} images")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Training iterations per epoch: {len(train_loader)}")
        print(f"Validation iterations per epoch: {len(val_loader)}")
        print(f"Total training iterations: {len(train_loader) * num_epochs}")
        print('='*60)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        total_start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            self.train()
            running_loss = 0.0
            running_corrects = 0
            
            # Progress bar for training
            train_pbar = tqdm(train_loader, desc=f"Training", unit="batch")
            batch_losses = []
            batch_accs = []
            
            for inputs, labels in train_pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                batch_loss = loss.item()
                batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
                
                batch_losses.append(batch_loss)
                batch_accs.append(batch_acc.item())
                
                # Update progress bar with latest metrics
                train_pbar.set_postfix({
                    'loss': f"{batch_loss:.4f}", 
                    'acc': f"{batch_acc.item():.4f}",
                    'avg_loss': f"{sum(batch_losses[-50:]) / min(len(batch_losses), 50):.4f}"
                })
                
                running_loss += batch_loss * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc.item())
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            val_corrects = 0
            
            # Progress bar for validation
            val_pbar = tqdm(val_loader, desc=f"Validating", unit="batch")
            
            with torch.no_grad():
                for inputs, labels in val_pbar:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    batch_val_loss = loss.item()
                    batch_val_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
                    
                    val_pbar.set_postfix({
                        'val_loss': f"{batch_val_loss:.4f}", 
                        'val_acc': f"{batch_val_acc.item():.4f}"
                    })
                    
                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)
            
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_corrects.double() / len(val_loader.dataset)
            
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc.item())
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Time: {epoch_time:.1f}s | ETA: {epoch_time * (num_epochs-epoch-1):.1f}s")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, 'best_skin_cancer_model.pth')
                print(f"New best model saved! Validation accuracy: {val_acc:.4f}")
            
            # Plot current learning curves
            if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
                self._plot_learning_curves(history, epoch+1)
        
        total_time = time.time() - total_start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
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