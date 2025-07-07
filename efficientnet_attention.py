import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv1(x)
        attention = torch.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention

class EfficientNetAttention(nn.Module):
    def __init__(self, num_classes=7):
        super(EfficientNetAttention, self).__init__()
        # Load EfficientNet-B3
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b3')
        
        # Get the number of features from the last layer
        in_features = self.efficientnet._fc.in_features
        
        # Remove the original classifier
        self.efficientnet._fc = nn.Identity()
        
        # Add attention module
        self.attention = AttentionModule(in_features)
        
        # Add new classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract feature map from EfficientNet backbone (before pooling)
        features = self.efficientnet.extract_features(x)  # Shape: [B, C, H, W]
        
        # Apply attention on the feature map
        features = self.attention(features)  # Shape: [B, C, H, W]
        
        # Global pooling to get a 1x1 feature map
        pooled = torch.mean(features, dim=[2, 3])  # Shape: [B, C]
        
        # Classify
        output = self.classifier(pooled)
        return output

    def get_attention_map(self, x):
        """Get attention map for visualization"""
        features = self.efficientnet(x)
        attention_map = self.attention(features)
        return attention_map

def train_efficientnet_attention(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Training function for EfficientNet with Attention"""
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_efficientnet_attention_model.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return history 