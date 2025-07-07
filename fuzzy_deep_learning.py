import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skfuzzy import control as ctrl
import skfuzzy as fuzz

class FuzzyLayer(nn.Module):
    def __init__(self, num_features, num_classes):
        super(FuzzyLayer, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Initialize fuzzy membership functions
        self.membership_functions = nn.Parameter(torch.randn(num_features, 3))  # 3 membership functions per feature
        
    def forward(self, x):
        # Convert input to fuzzy membership values
        fuzzy_values = torch.sigmoid((x.unsqueeze(-1) - self.membership_functions) / 0.1)
        return fuzzy_values

class FuzzyDeepLearning(nn.Module):
    def __init__(self, num_classes=7):
        super(FuzzyDeepLearning, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fuzzy layer
        self.fuzzy_layer = FuzzyLayer(128 * 28 * 28, num_classes)
        
        # Final classification layer
        self.fc = nn.Linear(128 * 28 * 28 * 3, num_classes)  # 3 membership functions per feature
        
        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # CNN feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 28 * 28)
        
        # Apply fuzzy logic
        fuzzy_values = self.fuzzy_layer(x)
        
        # Reshape for final classification
        x = fuzzy_values.view(-1, 128 * 28 * 28 * 3)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

    def get_fuzzy_rules(self):
        """Get the current fuzzy rules for explanation"""
        rules = []
        for i in range(self.num_features):
            membership_values = self.fuzzy_layer.membership_functions[i].detach().cpu().numpy()
            rules.append({
                'feature': i,
                'low': membership_values[0],
                'medium': membership_values[1],
                'high': membership_values[2]
            })
        return rules

def train_fuzzy_deep_learning(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Training function for Fuzzy Deep Learning model"""
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
            torch.save(model.state_dict(), 'best_fuzzy_deep_learning_model.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return history 