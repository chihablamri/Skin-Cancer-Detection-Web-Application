import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class RuleBasedLayer(nn.Module):
    def __init__(self, num_features, num_classes):
        super(RuleBasedLayer, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Rule weights for each class
        self.rule_weights = nn.Parameter(torch.randn(num_classes, num_features))
        
        # Rule thresholds
        self.thresholds = nn.Parameter(torch.randn(num_classes, num_features))
        
    def forward(self, x):
        # Apply rule-based logic
        rule_outputs = []
        for i in range(self.num_classes):
            # Calculate rule satisfaction for each feature
            rule_satisfaction = torch.sigmoid((x - self.thresholds[i]) * self.rule_weights[i])
            # Combine rules using product t-norm
            rule_output = torch.prod(rule_satisfaction, dim=1)
            rule_outputs.append(rule_output)
        
        # Stack rule outputs
        return torch.stack(rule_outputs, dim=1)

class HybridCNNRules(nn.Module):
    def __init__(self, num_classes=7):
        super(HybridCNNRules, self).__init__()
        
        # Load DenseNet-169
        self.densenet = models.densenet169(pretrained=True)
        
        # Remove the original classifier
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()
        
        # Rule-based layer
        self.rule_layer = RuleBasedLayer(num_features, num_classes)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(num_features, num_features // 8),
            nn.ReLU(),
            nn.Linear(num_features // 8, num_features),
            nn.Sigmoid()
        )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Get features from DenseNet
        features = self.densenet(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Apply rule-based layer
        rule_output = self.rule_layer(features)
        
        # Final classification
        output = self.classifier(features)
        
        # Combine CNN and rule-based outputs
        final_output = output + rule_output
        
        return final_output

    def get_rules(self):
        """Get the current rules for explanation"""
        rules = []
        for i in range(self.num_classes):
            weights = self.rule_layer.rule_weights[i].detach().cpu().numpy()
            thresholds = self.rule_layer.thresholds[i].detach().cpu().numpy()
            rules.append({
                'class': i,
                'weights': weights,
                'thresholds': thresholds
            })
        return rules

def train_hybrid_cnn_rules(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Training function for Hybrid CNN + Rules model"""
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
            torch.save(model.state_dict(), 'best_hybrid_cnn_rules_model.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return history 