import torch
import torch.nn as nn
from .efficientnet_attention import EfficientNetAttention
from .cnn_gradcam import CNNWithGradCAM
from .fuzzy_deep_learning import FuzzyDeepLearning
from .hybrid_cnn_rules import HybridCNNRules

class ModelManager:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {
            'efficientnet': None,
            'cnn_gradcam': None,
            'fuzzy': None,
            'hybrid': None
        }
        self.current_model = None
        self.model_names = {
            'efficientnet': 'EfficientNetB3 with Attention',
            'cnn_gradcam': 'CNN with Grad-CAM',
            'fuzzy': 'Fuzzy Logic + Deep Learning',
            'hybrid': 'Hybrid CNN + Rule-based'
        }
        
    def load_model(self, model_type):
        """Load a specific model"""
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
            
        if self.models[model_type] is None:
            if model_type == 'efficientnet':
                self.models[model_type] = EfficientNetAttention()
                self.models[model_type].load_state_dict(torch.load('best_efficientnet_attention_model.pth', map_location=self.device))
            elif model_type == 'cnn_gradcam':
                self.models[model_type] = CNNWithGradCAM()
                self.models[model_type].load_state_dict(torch.load('best_cnn_gradcam_model.pth', map_location=self.device))
            elif model_type == 'fuzzy':
                self.models[model_type] = FuzzyDeepLearning()
                self.models[model_type].load_state_dict(torch.load('best_fuzzy_deep_learning_model.pth', map_location=self.device))
            elif model_type == 'hybrid':
                self.models[model_type] = HybridCNNRules()
                self.models[model_type].load_state_dict(torch.load('best_hybrid_cnn_rules_model.pth', map_location=self.device))
                
            self.models[model_type].to(self.device)
            self.models[model_type].eval()
            
        self.current_model = self.models[model_type]
        return self.current_model
    
    def predict(self, image_tensor):
        """Make prediction using the current model"""
        if self.current_model is None:
            raise ValueError("No model loaded. Please load a model first.")
            
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.current_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            
        return prediction.item(), probabilities[0].cpu().numpy()
    
    def get_explanation(self, image_tensor):
        """Get model-specific explanation"""
        if self.current_model is None:
            raise ValueError("No model loaded. Please load a model first.")
            
        model_type = next(key for key, value in self.models.items() if value == self.current_model)
        
        if model_type == 'efficientnet':
            return self.current_model.get_attention_map(image_tensor)
        elif model_type == 'cnn_gradcam':
            return self.current_model.get_gradcam(image_tensor)
        elif model_type == 'fuzzy':
            return self.current_model.get_fuzzy_rules()
        elif model_type == 'hybrid':
            return self.current_model.get_rules()
            
    def get_available_models(self):
        """Get list of available models"""
        return list(self.model_names.keys())
    
    def get_model_name(self, model_type):
        """Get the display name of a model"""
        return self.model_names.get(model_type, "Unknown Model") 