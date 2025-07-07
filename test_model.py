import unittest
import torch
import torch.nn as nn
from torchvision import transforms
import sys
import os

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_pytorch import SkinCancerModelPyTorch

class TestSkinCancerModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.model = SkinCancerModelPyTorch(num_classes=7).to(cls.device)
        cls.batch_size = 4
        cls.input_size = (3, 224, 224)
        
    def test_model_initialization(self):
        """Test if model initializes correctly"""
        self.assertIsInstance(self.model, nn.Module)
        self.assertEqual(self.model.base_model.fc[-1].out_features, 7)
        
    def test_model_forward_pass(self):
        """Test if model forward pass works correctly"""
        x = torch.randn(self.batch_size, *self.input_size).to(cls.device)
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, 7))
        
    def test_model_layers(self):
        """Test if all required layers are present"""
        self.assertTrue(hasattr(self.model.base_model, 'conv1'))
        self.assertTrue(hasattr(self.model.base_model, 'bn1'))
        self.assertTrue(hasattr(self.model.base_model, 'layer1'))
        
    def test_model_output_range(self):
        """Test if model outputs are in expected range"""
        x = torch.randn(self.batch_size, *self.input_size).to(cls.device)
        output = self.model(x)
        self.assertTrue(torch.all(torch.isfinite(output)))
        
    def test_model_parameters(self):
        """Test if model parameters are properly initialized"""
        for name, param in self.model.named_parameters():
            self.assertTrue(torch.all(torch.isfinite(param)))
            
    def test_model_dropout(self):
        """Test if dropout is working during training and evaluation"""
        self.model.train()
        x = torch.randn(self.batch_size, *self.input_size).to(cls.device)
        output1 = self.model(x)
        
        self.model.eval()
        output2 = self.model(x)
        
        # Outputs should be different in training mode due to dropout
        self.assertFalse(torch.allclose(output1, output2))
        
    def test_model_gradient_flow(self):
        """Test if gradients flow properly through the model"""
        x = torch.randn(self.batch_size, *self.input_size).to(cls.device)
        x.requires_grad = True
        output = self.model(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.all(torch.isfinite(x.grad)))
        
    def test_model_device_placement(self):
        """Test if model is on the correct device"""
        self.assertEqual(next(self.model.parameters()).device, self.device)
        
    def test_model_input_normalization(self):
        """Test if model handles normalized inputs correctly"""
        transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        x = torch.randn(self.batch_size, *self.input_size).to(cls.device)
        x_normalized = transform(x)
        output = self.model(x_normalized)
        self.assertEqual(output.shape, (self.batch_size, 7))

if __name__ == '__main__':
    unittest.main() 