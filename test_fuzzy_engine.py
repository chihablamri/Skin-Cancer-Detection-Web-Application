import unittest
import torch
import numpy as np
import sys
import os

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experimental_models.models.fuzzy_deep_learning import FuzzyDeepLearning

class TestFuzzyEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.batch_size = 4
        cls.input_size = (3, 224, 224)
        cls.num_classes = 7
        
    def test_fuzzy_deep_learning_initialization(self):
        """Test if FuzzyDeepLearning model initializes correctly"""
        model = FuzzyDeepLearning(num_classes=self.num_classes).to(self.device)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'fuzzy_layer'))
        
    def test_fuzzy_deep_learning_forward(self):
        """Test if FuzzyDeepLearning forward pass works correctly"""
        model = FuzzyDeepLearning(num_classes=self.num_classes).to(self.device)
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        output = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertTrue(torch.all(torch.isfinite(output)))
        
    def test_fuzzy_rules(self):
        """Test if fuzzy rules are properly applied"""
        model = FuzzyDeepLearning(num_classes=self.num_classes).to(self.device)
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        
        # Get intermediate features through forward pass
        output = model(x)
        
        # Apply softmax to get probabilities
        output_probs = torch.nn.functional.softmax(output, dim=1)
        
        # Test fuzzy rule outputs
        self.assertTrue(torch.all(torch.isfinite(output_probs)))
        self.assertTrue(torch.all(output_probs >= 0) and torch.all(output_probs <= 1))
        self.assertTrue(torch.allclose(output_probs.sum(dim=1), torch.ones(self.batch_size).to(self.device)))
        
    def test_model_components(self):
        """Test if all model components are present and working"""
        model = FuzzyDeepLearning(num_classes=self.num_classes).to(self.device)
        
        # Test CNN components
        self.assertTrue(hasattr(model, 'conv1'))
        self.assertTrue(hasattr(model, 'conv2'))
        self.assertTrue(hasattr(model, 'conv3'))
        
        # Test fuzzy components
        self.assertTrue(hasattr(model, 'fuzzy_layer'))
        
        # Test classification components
        self.assertTrue(hasattr(model, 'fc'))
        
    def test_gradient_flow(self):
        """Test if gradients flow properly through the model"""
        model = FuzzyDeepLearning(num_classes=self.num_classes).to(self.device)
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        x.requires_grad = True
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.all(torch.isfinite(x.grad)))
        
    def test_model_dropout(self):
        """Test if dropout is working during training and evaluation"""
        model = FuzzyDeepLearning(num_classes=self.num_classes).to(self.device)
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        
        model.train()
        output1 = model(x)
        
        model.eval()
        output2 = model(x)
        
        # Outputs should be different in training mode due to dropout
        self.assertFalse(torch.allclose(output1, output2))
        
    def test_model_regularization(self):
        """Test if regularization is properly applied"""
        model = FuzzyDeepLearning(num_classes=self.num_classes).to(self.device)
        
        # Test if batch normalization is present
        self.assertTrue(any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()))
        
        # Test if dropout is present
        self.assertTrue(any(isinstance(m, torch.nn.Dropout) for m in model.modules()))

if __name__ == '__main__':
    unittest.main() 