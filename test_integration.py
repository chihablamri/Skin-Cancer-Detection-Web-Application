import unittest
import torch
import numpy as np
import sys
import os
from PIL import Image
import torchvision.transforms as transforms

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_pytorch import SkinCancerModelPyTorch
from explainable_ai import GradCAM
from experimental_models.models.fuzzy_deep_learning import FuzzyDeepLearning

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.batch_size = 4
        cls.input_size = (3, 224, 224)
        cls.num_classes = 7
        
        # Initialize models
        cls.cnn_model = SkinCancerModelPyTorch(num_classes=cls.num_classes).to(cls.device)
        cls.fuzzy_model = FuzzyDeepLearning(num_classes=cls.num_classes).to(cls.device)
        
        # Initialize GradCAM
        cls.gradcam = GradCAM(cls.cnn_model)
        
    def test_model_integration(self):
        """Test if all models work together correctly"""
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        
        # Test CNN model
        cnn_output = self.cnn_model(x)
        self.assertEqual(cnn_output.shape, (self.batch_size, self.num_classes))
        
        # Test Fuzzy model
        fuzzy_output = self.fuzzy_model(x)
        self.assertEqual(fuzzy_output.shape, (self.batch_size, self.num_classes))
        
    def test_explainability_integration(self):
        """Test if explainability works with both models"""
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        
        # Test GradCAM with CNN model
        heatmap = self.gradcam.generate_heatmap(x, 0)
        self.assertIsInstance(heatmap, np.ndarray)
        
        # Test visualization
        img = transforms.ToPILImage()(x[0].cpu())
        visualization = self.gradcam.visualize_explanation(img, heatmap)
        self.assertIsInstance(visualization, np.ndarray)
        
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction pipeline"""
        # Create a sample image
        img = torch.randn(1, *self.input_size).to(self.device)
        
        # Get predictions from both models
        cnn_pred = self.cnn_model(img)
        fuzzy_pred = self.fuzzy_model(img)
        
        # Get GradCAM explanation
        heatmap = self.gradcam.generate_heatmap(img, 0)
        
        # Test if all components work together
        self.assertTrue(torch.all(torch.isfinite(cnn_pred)))
        self.assertTrue(torch.all(torch.isfinite(fuzzy_pred)))
        self.assertTrue(np.all(np.isfinite(heatmap)))
        
    def test_model_consistency(self):
        """Test if models produce consistent results"""
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        
        # Get predictions
        cnn_pred1 = self.cnn_model(x)
        cnn_pred2 = self.cnn_model(x)
        
        # Predictions should be consistent for same input
        self.assertTrue(torch.allclose(cnn_pred1, cnn_pred2))
        
    def test_error_handling(self):
        """Test error handling across components"""
        # Test with invalid input size
        with self.assertRaises(ValueError):
            x = torch.randn(self.batch_size, 3, 100, 100).to(self.device)
            self.cnn_model(x)
            
        # Test with invalid class index
        with self.assertRaises(ValueError):
            x = torch.randn(self.batch_size, *self.input_size).to(self.device)
            self.gradcam.generate_heatmap(x, 10)
            
    def test_memory_management(self):
        """Test memory management across components"""
        # Test if models can handle large batches
        large_batch = torch.randn(32, *self.input_size).to(self.device)
        
        # Should not raise memory error
        self.cnn_model(large_batch)
        self.fuzzy_model(large_batch)
        
    def test_model_saving_loading(self):
        """Test model saving and loading integration"""
        # Save models
        torch.save(self.cnn_model.state_dict(), 'test_cnn_model.pth')
        torch.save(self.fuzzy_model.state_dict(), 'test_fuzzy_model.pth')
        
        # Load models
        new_cnn_model = SkinCancerModelPyTorch(num_classes=self.num_classes).to(self.device)
        new_fuzzy_model = FuzzyDeepLearning(num_classes=self.num_classes).to(self.device)
        
        new_cnn_model.load_state_dict(torch.load('test_cnn_model.pth'))
        new_fuzzy_model.load_state_dict(torch.load('test_fuzzy_model.pth'))
        
        # Test if loaded models work correctly
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        self.assertTrue(torch.allclose(self.cnn_model(x), new_cnn_model(x)))
        self.assertTrue(torch.allclose(self.fuzzy_model(x), new_fuzzy_model(x)))
        
        # Clean up
        os.remove('test_cnn_model.pth')
        os.remove('test_fuzzy_model.pth')
        
    def test_performance_metrics(self):
        """Test performance metrics integration"""
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        y = torch.randint(0, self.num_classes, (self.batch_size,)).to(self.device)
        
        # Test both models
        cnn_output = self.cnn_model(x)
        fuzzy_output = self.fuzzy_model(x)
        
        # Calculate loss
        criterion = torch.nn.CrossEntropyLoss()
        cnn_loss = criterion(cnn_output, y)
        fuzzy_loss = criterion(fuzzy_output, y)
        
        self.assertTrue(torch.isfinite(cnn_loss))
        self.assertTrue(torch.isfinite(fuzzy_loss))

if __name__ == '__main__':
    unittest.main() 