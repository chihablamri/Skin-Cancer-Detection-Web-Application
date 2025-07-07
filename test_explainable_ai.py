import unittest
import torch
import numpy as np
import sys
import os
from PIL import Image
import torchvision.transforms as transforms

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explainable_ai import GradCAM

class TestExplainableAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests"""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.batch_size = 1
        cls.input_size = (3, 224, 224)
        
        # Create a dummy model for testing
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super(DummyModel, self).__init__()
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
                self.pool = torch.nn.MaxPool2d(2)
                self.fc = torch.nn.Linear(64 * 56 * 56, 7)
                
            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
                
        cls.model = DummyModel().to(cls.device)
        cls.gradcam = GradCAM(cls.model)
        
    def test_gradcam_initialization(self):
        """Test if GradCAM initializes correctly"""
        self.assertIsNotNone(self.gradcam)
        self.assertEqual(self.gradcam.model, self.model)
        
    def test_heatmap_generation(self):
        """Test if heatmap generation works correctly"""
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        heatmap = self.gradcam.generate_heatmap(x, 0)
        
        self.assertIsInstance(heatmap, np.ndarray)
        self.assertEqual(len(heatmap.shape), 2)
        self.assertTrue(np.all(heatmap >= 0) and np.all(heatmap <= 1))
        
    def test_visualization(self):
        """Test if visualization works correctly"""
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        heatmap = self.gradcam.generate_heatmap(x, 0)
        
        # Convert tensor to PIL Image for visualization
        img = transforms.ToPILImage()(x[0].cpu())
        visualization = self.gradcam.visualize_explanation(img, heatmap)
        
        self.assertIsInstance(visualization, np.ndarray)
        self.assertEqual(len(visualization.shape), 3)
        self.assertEqual(visualization.shape[2], 3)  # RGB image
        
    def test_attention_percentage(self):
        """Test if attention percentage calculation works correctly"""
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        heatmap = self.gradcam.generate_heatmap(x, 0)
        
        # Test with different thresholds
        thresholds = [0.3, 0.5, 0.7]
        for threshold in thresholds:
            attention_percentage = np.mean(heatmap > threshold) * 100
            self.assertTrue(0 <= attention_percentage <= 100)
            
    def test_multiple_classes(self):
        """Test if GradCAM works for multiple classes"""
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        
        for class_idx in range(7):
            heatmap = self.gradcam.generate_heatmap(x, class_idx)
            self.assertIsInstance(heatmap, np.ndarray)
            self.assertEqual(len(heatmap.shape), 2)
            
    def test_invalid_inputs(self):
        """Test if GradCAM handles invalid inputs correctly"""
        with self.assertRaises(ValueError):
            self.gradcam.generate_heatmap(torch.randn(2, 3, 100, 100), 0)  # Wrong input size
            
        with self.assertRaises(ValueError):
            self.gradcam.generate_heatmap(torch.randn(1, 3, 224, 224), 10)  # Invalid class index
            
    def test_gradient_flow(self):
        """Test if gradients flow correctly in GradCAM"""
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        x.requires_grad = True
        
        heatmap = self.gradcam.generate_heatmap(x, 0)
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.all(torch.isfinite(x.grad)))
        
    def test_heatmap_resolution(self):
        """Test if heatmap resolution matches input"""
        x = torch.randn(self.batch_size, *self.input_size).to(self.device)
        heatmap = self.gradcam.generate_heatmap(x, 0)
        
        # Heatmap should be downsampled by the model's pooling layers
        expected_size = (self.input_size[1] // 2, self.input_size[2] // 2)
        self.assertEqual(heatmap.shape, expected_size)

if __name__ == '__main__':
    unittest.main() 