import torch
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    """
    Grad-CAM implementation for model explainability
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        # Get the last convolutional layer (we know it's in layer4 for ResNet)
        self.target_layer = self.model.base_model.layer4[-1].conv3
        
        # Register hooks
        self.gradients = None
        self.activations = None
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
        
    def generate_heatmap(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        output[0, class_idx].backward()
        
        # Get weights
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3])[0]
        
        # Weight the channels by corresponding gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        
        # ReLU on top of the heatmap
        heatmap = np.maximum(heatmap.cpu().numpy(), 0)
        
        # Normalize the heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
        
        return heatmap, class_idx
    
    def explain(self, image_path, target_class=None):
        """
        Generate visual explanation for the model's prediction
        """
        # Read and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # Prepare input tensor
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor / 255.0
        
        # Normalize with ImageNet stats
        normalize = lambda x: (x - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = normalize(img_tensor)
        
        # Move to same device as model
        img_tensor = img_tensor.to(self.model.device)
        
        # Generate heatmap
        heatmap, pred_class = self.generate_heatmap(img_tensor, target_class)
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (224, 224))
        
        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        overlay = heatmap_colored * 0.4 + img * 0.6
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # Save visualizations
        heatmap_path = image_path.replace('.', '_heatmap.')
        overlay_path = image_path.replace('.', '_overlay.')
        
        cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR))
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Calculate attention percentage
        attention_percentage = np.mean(heatmap > 0.5) * 100
        
        # Generate explanation text
        if attention_percentage > 20:
            explanation = "The model is analyzing large areas of the lesion"
        elif attention_percentage > 5:
            explanation = "The model is focusing on specific features of the lesion"
        else:
            explanation = "The model is examining very detailed aspects of the lesion"
            
        return {
            'heatmap_path': heatmap_path.split('/')[-1],
            'overlay_path': overlay_path.split('/')[-1],
            'pred_class': pred_class,
            'explanation': explanation
        } 