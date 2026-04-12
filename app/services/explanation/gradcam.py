import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    """
    Grad-CAM implementation specifically for DenseNet-121.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class_index=None):
        """
        Generates Class Activation Map.
        :param input_tensor: Preprocessed image tensor (1, C, H, W)
        :param target_class_index: Class index to optimize for. If None, uses max prediction.
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class_index is None:
            target_class_index = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        score = output[0, target_class_index]
        score.backward()
        
        # Generate CAM
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling of gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight the activations
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU on heatmap
        heatmap = F.relu(heatmap)
        
        # Normalize
        heatmap = heatmap.detach().cpu().numpy()
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        return heatmap, target_class_index

    @staticmethod
    def overlay_heatmap(image_pil: Image.Image, heatmap: np.ndarray, alpha=0.4) -> Image.Image:
        """
        Overlays heatmap on original image.
        """
        img_np = np.array(image_pil.convert("RGB"))
        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        
        # Colorize
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Overlay
        overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)
        return Image.fromarray(overlay)
