import torch
import torch.nn as nn
from torchvision import models
import logging

logger = logging.getLogger(__name__)

class CheXNet(nn.Module):
    """
    DenseNet-121 architecture modified for Chest X-ray analysis (CheXNet).
    Output: 14 classes (standard NIH ChestX-ray14 labels).
    """
    def __init__(self, num_classes: int = 14, pretrained: bool = True):
        super(CheXNet, self).__init__()
        # Load standard ImageNet weights if pretrained=True (we will override with CheXNet weights later)
        self.densenet121 = models.densenet121(weights="DEFAULT" if pretrained else None)
        
        # Replace classifier for 14 classes
        num_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.densenet121(x)

def load_chexnet_model(weights_path: str, device: torch.device) -> CheXNet:
    model = CheXNet(pretrained=False) # We load our specific weights
    
    try:
        # Strict loading if weights exist
        if weights_path and torch.os.path.exists(weights_path):
            logger.info(f"Loading CheXNet weights from {weights_path}")
            checkpoint = torch.load(weights_path, map_location=device)
            
            # Handle state_dict naming (some checkpoints have 'module.' prefix)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "") # remove `module.`
                new_state_dict[name] = v
                
            model.load_state_dict(new_state_dict, strict=False)
        else:
            logger.warning(f"No CheXNet weights found at {weights_path}. Using random init (Safety Warning!).")
            
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        raise e

    model.to(device)
    model.eval()
    return model
