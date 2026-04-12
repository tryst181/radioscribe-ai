import torch
import torchxrayvision as xrv
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
import time
from app.core.config import settings

logger = logging.getLogger(__name__)

class VisionInferenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Vision Service initializing on device: {self.device}")
        
        try:
            # Load DenseNet121 from torchxrayvision
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.model.eval()
            self.model.to(self.device)
            
            self.transform = transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                 xrv.datasets.XRayResizer(224)])
            self.labels = self.model.pathologies
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def predict(self, image: Image.Image) -> dict:
        """
        Returns:
            findings: dict {label: prob}
            heatmap: np.ndarray or None
            meta: dict {processing_time: float}
        """
        start_time = time.time()
        
        try:
            # 1. Preprocess
            img = np.array(image.convert("L"))
            img = xrv.datasets.normalize(img, 255)
            if len(img.shape) == 2:
                img = img[None, ...]
                
            img = self.transform(img)
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
            img_tensor.requires_grad_() 
            
            # 2. Inference
            logger.info("[VISION] Running CheXNet Inference...")
            with torch.no_grad():
                output = self.model(img_tensor)
                
            # 3. Re-run for Gradients (Blocking Explainability)
            logger.info("[EXPLANATION] Generating Grad-CAM heatmap...")
            heatmap = self._generate_gradcam_strict(img_tensor)
            
            # 4. Get Probs
            # We can re-use output if we handled the graph correctly, 
            # but for safety/clarity let's just use the no_grad output for values.
            # Convert to probabilities (Sigmoid is usually not included in XRV DenseNet forward, it returns logits?)
            # Checking docs: "The model outputs ... logits".
            probs = torch.sigmoid(output).cpu().numpy()[0]
            
            results = {label: float(prob) for label, prob in zip(self.labels, probs)}
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "findings": results,
                "heatmap": heatmap,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Prediction logic failed: {e}")
            raise e

    def _generate_gradcam_strict(self, img_tensor) -> np.ndarray:
        """
        Generates Grad-CAM. RAISES ERROR if fails.
        Strict requirement: "If explainability fails -> NO REPORT".
        """
        try:
            # Enable grad for this pass
            with torch.set_grad_enabled(True):
                # Simple Gradient Hook
                target_layer = self.model.features.denseblock4.denselayer16 
                # XRV Structure: features -> denseblock4 -> ...
                
                gradients = []
                activations = []
                
                def save_grad(grad):
                    gradients.append(grad)
                
                def save_fwd(module, input, output):
                    activations.append(output)
                    
                # Register
                handle_g = target_layer.register_full_backward_hook(lambda m, gi, go: save_grad(go[0]))
                handle_f = target_layer.register_forward_hook(save_fwd)
                
                # Forward
                out = self.model(img_tensor)
                
                # Backward on max class
                target_idx = out.argmax(dim=1).item()
                self.model.zero_grad()
                out[0, target_idx].backward()
                
                # Cleanup
                handle_g.remove()
                handle_f.remove()
                
                if not gradients or not activations:
                    raise ValueError("GradCAM hooks captured nothing.")
                    
                # Compute CAM
                grads = gradients[0] # [1, C, H, W]
                acts = activations[0]
                
                weights = torch.mean(grads, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * acts, dim=1).squeeze()
                cam = torch.nn.functional.relu(cam)
                
                # Normalize
                cam = cam.detach().cpu().numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                
                return cam
                
        except Exception as e:
            logger.error(f"Blocking Explainability Failed: {e}")
            # As per strict rule: Raise exception to abort report
            raise RuntimeError("Explainability generation failed. Aborting report.")



vision_service = VisionInferenceService()
