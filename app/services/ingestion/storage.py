import hashlib
import os
import shutil
from pathlib import Path
from app.core.config import settings

class StorageService:
    def __init__(self, base_path: str = "storage_bucket"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_file(self, content: bytes, extension: str = "png") -> tuple[str, str]:
        """
        Saves content immutably using its SHA256 hash as the filename.
        Returns: (storage_path, file_hash)
        """
        file_hash = hashlib.sha256(content).hexdigest()
        
        # Directory structure: storage/ab/cd/abcdef...png (to avoid huge flat lists)
        subdir = self.base_path / file_hash[:2] / file_hash[2:4]
        subdir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{file_hash}.{extension}"
        file_path = subdir / filename
        
        # Identity check: If file exists, we verify hash (optional), but mostly we just assume it's same.
        if not file_path.exists():
            with open(file_path, "wb") as f:
                f.write(content)
        
        # Return relative path for DB
        relative_path = f"{file_hash[:2]}/{file_hash[2:4]}/{filename}"
        return relative_path, file_hash

    def get_file_path(self, relative_path: str) -> Path:
        return self.base_path / relative_path

    def save_heatmap(self, heatmap_data: "np.ndarray", study_hash: str, filename_suffix: str = "gradcam.png") -> str:
        """
        Saves heatmap array as a PNG image.
        """
        from PIL import Image
        import numpy as np
        import cv2

        # Directory same as study
        subdir = self.base_path / study_hash[:2] / study_hash[2:4]
        subdir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{study_hash}_{filename_suffix}"
        file_path = subdir / filename
        
        # heatmap_data is 0-1 float. Convert to uint8 color.
        # Use simple colormap mapping if just 1 channel, or assume it's already colored?
        # VisionService returns 1-channel or 3-channel? 
        # VisionService._generate_gradcam_strict returns 1-channel float 0-1.
        # We should colorize it here or in Vision Service?
        # Let's simple save as grayscale if 1 channel, or apply colormap here.
        
        hm_uint8 = np.uint8(255 * heatmap_data)
        # Apply Jet usually looks best for medical overlays, but requires CV2
        hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        
        # Save
        cv2.imwrite(str(file_path), hm_color)
        
        return f"{study_hash[:2]}/{study_hash[2:4]}/{filename}"

storage_service = StorageService(settings.STORAGE_BUCKET_NAME)
