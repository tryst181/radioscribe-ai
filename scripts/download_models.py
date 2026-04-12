"""
RadioScribe AI Engine — Model Download Script

Downloads pre-trained model weights required for inference.
Run this script after cloning the repository.
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path


MODELS_DIR = Path(__file__).parent.parent / "models"

MODELS = {
    "densenet121-res224-all": {
        "description": "DenseNet-121 trained on multiple chest X-ray datasets (TorchXRayVision)",
        "note": "This model is downloaded automatically by TorchXRayVision on first use. "
                "This script pre-downloads it to avoid delays during inference.",
    }
}


def download_models():
    """Pre-download TorchXRayVision model weights."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RadioScribe AI — Model Download")
    print("=" * 60)

    try:
        import torchxrayvision as xrv
        print("\n[1/2] Loading DenseNet-121 weights via TorchXRayVision...")
        print("      This downloads ~50MB on first run.\n")

        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        print(f"      ✅ Model loaded successfully")
        print(f"      Pathologies: {model.pathologies}")
        print(f"      Parameters: {sum(p.numel() for p in model.parameters()):,}")

    except ImportError:
        print("❌ torchxrayvision not installed. Run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Download failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ All models ready. You can now run the server:")
    print("   uvicorn main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    download_models()
