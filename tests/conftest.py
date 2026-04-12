"""
RadioScribe AI Engine — Test Suite
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import io
import numpy as np
from PIL import Image


@pytest.fixture
def sample_image_bytes():
    """Generate a synthetic grayscale image mimicking a chest X-ray."""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224), dtype=np.uint8), mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_pil_image():
    """Generate a sample PIL Image."""
    return Image.fromarray(np.random.randint(0, 255, (224, 224), dtype=np.uint8), mode="L")
