"""
Tests for the Ingestion Service.
"""

import pytest
from PIL import Image
import io
import numpy as np


class TestIngestionService:
    """Tests for image ingestion and validation."""

    def test_standard_image_processing(self, sample_image_bytes):
        """Test PNG image ingestion produces valid output."""
        from app.services.ingestion.service import IngestionService

        image, metadata, patient_info = IngestionService.validate_and_process_image(
            sample_image_bytes, "test_xray.png"
        )

        assert isinstance(image, Image.Image)
        assert image.mode == "L"  # Grayscale
        assert metadata.modality == "DX"
        assert metadata.body_part == "CHEST"
        assert patient_info is None  # No patient info for standard images

    def test_jpeg_processing(self, sample_image_bytes):
        """Test JPEG image ingestion."""
        from app.services.ingestion.service import IngestionService

        # Convert PNG bytes to JPEG
        img = Image.open(io.BytesIO(sample_image_bytes))
        jpeg_buffer = io.BytesIO()
        img.save(jpeg_buffer, format="JPEG")
        jpeg_bytes = jpeg_buffer.getvalue()

        image, metadata, _ = IngestionService.validate_and_process_image(
            jpeg_bytes, "test.jpg"
        )

        assert isinstance(image, Image.Image)

    def test_unsupported_format_raises(self):
        """Test that unsupported formats raise HTTPException."""
        from app.services.ingestion.service import IngestionService
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            IngestionService.validate_and_process_image(b"fake", "test.bmp")

        assert exc_info.value.status_code == 400

    def test_invalid_image_raises(self):
        """Test that corrupted image data raises HTTPException."""
        from app.services.ingestion.service import IngestionService
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            IngestionService.validate_and_process_image(b"not_an_image", "test.png")


class TestStorageService:
    """Tests for content-addressed storage."""

    def test_save_produces_consistent_hash(self, sample_image_bytes, tmp_path):
        """Test that saving the same content twice produces the same hash."""
        from app.services.ingestion.storage import StorageService

        storage = StorageService(base_path=str(tmp_path))

        path1, hash1 = storage.save_file(sample_image_bytes, extension="png")
        path2, hash2 = storage.save_file(sample_image_bytes, extension="png")

        assert hash1 == hash2
        assert path1 == path2

    def test_different_content_different_hash(self, tmp_path):
        """Test that different content produces different hashes."""
        from app.services.ingestion.storage import StorageService

        storage = StorageService(base_path=str(tmp_path))

        _, hash1 = storage.save_file(b"content_a", extension="png")
        _, hash2 = storage.save_file(b"content_b", extension="png")

        assert hash1 != hash2
