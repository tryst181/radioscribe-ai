"""
Tests for the Vision Service components.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestConfidenceMapping:
    """Test the confidence mapping function."""

    def test_high_confidence(self):
        """Probabilities above 0.85 should map to HIGH."""
        from main import map_confidence
        assert map_confidence(0.90) == "HIGH"
        assert map_confidence(0.99) == "HIGH"
        assert map_confidence(0.86) == "HIGH"

    def test_moderate_confidence(self):
        """Probabilities between 0.50 and 0.85 should map to MODERATE."""
        from main import map_confidence
        assert map_confidence(0.75) == "MODERATE"
        assert map_confidence(0.51) == "MODERATE"

    def test_low_confidence(self):
        """Probabilities below 0.50 should map to LOW."""
        from main import map_confidence
        assert map_confidence(0.49) == "LOW"
        assert map_confidence(0.10) == "LOW"
        assert map_confidence(0.0) == "LOW"


class TestGradCAM:
    """Tests for Grad-CAM implementation."""

    def test_overlay_heatmap_dimensions(self, sample_pil_image):
        """Test that heatmap overlay preserves image dimensions."""
        from app.services.explanation.gradcam import GradCAM

        heatmap = np.random.rand(7, 7).astype(np.float32)
        overlay = GradCAM.overlay_heatmap(sample_pil_image, heatmap)

        assert overlay.size == sample_pil_image.size

    def test_overlay_returns_rgb(self, sample_pil_image):
        """Test that overlay converts grayscale to RGB."""
        from app.services.explanation.gradcam import GradCAM

        heatmap = np.random.rand(7, 7).astype(np.float32)
        overlay = GradCAM.overlay_heatmap(sample_pil_image, heatmap)

        assert overlay.mode == "RGB"


class TestResponseSchemas:
    """Test Pydantic response models."""

    def test_vision_finding_creation(self):
        """Test VisionFinding model creation."""
        from app.schemas.response import VisionFinding

        finding = VisionFinding(
            label="Pneumonia",
            probability=0.85,
            confidence="HIGH"
        )
        assert finding.label == "Pneumonia"
        assert finding.probability == 0.85

    def test_analyze_response_creation(self):
        """Test AnalyzeResponse model creation."""
        from app.schemas.response import AnalyzeResponse, ExplainabilityOutput, MetaData
        from datetime import datetime

        response = AnalyzeResponse(
            study_id="test-123",
            status="SUCCESS",
            vision_findings=[],
            explainability=ExplainabilityOutput(
                generated_at=datetime.now(),
                is_valid=True
            ),
            metadata=MetaData(
                model_version="test",
                confidence_score=0.9,
                processing_time_ms=100.0
            )
        )
        assert response.status == "SUCCESS"
