import base64
import asyncio
import types
import sys

from app.services.local_inference.service import LocalInferenceService


class TestLocalInferenceService:
    def test_hash_embedding_dimension_and_norm(self):
        service = LocalInferenceService()
        vector = service._hash_embedding("pleural effusion pleural", dim=768)

        assert len(vector) == 768
        assert abs(sum(v * v for v in vector) - 1.0) < 1e-4

    def test_decode_base64_with_data_url_prefix(self):
        service = LocalInferenceService()
        payload = base64.b64encode(b"hello").decode("utf-8")
        decoded = service._decode_base64(f"data:audio/wav;base64,{payload}")

        assert decoded == b"hello"

    def test_hash_embedding_empty_text_returns_zero_vector(self):
        service = LocalInferenceService()
        vector = service._hash_embedding("", dim=8)
        assert vector == [0.0] * 8

    def test_transcribe_audio_invalid_payload_returns_empty(self):
        service = LocalInferenceService()
        result = asyncio.run(service.transcribe_audio("not-base64"))
        assert result == ""

    def test_detect_bodypart_fallback_default(self):
        service = LocalInferenceService()
        result = asyncio.run(service.detect_bodypart("dGVzdA=="))
        assert result == "CHEST"

    def test_analyze_vision_fallback_without_ollama(self, monkeypatch):
        service = LocalInferenceService()
        fake_module = types.ModuleType("app.services.vision.service")

        class _FakeVision:
            def predict(self, image):
                return {"findings": {"Pneumonia": 0.8, "Effusion": 0.05}, "processing_time": 1, "heatmap": None}

        fake_module.vision_service = _FakeVision()
        monkeypatch.setitem(sys.modules, "app.services.vision.service", fake_module)

        png = base64.b64encode(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00:\x7e\x9bU\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
        ).decode("utf-8")
        output = asyncio.run(service.analyze_vision(png))

        assert "analysis_text" in output
        assert output["findings"][0]["label"] == "Pneumonia"

    def test_embed_text_falls_back_when_sentence_transformer_unavailable(self, monkeypatch):
        service = LocalInferenceService()

        broken_module = types.ModuleType("sentence_transformers")

        class _Broken:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("model unavailable")

        broken_module.SentenceTransformer = _Broken
        monkeypatch.setitem(sys.modules, "sentence_transformers", broken_module)

        vector = asyncio.run(service.embed_text("pneumonia finding"))
        assert len(vector) == 768
