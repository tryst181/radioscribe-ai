import base64

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
