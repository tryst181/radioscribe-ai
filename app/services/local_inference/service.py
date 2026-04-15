import base64
import hashlib
import io
import json
import logging
from typing import Any

import numpy as np
import requests
from PIL import Image

from app.core.config import settings
from app.schemas.contracts import AIReasoning
from app.schemas.response import VisionFinding

logger = logging.getLogger(__name__)


class LocalInferenceService:
    def __init__(self) -> None:
        self._embedder = None

    def _decode_base64(self, payload: str) -> bytes:
        raw = payload.split(",", 1)[1] if "," in payload else payload
        try:
            return base64.b64decode(raw, validate=True)
        except Exception as exc:
            raise ValueError("Invalid base64 payload") from exc

    def _to_pil_image(self, image_base64: str) -> Image.Image:
        return Image.open(io.BytesIO(self._decode_base64(image_base64))).convert("L")

    def _call_ollama(self, *, model: str, prompt: str, images: list[str] | None = None) -> str | None:
        try:
            response = requests.post(
                f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "images": images or [],
                    "format": "json",
                },
                timeout=settings.OLLAMA_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response")
        except Exception as exc:
            logger.warning(f"Ollama call failed: {exc}")
            return None

    async def generate_report(self, findings: list[VisionFinding]) -> AIReasoning:
        labels = [f"{f.label} ({f.probability:.2f})" for f in findings]
        if settings.LOCAL_LLM_PROVIDER == "ollama":
            prompt = (
                "You are a radiology drafting assistant. "
                "Return compact JSON with keys: impression, findings_narrative, differential_diagnosis, recommendations, criticality. "
                "Use hedging language and do not diagnose. "
                f"Findings: {labels or ['none']}."
            )
            raw = self._call_ollama(model=settings.LOCAL_TEXT_MODEL_NAME, prompt=prompt)
            if raw:
                try:
                    parsed = json.loads(raw)
                    return AIReasoning(
                        impression=parsed.get("impression", "Pending review"),
                        findings_narrative=parsed.get("findings_narrative", ""),
                        differential_diagnosis=parsed.get("differential_diagnosis", []),
                        recommendations=parsed.get("recommendations", []),
                        criticality=parsed.get("criticality", "ROUTINE"),
                    )
                except Exception:
                    logger.warning("Failed to parse ollama JSON response, falling back to deterministic response.")

        if not findings:
            return AIReasoning(
                impression="No acute radiologic abnormalities detected.",
                findings_narrative="No high-confidence abnormalities identified by the model.",
                differential_diagnosis=["Normal study"],
                recommendations=["Routine clinical correlation"],
                criticality="ROUTINE",
            )

        return AIReasoning(
            impression=f"Findings suggestive of {', '.join(f.label for f in findings)}.",
            findings_narrative=f"Model flagged: {', '.join(labels)}. Clinical review recommended.",
            differential_diagnosis=["Clinical correlation required"],
            recommendations=["Verify with patient history and radiologist read"],
            criticality="URGENT" if any(f.probability > 0.8 for f in findings) else "ROUTINE",
        )

    async def analyze_vision(self, image_base64: str, prompt: str | None = None) -> dict[str, Any]:
        analysis_text = None
        if settings.LOCAL_LLM_PROVIDER == "ollama" and prompt:
            analysis_text = self._call_ollama(
                model=settings.LOCAL_VISION_MODEL_NAME,
                prompt=prompt,
                images=[image_base64.split(",", 1)[1] if "," in image_base64 else image_base64],
            )

        # Lazy import avoids heavy torch model loading during non-vision paths/tests.
        from app.services.vision.service import vision_service

        image = self._to_pil_image(image_base64)
        vision = vision_service.predict(image)
        findings = [{"label": k, "probability": float(v)} for k, v in vision["findings"].items() if v > 0.1]
        if not analysis_text:
            if findings:
                analysis_text = f"Most probable findings: {', '.join(f['label'] for f in findings[:5])}."
            else:
                analysis_text = "No significant abnormalities above threshold."

        return {"analysis_text": analysis_text, "findings": findings, "annotations": []}

    async def detect_bodypart(self, image_base64: str) -> str:
        if settings.LOCAL_LLM_PROVIDER == "ollama":
            result = self._call_ollama(
                model=settings.LOCAL_VISION_MODEL_NAME,
                prompt="Reply with one word body part in uppercase for this radiology image.",
                images=[image_base64.split(",", 1)[1] if "," in image_base64 else image_base64],
            )
            if result:
                return result.strip().split()[0].upper()
        return "CHEST"

    async def generate_annotations(self, findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"label": f["label"], "box": [0.1, 0.1, 0.8, 0.8]}
            for f in findings
            if f.get("label")
        ]

    async def transcribe_audio(self, audio_base64: str) -> str:
        try:
            import speech_recognition as sr

            audio_bytes = self._decode_base64(audio_base64)
            recognizer = sr.Recognizer()
            with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data)
        except Exception as exc:
            logger.warning(f"Google Web Speech transcription failed: {exc}")
            return ""

    def _hash_embedding(self, text: str, dim: int) -> list[float]:
        """
        Deterministic low-cost fallback embedding when transformer models are unavailable.
        This is token-hash based and is less semantically expressive than neural embeddings.
        """
        vec = np.zeros(dim, dtype=np.float32)
        for token in text.lower().split():
            idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % dim
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32).tolist()

    async def embed_text(self, text: str) -> list[float]:
        dim = settings.LOCAL_EMBEDDING_DIM
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL_NAME)
            vec = self._embedder.encode([text], normalize_embeddings=True)[0]
            if len(vec) == dim:
                return vec.astype(np.float32).tolist()
            if len(vec) > dim:
                return np.asarray(vec[:dim], dtype=np.float32).tolist()
            padded = np.zeros(dim, dtype=np.float32)
            padded[: len(vec)] = vec
            return padded.tolist()
        except Exception as exc:
            logger.warning(f"Embedding model unavailable, using hash fallback: {exc}")
            return self._hash_embedding(text, dim)


local_inference_service = LocalInferenceService()
