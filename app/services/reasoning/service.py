"""
Reasoning Service — LLM-Powered Clinical Report Generation

Uses Google Gemini to transform structured vision findings into
human-readable radiology report drafts with mandatory safety hedging.

The system prompt enforces:
- No diagnostic assertions (only "suggestive of", "consistent with")
- Mandatory uncertainty language for all AI findings
- No hallucinations — empty input produces a normal report
- Low-confidence findings mentioned only as "equivocal"
"""

from app.core.config import settings
from app.schemas.response import VisionFinding as ResponseFinding
from app.schemas.contracts import AIReasoning
import json
import logging
import os

logger = logging.getLogger(__name__)

# System prompt is loaded from environment or uses a safety-constrained default.
# The full clinical prompt is not committed to version control.
# See docs/architecture.md for the safety constraints it enforces.
SYSTEM_PROMPT = os.getenv("RADIOSCRIBE_SYSTEM_PROMPT", (
    "You are RadiScribe, an AI assistant for radiologists. "
    "Your role is DRAFTING SUPPORT ONLY. You are NOT a doctor. "
    "Generate preliminary radiology report text for human review. "
    "NEVER diagnose. Always use hedging language. Do not hallucinate findings. "
    "Return JSON with: impression, findings_narrative, differential_diagnosis, recommendations."
))


class ReasoningService:
    def __init__(self):
        if settings.GOOGLE_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GOOGLE_API_KEY)
                self.model = genai.GenerativeModel(
                    settings.GEMINI_MODEL_NAME,
                    system_instruction=SYSTEM_PROMPT
                )
                logger.info("Gemini Reasoning Service Initialized.")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}. Using mock mode.")
                self.model = None
        else:
            logger.warning("GOOGLE_API_KEY not set. Reasoning Service will return mocks.")
            self.model = None

    async def generate_report(self, findings: list[ResponseFinding]) -> AIReasoning:
        """
        Generate a structured radiology report from vision findings.

        Args:
            findings: List of VisionFinding objects with labels, probabilities, and confidence

        Returns:
            AIReasoning: Structured report with impression, narrative, DDx, and recommendations
        """
        if not self.model:
            return self._mock_response(findings)

        findings_text = json.dumps([f.model_dump() for f in findings], indent=2)
        prompt = f"Generate a structured radiology draft for the following findings (Chest X-Ray):\n{findings_text}"

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )

            raw_text = response.text
            parsed = json.loads(raw_text)

            return AIReasoning(
                impression=parsed.get("impression", "Pending Review"),
                findings_narrative=parsed.get("findings_narrative", ""),
                differential_diagnosis=parsed.get("differential_diagnosis", []),
                recommendations=parsed.get("recommendations", []),
                criticality=parsed.get("criticality", "ROUTINE")
            )

        except Exception as e:
            logger.error(f"Gemini Inference Failed: {e}")
            return self._mock_response(findings)

    def _mock_response(self, findings: list[ResponseFinding]) -> AIReasoning:
        """
        Rule-based fallback when Gemini is unavailable.
        Produces safe, hedged output suitable for review.
        """
        if not findings:
            return AIReasoning(
                impression="No acute radiologic abnormalities detected.",
                findings_narrative=(
                    "The lungs are clear. Heart size is within normal limits. "
                    "No pleural effusion or pneumothorax is identified. "
                    "Osseous structures appear intact."
                ),
                differential_diagnosis=["Normal Study"],
                recommendations=["Routine clinical follow-up"],
                criticality="ROUTINE"
            )

        labels = [f.label for f in findings]
        return AIReasoning(
            impression=f"Findings suggestive of {', '.join(labels)}.",
            findings_narrative=(
                f"AI analysis detects probabilities for: {', '.join(labels)}. "
                "Please review specific regions."
            ),
            differential_diagnosis=["Clinical correlation required"],
            recommendations=["Verify findings with clinical history"],
            criticality="URGENT" if any(f.probability > 0.8 for f in findings) else "ROUTINE"
        )


reasoning_service = ReasoningService()
