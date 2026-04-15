"""
Reasoning Service — Local/Open-Weight Clinical Report Generation
"""

from app.schemas.contracts import AIReasoning
from app.schemas.response import VisionFinding as ResponseFinding
from app.services.local_inference.service import local_inference_service


class ReasoningService:
    async def generate_report(self, findings: list[ResponseFinding]) -> AIReasoning:
        return await local_inference_service.generate_report(findings)


reasoning_service = ReasoningService()
