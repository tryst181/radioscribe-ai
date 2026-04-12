from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

class PatientInfo(BaseModel):
    patient_id: str = Field(..., description="Anonymized Patient ID / Hash")
    age: Optional[int] = None
    gender: Optional[str] = None

class ImageMetadata(BaseModel):
    study_id: str
    series_id: str
    instance_id: str
    modality: Literal["CR", "DX", "CT", "MR"] = "DX"
    body_part: str = "CHEST"
    view_position: Optional[str] = None # PA, AP, LATERAL

class AnalysisRequest(BaseModel):
    """
    Main entry point for the API.
    Client sends image URL (or base64 - though URL preferred for efficiency)
    and metadata.
    """
    image_url: Optional[HttpUrl] = None
    image_base64: Optional[str] = None # For direct uploads
    metadata: ImageMetadata
    patient_info: Optional[PatientInfo] = None

class VisionFinding(BaseModel):
    label: str  # e.g., "Pneumonia"
    probability: float # 0.0 to 1.0
    confidence: float # Model's confidence score
    severity: Optional[str] = None # Mild, Moderate, Severe
    location: Optional[List[float]] = None # [x, y, w, h] normalized
    heatmap_url: Optional[str] = None

class AIReasoning(BaseModel):
    """
    Output from Gemini 2.5 Pro (Language Model).
    """
    impression: str
    findings_narrative: str
    differential_diagnosis: List[str]
    recommendations: List[str]
    criticality: Literal["ROUTINE", "URGENT", "CRITICAL"]

class FinalReport(BaseModel):
    report_id: str
    created_at: datetime
    model_version: str
    vision_findings: List[VisionFinding]
    reasoning: AIReasoning
    disclaimer: str = "AI-generated report. Not a diagnosis. Requires Radiologist review."
    
    class Config:
        json_schema_extra = {
            "example": {
                "report_id": "rpt_12345",
                "vision_findings": [
                    {"label": "Pleural Effusion", "probability": 0.89, "confidence": 0.92}
                ],
                "reasoning": {
                    "impression": "Probable right-sided pleural effusion.",
                    "criticality": "URGENT"
                }
            }
        }
