from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict
from datetime import datetime

class VisionFinding(BaseModel):
    label: str
    probability: float
    confidence: Literal["HIGH", "MODERATE", "LOW"]

class ExplainabilityOutput(BaseModel):
    heatmap_url: Optional[str] = None
    generated_at: datetime
    is_valid: bool

class ReasoningOutput(BaseModel):
    impression: str
    findings_narrative: str
    recommendations: List[str] = []
    differential_diagnosis: List[str] = []

class MetaData(BaseModel):
    model_version: str
    confidence_score: float
    processing_time_ms: float

class AnalyzeResponse(BaseModel):
    study_id: str
    status: Literal["SUCCESS", "MANUAL_REVIEW_REQUIRED", "ERROR"]
    
    vision_findings: List[VisionFinding]
    explainability: ExplainabilityOutput
    
    # Optional because if status != SUCCESS, reasoning might be skipped
    reasoning: Optional[ReasoningOutput] = None
    
    metadata: MetaData
