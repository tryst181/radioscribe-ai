from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager
from datetime import datetime
import uuid

from app.core.config import settings
from app.services.ingestion.service import IngestionService
from app.services.ingestion.storage import storage_service
from app.services.vision.service import vision_service
from app.services.reasoning.service import reasoning_service
from app.services.local_inference.service import local_inference_service
from app.schemas.response import AnalyzeResponse, VisionFinding, ExplainabilityOutput, ReasoningOutput, MetaData

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing RadiScribe AI Engine...")
    yield
    logger.info("Shutting down...")

from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# MOUNT STATIC FILES FOR VISUALIZATIONS
# Ensure bucket exists
import os
os.makedirs(settings.STORAGE_BUCKET_NAME, exist_ok=True)
app.mount("/visualizations", StaticFiles(directory=settings.STORAGE_BUCKET_NAME), name="visualizations")

# Feedback Schema
from typing import Literal
from pydantic import BaseModel

class FeedbackRequest(BaseModel):
    report_id: str
    target_type: Literal["PREDICTION", "REPORT"]
    feedback_type: Literal["ACCEPTED", "CORRECTED", "REJECTED"]
    correction_details: dict | None = None
    user_id: str | None = None


class VisionAnalyzeRequest(BaseModel):
    image_base64: str
    prompt: str | None = None


class BodyPartRequest(BaseModel):
    image_base64: str


class AnnotateRequest(BaseModel):
    findings: list[dict]


class TextReportRequest(BaseModel):
    transcript: str
    findings: list[dict] = []


class EscalateRequest(BaseModel):
    findings: list[dict]
    context: str | None = None


class AudioTranscribeRequest(BaseModel):
    audio_base64: str


class EmbedRequest(BaseModel):
    text: str

def map_confidence(prob: float) -> str:
    if prob > 0.85: return "HIGH"
    if prob > 0.50: return "MODERATE"
    return "LOW"

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_study(file: UploadFile = File(...)):
    """
    Strict Analysis Pipeline
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    # 1. Ingestion
    content = await file.read()
    try:
        image, metadata, _ = IngestionService.validate_and_process_image(content, file.filename)
        img_path, img_hash = storage_service.save_file(content, extension="png")
    except Exception as e:
        logger.error(f"Ingestion Error: {e}")
        return AnalyzeResponse(
            study_id=request_id,
            status="ERROR",
            vision_findings=[],
            explainability=ExplainabilityOutput(generated_at=datetime.now(), is_valid=False),
            metadata=MetaData(model_version="error", confidence_score=0.0, processing_time_ms=0)
        )

    # 2. Vision + Blocking Explainability
    try:
        vision_result = vision_service.predict(image)
        # Unpack
        findings_map = vision_result["findings"]
        heatmap = vision_result["heatmap"] # np.ndarray
        proc_time = vision_result["processing_time"]
        
        # Save Heatmap to Disk
        # We rely on storage_service to colorize (Jet) and save as PNG
        if heatmap is not None:
             rel_path = storage_service.save_heatmap(heatmap, img_hash, "gradcam.png")
             # URL construction: domain + mount_point + rel_path (which is partial)
             # Actually `save_heatmap` returns hash/hash/file.
             # Static mount is at /visualizations mapping to valid root.
             # but our storage logic uses subdirs. 
             # `StaticFiles` works on directory. So `/visualizations/ab/cd/hash.png` works.
             heatmap_url = f"/visualizations/{rel_path}"
        else:
             heatmap_url = None
        
    except RuntimeError as e:
        # Strict Rule: If explainability fails, we abort.
        logger.error(f"Blocking Safety Error: {e}")
        return AnalyzeResponse(
            study_id=request_id,
            status="MANUAL_REVIEW_REQUIRED",
            vision_findings=[],
            explainability=ExplainabilityOutput(generated_at=datetime.now(), is_valid=False),
            metadata=MetaData(model_version="chexnet-xrv-strict", confidence_score=0.0, processing_time_ms=0)
        )

    # 3. Format Findings
    formatted_findings = []
    max_prob = 0.0
    for label, prob in findings_map.items():
        if prob > 0.05: # Threshold for display
            formatted_findings.append(VisionFinding(
                label=label,
                probability=prob,
                confidence=map_confidence(prob)
            ))
            if prob > max_prob: max_prob = prob

    # 4. Success Check
    # If findings are ambiguous (e.g. all low confidence), we might flag manual review.
    # User rule: "If confidence < threshold -> MANUAL_REVIEW"
    pipeline_status = "SUCCESS"
    if max_prob > 0 and max_prob < settings.VISION_CONFIDENCE_THRESHOLD:
         pipeline_status = "MANUAL_REVIEW_REQUIRED"

    # 5. Reasoning (Always Attempt Draft, even if Low Confidence)
    # Rationale: User "needs a report I can read". The LLM System Prompt handles safety/hedging.
    # We pass the status context implicitly via the findings (low prob).
    reasoning_out = None
    try:
        # We allow reasoning even if MANUAL_REVIEW_REQUIRED, to give the user context.
        ai_reasoning = await reasoning_service.generate_report(formatted_findings) 
        reasoning_out = ReasoningOutput(
            impression=ai_reasoning.impression,
            findings_narrative=ai_reasoning.findings_narrative,
            recommendations=ai_reasoning.recommendations,
            differential_diagnosis=ai_reasoning.differential_diagnosis
        )
    except Exception as e:
        logger.error(f"Reasoning Step Failed: {e}")
        # Fallback is None, which UI handles.

    return AnalyzeResponse(
        study_id=request_id,
        status=pipeline_status,
        vision_findings=formatted_findings,
        explainability=ExplainabilityOutput(
            heatmap_url=heatmap_url,
            generated_at=datetime.now(),
            is_valid=True
        ),
        reasoning=reasoning_out,
        metadata=MetaData(
            model_version="chexnet-xrv-v1",
            confidence_score=max_prob,
            processing_time_ms=proc_time
        )
    )

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Active Learning Endpoint.
    Stores feedback for future training loops.
    """
    # In a real app, write to DB. Here we log.
    logger.info(f"FEEDBACK RECEIVED: {feedback.model_dump_json()}")
    return {"status": "accepted"}


@app.post("/vision/analyze")
async def vision_analyze(req: VisionAnalyzeRequest):
    try:
        result = await local_inference_service.analyze_vision(req.image_base64, req.prompt)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/vision/bodypart")
async def vision_bodypart(req: BodyPartRequest):
    try:
        body_part = await local_inference_service.detect_bodypart(req.image_base64)
        return {"body_part": body_part}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/vision/annotate")
async def vision_annotate(req: AnnotateRequest):
    annotations = await local_inference_service.generate_annotations(req.findings)
    return {"annotations": annotations}


@app.post("/text/report")
async def text_report(req: TextReportRequest):
    findings = [
        VisionFinding(
            label=item.get("label", "Unknown"),
            probability=float(item.get("probability", 0.0)),
            confidence=map_confidence(float(item.get("probability", 0.0))),
        )
        for item in req.findings
    ]
    result = await local_inference_service.generate_report(findings)
    return result.model_dump()


@app.post("/text/escalate")
async def text_escalate(req: EscalateRequest):
    findings = [
        VisionFinding(
            label=item.get("label", "Unknown"),
            probability=float(item.get("probability", 0.0)),
            confidence=map_confidence(float(item.get("probability", 0.0))),
        )
        for item in req.findings
    ]
    result = await local_inference_service.generate_report(findings)
    return {"escalated_report": result.model_dump(), "context": req.context}


@app.post("/audio/transcribe")
async def audio_transcribe(req: AudioTranscribeRequest):
    transcript = await local_inference_service.transcribe_audio(req.audio_base64)
    return {"transcript": transcript}


@app.post("/embed")
async def embed_text(req: EmbedRequest):
    embedding = await local_inference_service.embed_text(req.text)
    return {"embedding": embedding, "dimension": len(embedding)}

@app.get("/health")
def health_check():
    return {"status": "ok", "legacy_status": "healthy", "gpu": vision_service.device.type}
