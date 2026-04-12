import asyncio
import sys
import argparse
from pathlib import Path
import requests
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RadiScribe-CLI")

# Default Server URL
API_URL = "http://localhost:8000/analyze"
SAMPLE_XRAY_URL = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg"

async def main(file_path: str, use_api: bool):
    logger.info(f"--- RadiScribe AI Engine: CLI{' (API Mode)' if use_api else ' (Local Mode)'} ---")
    
    # 1. Load/Download Image
    content = None
    filename = "sample.jpg"
    
    if file_path == "demo":
        logger.info("Downloading sample X-ray...")
        response = requests.get(SAMPLE_XRAY_URL)
        content = response.content
        filename = "sample_p.jpg"
    else:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return
        with open(path, "rb") as f:
            content = f.read()
        filename = path.name

    logger.info(f"Processing: {filename}")

    if use_api:
        # --- API MODE ---
        logger.info(f"Sending request to {API_URL}...")
        try:
            files = {"file": (filename, content, "image/jpeg")}
            resp = requests.post(API_URL, files=files)
            
            if resp.status_code == 200:
                report = resp.json()
                print_report(report)
            else:
                logger.error(f"API Error {resp.status_code}: {resp.text}")
                
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to server at {API_URL}. Is it running? (Try 'uvicorn main:app')")
            
    else:
        # --- LOCAL MODE ---
        logger.info("Running in LOCAL MODE (Direct Service Call)...")
        from app.services.ingestion.service import IngestionService
        from app.services.vision.service import vision_service
        from app.services.reasoning.service import reasoning_service
        from app.schemas.response import VisionFinding
        
        # 1. Ingest
        path = Path(file_path)
        with open(path, "rb") as f:
            content = f.read()
            
        processed_image, metadata, _ = IngestionService.validate_and_process_image(content, filename)
        
        # 2. Vision
        # Returns {findings: {label: prob}, heatmap: array, ...}
        raw_result = vision_service.predict(processed_image)
        raw_findings = raw_result["findings"]
        
        print("\n--- RAW MODEL OUTPUT (Probabilities) ---")
        for k, v in raw_findings.items():
            if v is None:
                print(f"{k}: None")
            else:
                print(f"{k}: {float(v):.4f}")
        print("----------------------------------------\n")
        
        # Transform for Reasoning
        # Use a low threshold just to see what the reasoning service WOULD see if we were stricter
        # But actually reasoning service takes specific findings. 
        # Let's filter like the real app (e.g. > 0.5)
        
        threshold = 0.5
        vision_findings_objs = []
        for label, prob in raw_findings.items():
            if prob > threshold:
                vision_findings_objs.append(
                    VisionFinding(label=label, probability=prob, confidence="HIGH" if prob > 0.7 else "MODERATE")
                )
        
        # 3. Reasoning
        report = await reasoning_service.generate_report(vision_findings_objs)
        
        # Construct Report Dict
        full_report = {
            "status": "SUCCESS" if vision_findings_objs else "MANUAL_REVIEW_REQUIRED",
            "vision_findings": [f.model_dump() for f in vision_findings_objs],
            "reasoning": report.model_dump() if report else None,
            "metadata": {"model_version": "LOCAL", "confidence_score": max(raw_findings.values()) if raw_findings else 0.0},
            "explainability": {"is_valid": True, "heatmap_url": "N/A (Local Mode)"}
        }
        print_report(full_report)

def print_report(report):
    status = report.get("status")
    print("\n" + "="*60)
    print(f"RADIOLOGY REPORT STATUS: {status}")
    print("="*60)
    
    if status == "ERROR":
        print("Analysis Failed. Please try again.")
        return

    findings = report.get("vision_findings", [])
    print(f"FINDINGS ({len(findings)} detected):")
    for f in findings:
        # New schema: label, probability, confidence (str)
        print(f" - {f['label']}: {f['probability']:.1%} [{f['confidence']}]")
    
    explainability = report.get("explainability", {})
    if explainability.get("is_valid"):
        print(f"Visual Evidence: {explainability.get('heatmap_url')}")
    else:
        print("Visual Evidence: FAILED (Safety Hazard)")

    reasoning = report.get("reasoning")
    if reasoning:
        print("\nCLINICAL IMPRESSION:")
        print(f"{reasoning.get('impression')}\n")
        print(f"NARRATIVE:\n{reasoning.get('findings_narrative')}\n")
        
        recs = reasoning.get('recommendations', [])
        if recs:
            print(f"RECOMMENDATIONS:\n{', '.join(recs)}")
    else:
        print("\n(No clinical text generated - Manual Review Required)")
        
    meta = report.get("metadata", {})
    print("-" * 60)
    print(f"Model: {meta.get('model_version')} | Confidence: {meta.get('confidence_score'):.2f}")
    print(f"Processing Time: {meta.get('processing_time_ms'):.0f}ms")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RadiScribe Analysis")
    parser.add_argument("path", nargs="?", default="demo", help="Path to X-ray image (or 'demo')")
    parser.add_argument("--api", action="store_true", help="Use running API server instead of local lib")
    args = parser.parse_args()
    
    asyncio.run(main(args.path, args.api))
