# RadioScribe AI — API Reference

## Base URL

```
http://localhost:8000
```

## Endpoints

### POST `/analyze`

Analyze a medical image (Chest X-ray) and generate a structured radiology report.

**Content-Type**: `multipart/form-data`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | DICOM (.dcm), PNG, or JPEG image |

**Response Status Codes:**

| Code | Meaning |
|------|---------|
| `SUCCESS` | Analysis completed with high confidence |
| `MANUAL_REVIEW_REQUIRED` | Low confidence — requires radiologist review |
| `ERROR` | Pipeline failure |

**Response Schema:**

```json
{
  "study_id": "string (UUID)",
  "status": "SUCCESS | MANUAL_REVIEW_REQUIRED | ERROR",
  "vision_findings": [
    {
      "label": "string",
      "probability": "float (0.0-1.0)",
      "confidence": "HIGH | MODERATE | LOW"
    }
  ],
  "explainability": {
    "heatmap_url": "string | null",
    "generated_at": "datetime",
    "is_valid": "boolean"
  },
  "reasoning": {
    "impression": "string",
    "findings_narrative": "string",
    "recommendations": ["string"],
    "differential_diagnosis": ["string"]
  },
  "metadata": {
    "model_version": "string",
    "confidence_score": "float",
    "processing_time_ms": "float"
  }
}
```

---

### POST `/feedback`

Submit radiologist feedback for the active learning loop.

**Content-Type**: `application/json`

```json
{
  "report_id": "string",
  "target_type": "PREDICTION | REPORT",
  "feedback_type": "ACCEPTED | CORRECTED | REJECTED",
  "correction_details": {
    "added": ["string"],
    "removed": ["string"]
  },
  "user_id": "string | null"
}
```

**Response:**
```json
{"status": "accepted"}
```

---

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "gpu": "cuda | cpu"
}
```

---

## Interactive Documentation

When the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
