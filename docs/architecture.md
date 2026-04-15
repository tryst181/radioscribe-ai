# RadioScribe AI Engine — System Architecture

## Overview

RadioScribe AI is a **safety-first** medical imaging analysis engine built on a strict pipeline architecture. Every component is designed with clinical safety constraints that make it suitable for integration into radiology workflows.

## Pipeline Flow

```
Image Upload → Ingestion → Vision (CheXNet) → Explainability (Grad-CAM) → Reasoning (Local/Open-Weight LLM) → Report
```

### Key Design Decisions

1. **Blocking Explainability**: Unlike typical ML services where explainability is optional, RadioScribe treats it as a **hard dependency**. If Grad-CAM fails to generate a heatmap, the entire pipeline aborts. This ensures no prediction is ever served without visual evidence.

2. **Confidence Triage**: The system doesn't just return predictions — it triages them. High-confidence findings produce `SUCCESS` status, while ambiguous results trigger `MANUAL_REVIEW_REQUIRED`, explicitly signaling to the downstream UI that human review is mandatory.

3. **Hedged Reasoning**: The LLM reasoning layer uses carefully constrained system prompts that enforce medical hedging language. The system never says "the patient has pneumonia" — it says "findings are suggestive of pneumonia."

## Component Details

### Ingestion Service (`app/services/ingestion/`)

- **DICOM Parser**: Extracts StudyInstanceUID, Modality, ViewPosition, PhotometricInterpretation
- **Modality Validation**: Rejects non-chest modalities (only DX/CR accepted)
- **MONOCHROME1 Handling**: Inverts pixel data for correct radiographic display
- **Content-Addressed Storage**: SHA256 hashing for deduplication with sharded directory structure

### Vision Service (`app/services/vision/`)

- **Model**: DenseNet-121 via TorchXRayVision (`densenet121-res224-all`)
- **Preprocessing**: Grayscale conversion → XRV normalization → Center crop → Resize 224x224
- **Output**: 14-class probability vector (Sigmoid activation) + Grad-CAM heatmap
- **Pathologies**: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax

### Grad-CAM Explainability (`app/services/explanation/`)

- **Target Layer**: `features.denseblock4.denselayer16` (final dense block)
- **Method**: Register forward/backward hooks → compute weighted activation map → ReLU → normalize
- **Output**: 2D float array (0-1) representing model attention intensity

### Reasoning Service (`app/services/reasoning/`)

- **Model**: Local/Open-Weight LLM (or Ollama endpoint)
- **Input**: Structured findings array with labels, probabilities, and confidence levels
- **Output**: Structured JSON with impression, findings narrative, recommendations, differential diagnosis
- **Fallback**: Rule-based mock generator when API is unavailable
- **Safety**: System prompt enforces mandatory hedging, prohibits diagnostic assertions

## Database Design

The PostgreSQL schema follows clinical data management principles:

- **Immutability**: Predictions are never modified after creation
- **Traceability**: Full audit log for regulatory compliance
- **Active Learning**: Feedback table links corrections to specific predictions
- **Row-Level Security**: Supabase RLS policies restrict access to service role

## Security Considerations

- All API keys are loaded from environment variables (never hardcoded)
- DICOM patient data should be anonymized before upload in production
- Heatmaps are stored with the same content-addressed scheme as originals
- No PHI (Protected Health Information) is logged
