import io
import uuid
import pydicom
import numpy as np
from PIL import Image
from fastapi import UploadFile, HTTPException
from app.schemas.contracts import ImageMetadata, PatientInfo
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class IngestionService:
    @staticmethod
    def validate_and_process_image(file_content: bytes, filename: str) -> tuple[Image.Image, ImageMetadata, PatientInfo | None]:
        logger.info(f"[INGESTION] Processing file: {filename} ({len(file_content)} bytes)")
        """
        Validates the input image, extracts metadata, and returns a normalized PIL Image.
        Supports DICOM, PNG, JPG.
        """
        file_ext = filename.split('.')[-1].lower()
        
        if file_ext in ['dcm', 'dicom']:
             return IngestionService._process_dicom(file_content)
        elif file_ext in ['png', 'jpg', 'jpeg']:
             return IngestionService._process_standard_image(file_content, filename)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

    @staticmethod
    def _process_dicom(content: bytes) -> tuple[Image.Image, ImageMetadata, PatientInfo]:
        try:
            ds = pydicom.dcmread(io.BytesIO(content))
            
            # 1. Modality Validation
            modality = ds.get("Modality", "Unknown")
            if modality not in ['DX', 'CR']:
                # We might allow CT later, but Master Prompt says primarily Chest X-rays
                logger.warning(f"Ingested DICOM with non-standard modality for this engine: {modality}")
                # For strict safety, we could reject. For now, we log specific warning.
                # User constraint: "Validate modality (Chest X-ray only)"
                if modality not in ['DX', 'CR']:
                     raise HTTPException(status_code=400, detail=f"Invalid Modality: {modality}. Only DX (Digital X-Ray) or CR (Computed Radiography) are accepted.")

            # 2. Extract Metadata
            metadata = ImageMetadata(
                study_id=str(ds.get("StudyInstanceUID", "unknown")),
                series_id=str(ds.get("SeriesInstanceUID", "unknown")),
                instance_id=str(ds.get("SOPInstanceUID", "unknown")),
                modality=modality,
                body_part=str(ds.get("BodyPartExamined", "CHEST")),
                view_position=str(ds.get("ViewPosition", ""))
            )
            
            # Anonymize/Extract Patient Info
            patient_info = PatientInfo(
                patient_id=str(ds.get("PatientID", "unknown")), # In real prod, hash this immediately
                age=None, # Extract from AgeString if needed
                gender=str(ds.get("PatientSex", ""))
            )
            
            # 3. Pixel Data Normalization
            # Handle VOI LUT/Windowing if present, else just raw
            # This is a simplified conversion. Real medical imaging needs robust windowing.
            if "PixelData" not in ds:
                raise HTTPException(status_code=400, detail="DICOM file contains no PixelData.")
            
            pixel_array = ds.pixel_array
            
            # Normalize to 0-255 uint8
            if pixel_array.max() != pixel_array.min():
                 pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255.0
            
            # Handle Photometric Interpretation (Monochrome1 means inverted)
            if ds.get("PhotometricInterpretation") == "MONOCHROME1":
                pixel_array = 255.0 - pixel_array
                
            image = Image.fromarray(pixel_array.astype(np.uint8))
            
            return image, metadata, patient_info

        except pydicom.errors.InvalidDicomError:
             raise HTTPException(status_code=400, detail="Invalid DICOM file.")
        except Exception as e:
            logger.error(f"DICOM processing error: {e}")
            raise HTTPException(status_code=500, detail="Error processing DICOM image.")

    @staticmethod
    def _process_standard_image(content: bytes, filename: str) -> tuple[Image.Image, ImageMetadata, PatientInfo | None]:
        try:
            image = Image.open(io.BytesIO(content)).convert("L") # Convert to grayscale
            
            # Generate dummy metadata for non-DICOM
            import uuid
            metadata = ImageMetadata(
                study_id=f"ext_{uuid.uuid4()}",
                series_id=f"ser_{uuid.uuid4()}",
                instance_id=f"img_{uuid.uuid4()}",
                modality="DX", # Assume DX for uploaded PNGs of X-rays
                body_part="CHEST"
            )
            
            return image, metadata, None
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            raise HTTPException(status_code=400, detail="Invalid image file.")
