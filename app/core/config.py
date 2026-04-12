from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # App
    PROJECT_NAME: str = "RadiScribe AI Engine"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    
    # Database (Supabase)
    SUPABASE_URL: str = Field(..., env="SUPABASE_URL")
    SUPABASE_KEY: str = Field(..., env="SUPABASE_KEY")
    
    # ML Models
    MODEL_PATH_CHEXNET: str = "models/chexnet.pth"
    MODEL_PATH_SWIN: str = "models/swin.pth"
    VISION_CONFIDENCE_THRESHOLD: float = 0.75
    
    # Google Gemini
    GOOGLE_API_KEY: str | None = Field(None, env="GOOGLE_API_KEY")
    GEMINI_MODEL_NAME: str = "gemini-2.5-pro"
    
    # Storage
    STORAGE_BUCKET_NAME: str = "radiscribe-images"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
