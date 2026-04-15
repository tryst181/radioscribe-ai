from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # App
    PROJECT_NAME: str = "RadiScribe AI Engine"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    
    # Database (Supabase)
    SUPABASE_URL: str = Field("https://example.supabase.co", env="SUPABASE_URL")
    SUPABASE_KEY: str = Field("local-dev-key", env="SUPABASE_KEY")
    
    # ML Models
    MODEL_PATH_CHEXNET: str = "models/chexnet.pth"
    MODEL_PATH_SWIN: str = "models/swin.pth"
    VISION_CONFIDENCE_THRESHOLD: float = 0.75
    
    # Local / OSS inference config
    LOCAL_LLM_PROVIDER: str = "mock"  # mock | ollama
    LOCAL_TEXT_MODEL_NAME: str = "llama3:8b-instruct-q4_K_M"
    LOCAL_VISION_MODEL_NAME: str = "llava:7b-v1.6-mistral-q4_K_M"
    LOCAL_EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
    LOCAL_EMBEDDING_DIM: int = 768
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT_SECONDS: int = 45
    
    # Storage
    STORAGE_BUCKET_NAME: str = "radiscribe-images"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
