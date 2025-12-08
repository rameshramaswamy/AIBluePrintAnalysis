import os
import logging
from pathlib import Path
from typing import Dict, List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Info
    APP_NAME: str = "BlueprintAI-Brain"
    ENV: str = "production"
    LOG_LEVEL: str = "INFO"

    # Paths (Absolute paths for container safety)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_PATH: Path = BASE_DIR / "data"
    RAW_DATA_PATH: Path = DATA_PATH / "raw" / "cubicasa5k"
    PROCESSED_PATH: Path = DATA_PATH / "processed"
    MODEL_ARTIFACTS_PATH: Path = BASE_DIR / "artifacts"
    
    # Processing Config
    TILE_SIZE: int = 640
    TILE_OVERLAP: float = 0.2
    NUM_WORKERS: int = os.cpu_count() or 4  # Auto-detect CPU cores

    # Model Config
    DEFAULT_MODEL_VERSION: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45

    # Class Map (Immutable)
    CLASS_MAP: Dict[str, int] = {
        "Wall": 0, "Window": 1, "Door": 2, "Room": 3, 
        "Toilet": 4, "Sink": 5, "Electrical": 6
    }
    
    # Infrastructure Config
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Storage Config (S3 / MinIO)
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_BUCKET_NAME: str = "blueprints"
    AWS_ACCESS_KEY: str = "minioadmin"
    AWS_SECRET_KEY: str = "minioadmin"
    AWS_REGION: str = "us-east-1"
    
    # Task Config
    TASK_QUEUE_NAME: str = "blueprint_tasks"
    API_SECRET_KEY: str = "change_me_in_prod"

    DATABASE_URL: str = "postgresql://user:password@localhost:5432/blueprint_db"

    class Config:
        env_file = ".env"

settings = Settings()

# Configure Enterprise Logging (JSON format preferred for Prod, Console for Dev)
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(settings.APP_NAME)