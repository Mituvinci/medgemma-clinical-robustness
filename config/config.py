"""
Configuration management for MedGemma Clinical Robustness Assistant.
Loads environment variables and provides centralized config access.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys - Active
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    google_cloud_project: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    google_application_credentials: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

    # API Keys - Future (add when available)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    claude_api_key: str = os.getenv("CLAUDE_API_KEY", "")

    # Active Model Selection
    active_model: str = os.getenv("ACTIVE_MODEL", "medgemma")  # Options: medgemma, gemini, gpt4, claude

    # Model Configuration
    medgemma_model_id: str = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-27b-it")
    gemini_model_id: str = os.getenv("GEMINI_MODEL_ID", "gemini-pro-latest")
    openai_model_id: str = os.getenv("OPENAI_MODEL_ID", "gpt-4-turbo")
    claude_model_id: str = os.getenv("CLAUDE_MODEL_ID", "claude-3-opus-20240229")

    # ChromaDB Configuration
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "dermatology_guidelines")

    # Embedding Model
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Chunk Configuration
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: str = os.getenv("LOG_DIR", "./logs")

    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    guidelines_dir: Path = data_dir / "guidelines"
    cases_dir: Path = data_dir / "cases"
    local_cache_dir: Path = project_root / "local_cache"

    # Uppercase aliases for backwards compatibility
    @property
    def BASE_DIR(self) -> Path:
        return self.project_root

    @property
    def CHROMA_PERSIST_DIR(self) -> str:
        return self.chroma_persist_dir

    @property
    def EMBEDDING_MODEL(self) -> str:
        return self.embedding_model

    @property
    def CHUNK_SIZE(self) -> int:
        return self.chunk_size

    @property
    def CHUNK_OVERLAP(self) -> int:
        return self.chunk_overlap

    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

def validate_config():
    """Validate that required configuration is present."""
    errors = []

    if not settings.huggingface_api_key:
        errors.append("HUGGINGFACE_API_KEY is not set")

    if not settings.gemini_api_key:
        errors.append("GEMINI_API_KEY is not set")

    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    print("Configuration validated successfully")
    return True
