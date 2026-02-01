"""
Unit tests for configuration management.
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Settings, settings


class TestConfiguration:
    """Test suite for configuration loading and validation."""

    def test_settings_instance(self):
        """Test that settings instance is created."""
        assert settings is not None
        assert isinstance(settings, Settings)

    def test_default_values(self):
        """Test that default configuration values are set."""
        assert settings.medgemma_model_id == "google/medgemma-27b-it"
        assert settings.gemini_model_id == "gemini-pro"
        assert settings.chunk_size == 512
        assert settings.chunk_overlap == 50

    def test_chroma_config(self):
        """Test ChromaDB configuration."""
        assert settings.chroma_collection_name == "dermatology_guidelines"
        assert "chroma" in settings.chroma_persist_dir

    def test_paths_exist(self):
        """Test that configured paths exist or can be created."""
        assert settings.project_root.exists()
        # Data directories may not exist yet, that's OK
        assert isinstance(settings.data_dir, Path)
        assert isinstance(settings.guidelines_dir, Path)

    def test_embedding_model_config(self):
        """Test embedding model configuration."""
        assert "sentence-transformers" in settings.embedding_model or "all-MiniLM" in settings.embedding_model

    def test_log_level(self):
        """Test logging configuration."""
        assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class TestAPIKeyValidation:
    """Test API key validation (requires .env file)."""

    def test_api_key_fields_exist(self):
        """Test that API key fields exist in settings."""
        assert hasattr(settings, "huggingface_api_key")
        assert hasattr(settings, "gemini_api_key")

    @pytest.mark.skipif(
        not settings.huggingface_api_key,
        reason="HUGGINGFACE_API_KEY not set in environment"
    )
    def test_huggingface_key_set(self):
        """Test HuggingFace API key is set (if available)."""
        assert len(settings.huggingface_api_key) > 0

    @pytest.mark.skipif(
        not settings.gemini_api_key,
        reason="GEMINI_API_KEY not set in environment"
    )
    def test_gemini_key_set(self):
        """Test Gemini API key is set (if available)."""
        assert len(settings.gemini_api_key) > 0
