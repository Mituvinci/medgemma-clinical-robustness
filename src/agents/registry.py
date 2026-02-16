"""
Model Registry

Central registry for all available LLM models.
Maps model names to their adapters and requirements.
"""

from typing import Dict, Any, List
import logging
from config.config import settings

from src.agents.models.medgemma_adapter import MedGemmaAdapter
from src.agents.models.gemini_adapter import GeminiAdapter
from src.agents.models.vertex_medgemma_adapter import VertexMedGemmaAdapter
from src.agents.models.openai_adapter import OpenAIAdapter
from src.agents.models.claude_adapter import ClaudeAdapter

logger = logging.getLogger(__name__)


# Model Registry: Maps model names to adapters
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ====== ACTIVE MODELS (API keys available) ======

    "medgemma": {
        "adapter": MedGemmaAdapter,
        "status": "active",
        "description": "Google MedGemma-27B-IT - Medical specialist model (27B params)",
        "provider": "huggingface",
        "requires": "HUGGINGFACE_API_KEY",
        "model_id": settings.medgemma_27b_model_id,
        "strengths": [
            "Medical knowledge",
            "Clinical reasoning",
            "Diagnostic accuracy"
        ],
        "use_cases": [
            "Clinical diagnosis",
            "Medical literature review",
            "Patient case analysis"
        ]
    },

    "medgemma-4b": {
        "adapter": MedGemmaAdapter,
        "status": "active",
        "description": "Google MedGemma-4B-IT - Lightweight medical model (4B params)",
        "provider": "huggingface",
        "requires": "HUGGINGFACE_API_KEY",
        "model_id": settings.medgemma_4b_model_id,
        "strengths": [
            "Medical knowledge",
            "Fast inference",
            "Lower memory requirements"
        ],
        "use_cases": [
            "Clinical diagnosis",
            "Rapid medical consultation",
            "Resource-constrained environments"
        ]
    },

    "gemini": {
        "adapter": GeminiAdapter,
        "status": "active",
        "description": "Google Gemini Pro - General purpose SOTA model",
        "provider": "google",
        "requires": "GEMINI_API_KEY",
        "model_id": settings.gemini_model_id,
        "strengths": [
            "General knowledge",
            "Reasoning",
            "Instruction following"
        ],
        "use_cases": [
            "General medical questions",
            "Complex reasoning",
            "Multimodal analysis (future)"
        ]
    },

    # ====== VERTEX AI MODELS (Deployed endpoints) ======

    "medgemma-vertex": {
        "adapter": VertexMedGemmaAdapter,
        "status": "active",
        "description": "MedGemma-1.5-4B-IT via Vertex AI endpoint",
        "provider": "vertex_ai",
        "requires": "GOOGLE_APPLICATION_CREDENTIALS",
        "model_id": "medgemma-1.5-4b-it",
        "project_id": settings.google_cloud_project,
        "region": "us-east4",
        "endpoint_id": "mg-endpoint-aadfd050-433b-44cb-ab84-d72ee01e0a6a",
        "strengths": [
            "Medical knowledge",
            "Fast inference (4B params)",
            "Cloud-hosted (no local GPU needed)"
        ],
        "use_cases": [
            "Clinical diagnosis",
            "Multi-model comparison",
            "Lightweight medical reasoning"
        ]
    },

    "medgemma-27b-it-vertex": {
        "adapter": VertexMedGemmaAdapter,
        "status": "active",
        "description": "MedGemma-27B-IT via Vertex AI endpoint (multimodal: image+text)",
        "provider": "vertex_ai",
        "requires": "GOOGLE_APPLICATION_CREDENTIALS",
        "model_id": "google/medgemma-27b-it",
        "project_id": settings.google_cloud_project,
        "region": "us-central1",
        "endpoint_id": "mg-endpoint-ec89fb32-fd4d-4bfb-82e1-7ef11b6c4035",
        "strengths": [
            "Medical knowledge (27B params)",
            "Advanced clinical reasoning",
            "Multimodal (image+text)",
            "Cloud-hosted (no local GPU needed)",
            "Best accuracy among MedGemma variants"
        ],
        "use_cases": [
            "Complex clinical diagnosis",
            "Primary evaluation model",
            "Production medical reasoning",
            "Dermatology image analysis"
        ]
    },

    "medgemma-4b-it-vertex": {
        "adapter": VertexMedGemmaAdapter,
        "status": "active",
        "description": "MedGemma-4B-IT via Vertex AI endpoint (multimodal: image+text)",
        "provider": "vertex_ai",
        "requires": "GOOGLE_APPLICATION_CREDENTIALS",
        "model_id": "google/medgemma-4b-it",
        "project_id": settings.google_cloud_project,
        "region": "us-east4",
        "endpoint_id": "mg-endpoint-669bef4e-1e44-4544-af3d-b142c8b842e9",
        "strengths": [
            "Medical knowledge (4B params)",
            "Fast inference",
            "Multimodal (image+text)",
            "Cloud-hosted (no local GPU needed)",
            "Resource-efficient"
        ],
        "use_cases": [
            "Clinical diagnosis",
            "Rapid evaluation",
            "Lightweight medical reasoning",
            "Dermatology image analysis"
        ]
    },

    # ====== STUB MODELS (API keys not yet provided) ======

    "gpt4": {
        "adapter": OpenAIAdapter,
        "status": "stub",
        "description": "OpenAI GPT-4 Turbo - Advanced general purpose model",
        "provider": "openai",
        "requires": "OPENAI_API_KEY",
        "model_id": "gpt-4-turbo",
        "message": "OpenAI integration ready - add OPENAI_API_KEY to enable",
        "strengths": [
            "Advanced reasoning",
            "Code generation",
            "Structured outputs"
        ],
        "use_cases": [
            "Complex diagnostics",
            "Medical research",
            "Clinical decision support"
        ]
    },

    "claude": {
        "adapter": ClaudeAdapter,
        "status": "stub",
        "description": "Anthropic Claude 3 Opus - Long context specialist",
        "provider": "anthropic",
        "requires": "CLAUDE_API_KEY",
        "model_id": "claude-3-opus-20240229",
        "message": "Claude integration ready - add CLAUDE_API_KEY to enable",
        "strengths": [
            "Long context (200K tokens)",
            "Careful reasoning",
            "Nuanced analysis"
        ],
        "use_cases": [
            "Comprehensive case review",
            "Literature synthesis",
            "Detailed clinical notes"
        ]
    }
}


def get_available_models() -> List[str]:
    """
    Get list of available (active) model names.

    Returns:
        List of active model names
    """
    return [
        name for name, info in MODEL_REGISTRY.items()
        if info["status"] == "active"
    ]


def get_all_models() -> List[str]:
    """
    Get list of all registered models (active + stub).

    Returns:
        List of all model names
    """
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model.

    Args:
        model_name: Name of the model

    Returns:
        Model information dict

    Raises:
        ValueError: If model not found
    """
    if model_name not in MODEL_REGISTRY:
        available = get_all_models()
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {', '.join(available)}"
        )

    return MODEL_REGISTRY[model_name]


def is_model_available(model_name: str) -> bool:
    """
    Check if a model is available (has API key).

    Args:
        model_name: Name of the model

    Returns:
        True if model is active, False if stub or unavailable
    """
    try:
        info = get_model_info(model_name)
        return info["status"] == "active"
    except ValueError:
        return False


def get_model_adapter(model_name: str):
    """
    Get the adapter class for a model.

    Args:
        model_name: Name of the model

    Returns:
        Adapter class

    Raises:
        ValueError: If model not found or not available
    """
    info = get_model_info(model_name)

    # Check if model is active
    if info["status"] != "active":
        raise ValueError(
            f"\n=== Model '{model_name}' Not Available ===\n"
            f"{info.get('message', '')}\n\n"
            f"To enable:\n"
            f"1. Add {info['requires']} to your .env file\n"
            f"2. Implement the adapter in src/agents/models/{info['provider']}_adapter.py\n\n"
            f"Currently available models: {', '.join(get_available_models())}\n"
        )

    return info["adapter"]


def print_model_registry():
    """Print formatted model registry information."""
    print("\n" + "="*70)
    print("MODEL REGISTRY")
    print("="*70)

    # Active models
    active = [name for name, info in MODEL_REGISTRY.items() if info["status"] == "active"]
    if active:
        print("\n✅ ACTIVE MODELS (Ready to use):\n")
        for name in active:
            info = MODEL_REGISTRY[name]
            print(f"  {name:12} - {info['description']}")
            print(f"               Provider: {info['provider']}")
            print(f"               Model ID: {info['model_id']}")
            print()

    # Stub models
    stubs = [name for name, info in MODEL_REGISTRY.items() if info["status"] == "stub"]
    if stubs:
        print("⏳ STUB MODELS (Require API keys):\n")
        for name in stubs:
            info = MODEL_REGISTRY[name]
            print(f"  {name:12} - {info['description']}")
            print(f"               Requires: {info['requires']}")
            print(f"               {info.get('message', '')}")
            print()

    print("="*70 + "\n")


# Auto-print registry on import (for debugging)
if __name__ == "__main__":
    print_model_registry()
