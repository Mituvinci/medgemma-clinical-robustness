"""
Model Registry — Central registry for all LLM models used in this project.

Maps short model names (used in CLI args and config) to their adapter classes,
provider configuration, and capability metadata.

Registry Schema (each entry in MODEL_REGISTRY):
─────────────────────────────────────────────────
  adapter      : (required) Adapter class implementing the BaseLLMAdapter interface.
                 Located in src/agents/models/. Wraps model-specific API calls.
  status       : (required) "active" — fully configured and ready to use.
                             "stub"   — adapter exists but API key not provided.
  description  : Human-readable model description (shown in UI and logs).
  provider     : API provider: "huggingface", "vertex_ai", "google", "openai", "anthropic".
  requires     : Environment variable that must be set to use this model.
  model_id     : Provider-specific model identifier passed to the adapter.

  HuggingFace-only fields:
    (none — model_id is the HF repo path, e.g. "google/medgemma-27b-it")

  Vertex AI-only fields:
    project_id  : Google Cloud project ID (from settings.google_cloud_project)
    region      : Vertex AI region (e.g., "us-central1")
    endpoint_id : Deployed Vertex AI endpoint ID for the model

  strengths   : List of model capability highlights (shown in UI).
  use_cases   : List of recommended use cases (shown in UI).
  message     : (stub only) Instruction for how to enable the model.

Architecture note:
    In this system, MedGemma models (27B, 4B, 1.5-4B-IT) perform ALL clinical
    reasoning. The Gemini/Google models act only as workflow orchestrators in
    Google ADK — they do NOT perform medical diagnosis. See adk_agents.py for
    the full multi-agent design.
"""

from typing import Dict, Any, List
import logging
from config.config import settings

from src.agents.models.medgemma_adapter import MedGemmaAdapter
from src.agents.models.gemini_adapter import GeminiAdapter
from src.agents.models.vertex_medgemma_adapter import VertexMedGemmaAdapter
from src.agents.models.hf_inference_adapter import HFInferenceAdapter
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

    "medgemma-1.5-4b": {
        "adapter": MedGemmaAdapter,
        "status": "active",
        "description": "MedGemma-1.5-4B-IT - Local GPU (download once, load from cache)",
        "provider": "huggingface",
        "requires": "HUGGINGFACE_API_KEY",
        "model_id": "google/medgemma-1.5-4b-it",
        "strengths": [
            "Medical knowledge (1.5-4B params)",
            "Low VRAM (~4-6 GB in 4-bit)",
            "Fast inference vs 27B"
        ],
        "use_cases": [
            "Gradio demo testing (GPU node)",
            "Clinical diagnosis",
            "Lightweight local reasoning"
        ]
    },

    "medgemma-hf": {
        "adapter": HFInferenceAdapter,
        "status": "active",
        "description": "MedGemma-1.5-4B-IT via HuggingFace Inference API (no local GPU)",
        "provider": "huggingface_api",
        "requires": "HUGGINGFACE_API_KEY",
        "model_id": "google/medgemma-1.5-4b-it",
        "strengths": [
            "Medical knowledge (1.5-4B params)",
            "No GPU required — serverless HF API",
            "Fast startup, no model loading"
        ],
        "use_cases": [
            "Gradio demo (lightweight)",
            "Clinical diagnosis via API",
            "Development and testing"
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
        "region": "us-central1",
        "endpoint_id": "mg-endpoint-3a7adf00-fcbd-4aa5-b768-6d5991e4dab1",
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
        "endpoint_id": "mg-endpoint-b05f7ec9-3ba1-417c-8c59-d272fd5fa70b",
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
        "region": "us-central1",
        "endpoint_id": "mg-endpoint-72307e8e-1e40-4b39-bdab-bf68ba67c0a2",
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
