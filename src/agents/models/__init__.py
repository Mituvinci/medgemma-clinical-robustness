"""
Model Adapters Package

Provides uniform interface for different LLM providers:
- MedGemma-27B (Hugging Face) - Medical specialist
- Gemini Pro (Google) - General purpose
- GPT-4 (OpenAI) - Future
- Claude (Anthropic) - Future
"""

from src.agents.models.base_model import BaseLLM
from src.agents.models.medgemma_adapter import MedGemmaAdapter
from src.agents.models.gemini_adapter import GeminiAdapter
from src.agents.models.vertex_medgemma_adapter import VertexMedGemmaAdapter

# Future imports (will be activated when API keys are available)
# from src.agents.models.openai_adapter import OpenAIAdapter
# from src.agents.models.claude_adapter import ClaudeAdapter

__all__ = [
    "BaseLLM",
    "MedGemmaAdapter",
    "GeminiAdapter",
    "VertexMedGemmaAdapter",
    # "OpenAIAdapter",
    # "ClaudeAdapter",
]
