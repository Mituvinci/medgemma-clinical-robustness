"""
Base LLM Interface

Abstract base class that all model adapters must implement.
This ensures consistent API across different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract base class for LLM adapters."""

    def __init__(self, model_id: str, api_key: str, **kwargs):
        """
        Initialize LLM adapter.

        Args:
            model_id: Model identifier (e.g., "google/medgemma-27b")
            api_key: API key for the provider
            **kwargs: Additional provider-specific parameters
        """
        self.model_id = model_id
        self.api_key = api_key
        self.kwargs = kwargs

        logger.info(f"Initialized {self.__class__.__name__} with model {model_id}")

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Generate text completion from prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured output (JSON) conforming to schema.

        Args:
            prompt: Input prompt
            schema: Pydantic model or JSON schema
            temperature: Sampling temperature (lower for structured output)
            **kwargs: Additional generation parameters

        Returns:
            Parsed structured output
        """
        pass

    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Generate with separate system and user prompts.

        Default implementation combines prompts. Override if model
        supports native system messages.

        Args:
            system_prompt: System/instruction prompt
            user_prompt: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        # Default: combine prompts
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        return self.generate(combined_prompt, temperature, max_tokens, **kwargs)

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        pass

    def validate_response(self, response: str) -> bool:
        """
        Validate LLM response.

        Args:
            response: Generated response

        Returns:
            True if valid, False otherwise
        """
        # Basic validation
        if not response or not response.strip():
            logger.warning("Empty response from LLM")
            return False

        return True

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dict with model metadata
        """
        return {
            "adapter": self.__class__.__name__,
            "model_id": self.model_id,
            "provider": self.get_provider_name(),
        }

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get provider name.

        Returns:
            Provider name (e.g., "huggingface", "google", "openai")
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(model_id='{self.model_id}')"
