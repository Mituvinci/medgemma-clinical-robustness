"""
MedGemma-27B Adapter

Adapter for Google's MedGemma-27B model via Hugging Face Inference API.
Medical specialist model trained on clinical data.
"""

from typing import Dict, Any, Optional
import logging
import json
from huggingface_hub import InferenceClient
import re

from src.agents.models.base_model import BaseLLM

logger = logging.getLogger(__name__)


class MedGemmaAdapter(BaseLLM):
    """Hugging Face adapter for MedGemma-27B."""

    def __init__(
        self,
        model_id: str = "google/medgemma-27b",
        api_key: str = None,
        **kwargs
    ):
        """
        Initialize MedGemma adapter.

        Args:
            model_id: Hugging Face model ID
            api_key: Hugging Face API token
            **kwargs: Additional parameters
        """
        super().__init__(model_id, api_key, **kwargs)

        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY is required for MedGemma")

        # Initialize Hugging Face Inference Client
        self.client = InferenceClient(
            model=model_id,
            token=api_key
        )

        logger.info(f"MedGemma adapter initialized with model: {model_id}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Generate text from MedGemma.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        try:
            # Call Hugging Face Inference API
            response = self.client.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False,
                **kwargs
            )

            logger.debug(f"MedGemma generated {len(response)} characters")
            return response

        except Exception as e:
            logger.error(f"Error calling MedGemma API: {e}")
            raise

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output.

        Args:
            prompt: Input prompt
            schema: Expected JSON schema
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Parsed JSON object
        """
        # Add JSON formatting instructions to prompt
        json_prompt = f"""{prompt}

IMPORTANT: Respond with ONLY valid JSON. No extra text before or after.

Expected JSON format:
{json.dumps(schema, indent=2)}

Response (JSON only):"""

        try:
            response = self.generate(
                json_prompt,
                temperature=temperature,
                max_tokens=2048,
                **kwargs
            )

            # Extract JSON from response (handle cases with markdown code blocks)
            json_str = self._extract_json(response)

            # Parse JSON
            parsed = json.loads(json_str)

            logger.debug("Successfully parsed structured output")
            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from MedGemma: {e}")
            logger.debug(f"Raw response: {response}")
            raise ValueError(f"Invalid JSON response from MedGemma: {e}")

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text (handles markdown code blocks).

        Args:
            text: Text potentially containing JSON

        Returns:
            Clean JSON string
        """
        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # Try to find JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)

        # Try to find JSON array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return match.group(0)

        # Return as-is if no patterns found
        return text.strip()

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count (rough approximation).

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 characters per token
        # MedGemma uses similar tokenization to other Gemma models
        return len(text) // 4

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "huggingface"

    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Generate with system and user prompts.

        MedGemma doesn't have native system message support,
        so we format it as a special instruction.

        Args:
            system_prompt: System instructions
            user_prompt: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        # Format for MedGemma
        formatted_prompt = f"""<|system|>
{system_prompt}
<|end|>

<|user|>
{user_prompt}
<|end|>

<|assistant|>"""

        return self.generate(formatted_prompt, temperature, max_tokens, **kwargs)
