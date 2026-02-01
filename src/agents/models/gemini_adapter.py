"""
Gemini Pro Adapter

Adapter for Google's Gemini Pro model via Google AI API.
General purpose state-of-the-art LLM.
"""

from typing import Dict, Any, Optional
import logging
import json
import google.generativeai as genai

from src.agents.models.base_model import BaseLLM

logger = logging.getLogger(__name__)


class GeminiAdapter(BaseLLM):
    """Google Gemini Pro adapter."""

    def __init__(
        self,
        model_id: str = "gemini-pro",
        api_key: str = None,
        **kwargs
    ):
        """
        Initialize Gemini adapter.

        Args:
            model_id: Gemini model ID (gemini-pro, gemini-pro-vision, etc.)
            api_key: Google API key
            **kwargs: Additional parameters
        """
        super().__init__(model_id, api_key, **kwargs)

        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini")

        # Configure Gemini
        genai.configure(api_key=api_key)

        # Initialize model
        self.model = genai.GenerativeModel(model_id)

        logger.info(f"Gemini adapter initialized with model: {model_id}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Generate text from Gemini.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        try:
            # Configure generation
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )

            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            # Extract text
            text = response.text

            logger.debug(f"Gemini generated {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
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
        # Add JSON formatting instructions
        json_prompt = f"""{prompt}

IMPORTANT: Respond with ONLY valid JSON. No markdown, no extra text.

Expected JSON format:
{json.dumps(schema, indent=2)}

Response (JSON only):"""

        try:
            response_text = self.generate(
                json_prompt,
                temperature=temperature,
                max_tokens=2048,
                **kwargs
            )

            # Clean response (remove markdown if present)
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]  # Remove ```json
            if json_str.startswith("```"):
                json_str = json_str[3:]  # Remove ```
            if json_str.endswith("```"):
                json_str = json_str[:-3]  # Remove ```
            json_str = json_str.strip()

            # Parse JSON
            parsed = json.loads(json_str)

            logger.debug("Successfully parsed structured output from Gemini")
            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemini: {e}")
            logger.debug(f"Raw response: {response_text}")
            raise ValueError(f"Invalid JSON response from Gemini: {e}")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using Gemini's tokenizer.

        Args:
            text: Input text

        Returns:
            Token count
        """
        try:
            # Use Gemini's built-in token counter
            token_count = self.model.count_tokens(text)
            return token_count.total_tokens
        except Exception as e:
            logger.warning(f"Error counting tokens, using approximation: {e}")
            # Fallback approximation
            return len(text) // 4

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "google"

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

        Gemini doesn't have explicit system messages in the basic API,
        but we can use system instructions in the model initialization.

        Args:
            system_prompt: System instructions
            user_prompt: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        # For system prompts, create a new model instance with system instruction
        try:
            model_with_system = genai.GenerativeModel(
                self.model_id,
                system_instruction=system_prompt
            )

            # Configure generation
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )

            # Generate response
            response = model_with_system.generate_content(
                user_prompt,
                generation_config=generation_config
            )

            return response.text

        except Exception as e:
            logger.error(f"Error with system prompt: {e}")
            # Fallback: combine prompts
            return super().generate_with_system(
                system_prompt,
                user_prompt,
                temperature,
                max_tokens,
                **kwargs
            )
