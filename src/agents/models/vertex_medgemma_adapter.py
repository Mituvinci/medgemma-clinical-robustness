"""
Vertex AI MedGemma Adapter

Adapter for MedGemma models deployed on Google Cloud Vertex AI.
Supports any MedGemma variant deployed as a Vertex AI endpoint
(e.g., MedGemma-1.5-4B-IT, MedGemma-4B-IT, etc.).
"""

import base64
import json
import logging
from typing import Dict, Any, Optional

from google.cloud import aiplatform

from src.agents.models.base_model import BaseLLM

logger = logging.getLogger(__name__)


class VertexMedGemmaAdapter(BaseLLM):
    """Adapter for MedGemma models deployed on Vertex AI endpoints."""

    def __init__(
        self,
        model_id: str = "medgemma-1.5-4b-it",
        api_key: str = None,
        project_id: str = None,
        region: str = "us-central1",
        endpoint_id: str = None,
        **kwargs
    ):
        """
        Initialize Vertex AI MedGemma adapter.

        Args:
            model_id: Model identifier for logging/tracking
            api_key: Not used (Vertex AI uses service account auth)
            project_id: Google Cloud project ID
            region: Vertex AI region
            endpoint_id: Deployed endpoint ID
            **kwargs: Additional parameters
        """
        super().__init__(model_id, api_key or "", **kwargs)

        if not project_id:
            raise ValueError("project_id is required for Vertex AI")
        if not endpoint_id:
            raise ValueError("endpoint_id is required for Vertex AI")

        self.project_id = project_id
        self.region = region
        self.endpoint_id = endpoint_id

        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)
        self.endpoint = aiplatform.Endpoint(endpoint_id)

        logger.info(
            f"Vertex AI MedGemma adapter initialized: "
            f"model={model_id}, project={project_id}, "
            f"region={region}, endpoint={endpoint_id}"
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Generate text from Vertex AI MedGemma endpoint.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (image_path for multimodal)

        Returns:
            Generated text
        """
        try:
            instance = {"prompt": prompt}

            # Handle image input if provided
            image_path = kwargs.get("image_path")
            if image_path:
                instance["image"] = self._encode_image(image_path)

            response = self.endpoint.predict(instances=[instance])

            # Extract text from predictions
            if response.predictions:
                result = response.predictions[0]
                if isinstance(result, str):
                    text = result
                elif isinstance(result, dict):
                    text = result.get("generated_text", result.get("text", str(result)))
                else:
                    text = str(result)
            else:
                text = ""

            logger.debug(f"Vertex AI MedGemma generated {len(text)} characters")
            return text

        except Exception as e:
            logger.error(f"Error calling Vertex AI endpoint: {e}")
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

            # Clean response
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()

            parsed = json.loads(json_str)
            logger.debug("Successfully parsed structured output from Vertex AI MedGemma")
            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Vertex AI MedGemma: {e}")
            logger.debug(f"Raw response: {response_text}")
            raise ValueError(f"Invalid JSON response from Vertex AI MedGemma: {e}")

    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Generate with system and user prompts combined.

        Args:
            system_prompt: System instructions
            user_prompt: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        combined = f"{system_prompt}\n\n{user_prompt}"
        return self.generate(combined, temperature, max_tokens, **kwargs)

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count (Vertex AI endpoints don't expose tokenizer).

        Args:
            text: Input text

        Returns:
            Approximate token count
        """
        return len(text) // 4

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "vertex_ai"

    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 for Vertex AI.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
