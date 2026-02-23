"""
Vertex AI MedGemma Adapter

Adapter for MedGemma models deployed on Google Cloud Vertex AI.
Supports any MedGemma variant deployed as a Vertex AI endpoint
(e.g., MedGemma-1.5-4B-IT, MedGemma-4B-IT, etc.).
"""

import base64
import json
import logging
import re
import time
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

        # Human-readable label derived from model_id (e.g. "google/medgemma-27b-it" → "MedGemma-27B-IT")
        self._model_label = model_id.split("/")[-1].replace("medgemma", "MedGemma")

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
        # Retry loop — Vertex AI online prediction endpoints have per-minute quotas.
        # 8 micro-calls per analysis can exhaust the quota during rapid testing.
        # Retry up to 3 times with 15-second backoff on 429 / RESOURCE_EXHAUSTED.
        _MAX_RETRIES = 3
        for _attempt in range(_MAX_RETRIES):
          try:
            # chatCompletions format — required by new MedGemma Vertex AI deployments.
            # Old {"prompt": ..., "image": ...} format is rejected with 500.
            image_path = kwargs.get("image_path")
            content = [{"type": "text", "text": prompt}]
            if image_path:
                import os as _os
                ext = _os.path.splitext(image_path)[1].lower().lstrip(".")
                mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
                b64 = self._encode_image(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"}
                })

            instance = {
                "@requestFormat": "chatCompletions",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "temperature": max(temperature, 1e-6),
            }

            response = self.endpoint.predict(instances=[instance], parameters={})

            # Extract text from chatCompletions response.
            # response.predictions may be a list OR a dict depending on the deployment.
            print(f"\n[{self._model_label}] Raw response received", flush=True)
            preds = response.predictions
            if isinstance(preds, list) and len(preds) > 0:
                result = preds[0]
            elif isinstance(preds, dict):
                result = preds
            else:
                result = None

            if result is None:
                text = ""
            elif isinstance(result, str):
                text = result
            elif isinstance(result, dict):
                choices = result.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                else:
                    text = result.get("generated_text", result.get("text", str(result)))
            else:
                text = str(result)

            print(f"[{self._model_label}] >>> {text[:200]}{'...' if len(text) > 200 else ''}", flush=True)

            # One-click-deploy endpoints echo the full prompt before the response.
            # Strip everything up to and including the last "Output:" marker.
            for marker in ["\nOutput:\n", "\nOutput:\r\n", "Output:\n"]:
                idx = text.rfind(marker)
                if idx != -1:
                    text = text[idx + len(marker):].strip()
                    break
            # Also strip "Final Output:" prefix if present
            if text.startswith("Final Output:"):
                text = text[len("Final Output:"):].strip()
            if text.startswith("---"):
                text = text[3:].strip()

            # MedGemma-1.5-4B-IT may wrap reasoning in <thought>...</thought> tags.
            # Two observed behaviors:
            #   A) Reasoning in <thought>, answer after </thought>  → keep after-thought text
            #   B) Full answer inside <thought>, short stub after   → keep thought content
            # Behavior B happens when the token budget runs out mid-answer: the model
            # generates the SOAP note inside the thought block, then starts "Okay, here
            # is a SOAP note..." after </thought> but gets cut off after a few words.
            # Naively stripping <thought>...</thought> in case B throws away the real answer.
            # Fix: if after-thought text is < 200 chars, fall back to the thought content.
            import re as _re
            thought_match = _re.search(r'<thought>(.*?)</thought>\s*(.*)', text, flags=_re.DOTALL)
            if thought_match:
                thought_content = thought_match.group(1).strip()
                after_thought   = thought_match.group(2).strip()
                if len(after_thought) >= 200:
                    # Normal case A: substantial answer follows </thought>
                    text = after_thought
                    logger.debug(f"<thought> stripped — using after-thought text ({len(text)} chars)")
                else:
                    # Case B: answer was inside <thought>, stub after is too short to use
                    text = thought_content
                    logger.debug(f"<thought> content kept — after-thought too short ({len(after_thought)} chars)")
            elif text.lower().startswith("thought"):
                # Partial tag — strip the leading "thought" word if no actual answer follows
                after = _re.sub(r'^thought\S*\s*', '', text, flags=_re.IGNORECASE).strip()
                if after:
                    text = after

            logger.debug(f"Vertex AI MedGemma generated {len(text)} characters")
            return text

          except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            if is_rate_limit and _attempt < _MAX_RETRIES - 1:
                wait_sec = 15 * (_attempt + 1)  # 15s, 30s
                logger.warning(
                    f"Vertex AI 429 rate limit on attempt {_attempt + 1}/{_MAX_RETRIES}. "
                    f"Waiting {wait_sec}s before retry..."
                )
                time.sleep(wait_sec)
                continue
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

            # Sanitize invalid JSON escape sequences (e.g., \a, \p from medical notation)
            json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)

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
