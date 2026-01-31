"""
OpenAI GPT-4 Adapter (STUB)

Status: STUB - Requires OPENAI_API_KEY to activate.

To enable:
1. Get OpenAI API key from https://platform.openai.com/api-keys
2. Add to .env: OPENAI_API_KEY=sk-...
3. Install: pip install openai
4. Uncomment implementation below
"""

from typing import Dict, Any
import logging

from src.agents.models.base_model import BaseLLM

logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseLLM):
    """
    OpenAI GPT-4 adapter.

    STUB: This is a placeholder for future OpenAI integration.
    """

    def __init__(
        self,
        model_id: str = "gpt-4-turbo",
        api_key: str = None,
        **kwargs
    ):
        """Initialize OpenAI adapter (STUB)."""
        super().__init__(model_id, api_key, **kwargs)

        if not api_key:
            raise ValueError(
                "\n=== OpenAI Integration Not Yet Activated ===\n"
                "To enable GPT-4:\n"
                "1. Get API key from: https://platform.openai.com/api-keys\n"
                "2. Add to .env: OPENAI_API_KEY=sk-...\n"
                "3. Install: pip install openai\n"
                "4. Implement methods in: src/agents/models/openai_adapter.py\n"
            )

        # TODO: Uncomment when ready to implement
        # from openai import OpenAI
        # self.client = OpenAI(api_key=api_key)

        logger.warning("OpenAI adapter is a STUB - needs implementation")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Generate text (STUB)."""
        raise NotImplementedError(
            "OpenAI adapter not yet implemented. "
            "Add OPENAI_API_KEY and implement this method."
        )

        # TODO: Uncomment when implementing
        # response = self.client.chat.completions.create(
        #     model=self.model_id,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=temperature,
        #     max_tokens=max_tokens,
        #     **kwargs
        # )
        # return response.choices[0].message.content

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.3,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structured output (STUB)."""
        raise NotImplementedError(
            "OpenAI structured generation not yet implemented."
        )

        # TODO: Uncomment when implementing
        # Use response_format={"type": "json_object"} for structured output

    def count_tokens(self, text: str) -> int:
        """Count tokens (STUB)."""
        # Rough approximation until implemented
        return len(text) // 4

        # TODO: Use tiktoken when implementing
        # import tiktoken
        # encoding = tiktoken.encoding_for_model(self.model_id)
        # return len(encoding.encode(text))

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "openai"


# IMPLEMENTATION TEMPLATE (uncomment when ready):
"""
from openai import OpenAI
import json

class OpenAIAdapter(BaseLLM):
    def __init__(self, model_id: str = "gpt-4-turbo", api_key: str = None, **kwargs):
        super().__init__(model_id, api_key, **kwargs)
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt, temperature=0.7, max_tokens=1024, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content

    def generate_structured(self, prompt, schema, temperature=0.3, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
            **kwargs
        )
        return json.loads(response.choices[0].message.content)
"""
