"""
Claude 3 Adapter (STUB)

Status: STUB - Requires CLAUDE_API_KEY to activate.

To enable:
1. Get Anthropic API key from https://console.anthropic.com/
2. Add to .env: CLAUDE_API_KEY=sk-ant-...
3. Install: pip install anthropic
4. Uncomment implementation below
"""

from typing import Dict, Any
import logging

from src.agents.models.base_model import BaseLLM

logger = logging.getLogger(__name__)


class ClaudeAdapter(BaseLLM):
    """
    Anthropic Claude 3 adapter.

    STUB: This is a placeholder for future Claude integration.
    """

    def __init__(
        self,
        model_id: str = "claude-3-opus-20240229",
        api_key: str = None,
        **kwargs
    ):
        """Initialize Claude adapter (STUB)."""
        super().__init__(model_id, api_key, **kwargs)

        if not api_key:
            raise ValueError(
                "\n=== Claude Integration Not Yet Activated ===\n"
                "To enable Claude:\n"
                "1. Get API key from: https://console.anthropic.com/\n"
                "2. Add to .env: CLAUDE_API_KEY=sk-ant-...\n"
                "3. Install: pip install anthropic\n"
                "4. Implement methods in: src/agents/models/claude_adapter.py\n"
            )

        # TODO: Uncomment when ready to implement
        # from anthropic import Anthropic
        # self.client = Anthropic(api_key=api_key)

        logger.warning("Claude adapter is a STUB - needs implementation")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Generate text (STUB)."""
        raise NotImplementedError(
            "Claude adapter not yet implemented. "
            "Add CLAUDE_API_KEY and implement this method."
        )

        # TODO: Uncomment when implementing
        # message = self.client.messages.create(
        #     model=self.model_id,
        #     max_tokens=max_tokens,
        #     temperature=temperature,
        #     messages=[{"role": "user", "content": prompt}],
        #     **kwargs
        # )
        # return message.content[0].text

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.3,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structured output (STUB)."""
        raise NotImplementedError(
            "Claude structured generation not yet implemented."
        )

        # TODO: Implement with JSON mode

    def count_tokens(self, text: str) -> int:
        """Count tokens (STUB)."""
        # Rough approximation until implemented
        return len(text) // 4

        # TODO: Use Anthropic's token counter when implementing
        # token_count = self.client.count_tokens(text)
        # return token_count

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "anthropic"

    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Generate with system prompt (STUB)."""
        raise NotImplementedError(
            "Claude adapter not yet implemented."
        )

        # TODO: Claude has native system message support
        # message = self.client.messages.create(
        #     model=self.model_id,
        #     max_tokens=max_tokens,
        #     temperature=temperature,
        #     system=system_prompt,  # Native system message!
        #     messages=[{"role": "user", "content": user_prompt}],
        #     **kwargs
        # )
        # return message.content[0].text


# IMPLEMENTATION TEMPLATE (uncomment when ready):
"""
from anthropic import Anthropic
import json

class ClaudeAdapter(BaseLLM):
    def __init__(self, model_id: str = "claude-3-opus-20240229", api_key: str = None, **kwargs):
        super().__init__(model_id, api_key, **kwargs)
        self.client = Anthropic(api_key=api_key)

    def generate(self, prompt, temperature=0.7, max_tokens=1024, **kwargs):
        message = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return message.content[0].text

    def generate_with_system(self, system_prompt, user_prompt, temperature=0.7, max_tokens=1024, **kwargs):
        message = self.client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            **kwargs
        )
        return message.content[0].text
"""
