"""
Agent Factory

Dynamic factory for creating agents with different LLM models.
Allows runtime model switching without code changes.
"""

from typing import Optional, Dict, Any
import logging
from config.config import settings

from src.agents.registry import get_model_adapter, get_model_info, get_available_models
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating agents with specified LLM model."""

    def __init__(self, model_name: str = None):
        """
        Initialize factory.

        Args:
            model_name: Name of LLM model to use (default from config)
        """
        self.model_name = model_name or settings.active_model

        # Validate model
        if self.model_name not in get_available_models():
            available = get_available_models()
            raise ValueError(
                f"Model '{self.model_name}' is not available. "
                f"Available models: {', '.join(available)}"
            )

        # Get model info
        self.model_info = get_model_info(self.model_name)

        # Get adapter class
        adapter_class = get_model_adapter(self.model_name)

        # Get API key based on provider
        api_key = self._get_api_key()

        # Initialize LLM adapter
        self.llm = adapter_class(
            model_id=self.model_info["model_id"],
            api_key=api_key
        )

        logger.info(f"Agent factory initialized with model: {self.model_name}")

    def _get_api_key(self) -> str:
        """
        Get API key for the current model.

        Returns:
            API key string

        Raises:
            ValueError: If API key not found
        """
        provider = self.model_info["provider"]

        if provider == "huggingface":
            key = settings.huggingface_api_key
        elif provider == "google":
            key = settings.google_api_key
        elif provider == "openai":
            key = getattr(settings, "openai_api_key", "")
        elif provider == "anthropic":
            key = getattr(settings, "claude_api_key", "")
        else:
            raise ValueError(f"Unknown provider: {provider}")

        if not key:
            raise ValueError(
                f"API key not found for {provider}. "
                f"Required: {self.model_info['requires']}"
            )

        return key

    def create_triage_agent(self, **kwargs):
        """
        Create triage agent.

        Args:
            **kwargs: Additional arguments for agent

        Returns:
            Triage agent instance
        """
        from src.agents.triage_agent import TriageAgent

        return TriageAgent(llm=self.llm, **kwargs)

    def create_research_agent(self, retriever: Optional[Retriever] = None, **kwargs):
        """
        Create research agent.

        Args:
            retriever: Retriever instance (creates new if None)
            **kwargs: Additional arguments for agent

        Returns:
            Research agent instance
        """
        from src.agents.research_agent import ResearchAgent

        if retriever is None:
            retriever = Retriever()

        return ResearchAgent(llm=self.llm, retriever=retriever, **kwargs)

    def create_diagnostic_agent(self, **kwargs):
        """
        Create diagnostic agent.

        Args:
            **kwargs: Additional arguments for agent

        Returns:
            Diagnostic agent instance
        """
        from src.agents.diagnostic_agent import DiagnosticAgent

        return DiagnosticAgent(llm=self.llm, **kwargs)

    def create_workflow(self, retriever: Optional[Retriever] = None, **kwargs):
        """
        Create complete agent workflow (coordinator).

        Args:
            retriever: Retriever instance
            **kwargs: Additional arguments

        Returns:
            Coordinator instance
        """
        from src.agents.coordinator import Coordinator

        # Create all agents
        triage = self.create_triage_agent()
        research = self.create_research_agent(retriever)
        diagnostic = self.create_diagnostic_agent()

        # Create coordinator
        return Coordinator(
            triage_agent=triage,
            research_agent=research,
            diagnostic_agent=diagnostic,
            **kwargs
        )

    def get_info(self) -> Dict[str, Any]:
        """
        Get factory information.

        Returns:
            Dict with model and adapter info
        """
        return {
            "model_name": self.model_name,
            "model_info": self.model_info,
            "llm_adapter": self.llm.get_model_info()
        }


# Convenience function
def create_workflow(model: str = None, retriever: Optional[Retriever] = None, **kwargs):
    """
    Create agent workflow with specified model.

    Args:
        model: Model name (default from config)
        retriever: Retriever instance
        **kwargs: Additional arguments

    Returns:
        Coordinator instance

    Example:
        >>> # Use MedGemma
        >>> workflow = create_workflow(model="medgemma")
        >>> result = workflow.run(case)
        >>>
        >>> # Use Gemini
        >>> workflow = create_workflow(model="gemini")
        >>> result = workflow.run(case)
    """
    factory = AgentFactory(model_name=model)
    return factory.create_workflow(retriever=retriever, **kwargs)


# Comparison utility
def compare_models(case, models: list = None, retriever: Optional[Retriever] = None):
    """
    Run the same case through multiple models for comparison.

    Args:
        case: Clinical case to analyze
        models: List of model names (default: all available)
        retriever: Shared retriever instance

    Returns:
        Dict mapping model names to results

    Example:
        >>> case = load_case("nejm_001")
        >>> results = compare_models(case, models=["medgemma", "gemini"])
        >>> for model, result in results.items():
        ...     print(f"{model}: {result.diagnosis}")
    """
    if models is None:
        models = get_available_models()

    if retriever is None:
        retriever = Retriever()

    results = {}

    for model_name in models:
        logger.info(f"Running comparison with model: {model_name}")

        try:
            workflow = create_workflow(model=model_name, retriever=retriever)
            result = workflow.run(case)
            results[model_name] = result
        except Exception as e:
            logger.error(f"Error with model {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    return results
