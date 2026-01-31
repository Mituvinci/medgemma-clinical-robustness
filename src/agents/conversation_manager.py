"""
Conversation Manager

Tracks complete workflow sessions with full audit trail for:
- Explainability (competition judging criteria)
- Medical compliance (PII-safe logging)
- Debugging and analysis
- Model comparison

This is critical for the competition's explainability scoring (25% of total).
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from src.utils.logger import setup_logger, pii_filter
from config.config import settings

logger = setup_logger(__name__)


class ConversationSession:
    """
    Represents a single workflow session.

    Tracks all interactions, agent decisions, and retrieved evidence
    for complete transparency and explainability.
    """

    def __init__(
        self,
        case_id: str,
        model_name: str,
        session_id: Optional[str] = None
    ):
        """
        Initialize conversation session.

        Args:
            case_id: Unique case identifier
            model_name: LLM model being used
            session_id: Optional session ID (auto-generated if None)
        """
        self.session_id = session_id or self._generate_session_id()
        self.case_id = case_id
        self.model_name = model_name

        self.timestamp_start = datetime.utcnow().isoformat() + "Z"
        self.timestamp_end = None

        # Workflow tracking
        self.workflow_steps: List[Dict[str, Any]] = []
        self.current_step = 0

        # Input/output
        self.initial_input: Optional[Dict] = None
        self.final_output: Optional[Dict] = None

        # Metrics
        self.total_tokens = 0
        self.total_latency_ms = 0

        # Flags
        self.pii_redacted = False
        self.completed = False

        logger.info(
            f"Started conversation session: {self.session_id} "
            f"(case={case_id}, model={model_name})"
        )

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique = str(uuid.uuid4())[:8]
        return f"session_{timestamp}_{unique}"

    def set_initial_input(self, input_data: Dict[str, Any]):
        """
        Set initial case input.

        Args:
            input_data: Initial case data
        """
        # Apply PII filtering
        filtered = pii_filter(json.dumps(input_data))
        self.initial_input = json.loads(filtered)
        self.pii_redacted = True

    def add_step(
        self,
        agent_name: str,
        step_data: Dict[str, Any]
    ):
        """
        Add a workflow step.

        Args:
            agent_name: Name of agent (triage, research, diagnostic)
            step_data: Step details including input, output, reasoning
        """
        self.current_step += 1

        step = {
            "step": self.current_step,
            "agent": agent_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **step_data
        }

        # Apply PII filtering to entire step
        filtered = pii_filter(json.dumps(step))
        step = json.loads(filtered)

        self.workflow_steps.append(step)

        # Update metrics
        if "metrics" in step_data:
            metrics = step_data["metrics"]
            self.total_tokens += metrics.get("tokens_used", 0)
            self.total_latency_ms += metrics.get("latency_ms", 0)

        logger.debug(f"Added step {self.current_step}: {agent_name}")

    def set_final_output(self, output_data: Dict[str, Any]):
        """
        Set final workflow output.

        Args:
            output_data: Final diagnostic result
        """
        # Apply PII filtering
        filtered = pii_filter(json.dumps(output_data))
        self.final_output = json.loads(filtered)

    def complete(self):
        """Mark session as completed."""
        self.timestamp_end = datetime.utcnow().isoformat() + "Z"
        self.completed = True
        logger.info(f"Completed session: {self.session_id}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary.

        Returns:
            Complete session data as dict
        """
        return {
            "session_id": self.session_id,
            "case_id": self.case_id,
            "model": self.model_name,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "completed": self.completed,

            "initial_input": self.initial_input,
            "workflow_steps": self.workflow_steps,
            "final_output": self.final_output,

            "metadata": {
                "total_steps": self.current_step,
                "total_tokens": self.total_tokens,
                "total_latency_ms": self.total_latency_ms,
                "pii_redacted": self.pii_redacted
            }
        }

    def to_json(self) -> str:
        """
        Convert session to JSON string.

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=2)

    def save(self, directory: Optional[Path] = None):
        """
        Save session to JSON file.

        Args:
            directory: Directory to save to (default: logs/sessions/)
        """
        if directory is None:
            directory = Path(settings.log_dir) / "sessions"

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Filename: session_id.json
        filepath = directory / f"{self.session_id}.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

        logger.info(f"Saved session to: {filepath}")
        return filepath

    @classmethod
    def load(cls, session_id: str, directory: Optional[Path] = None) -> 'ConversationSession':
        """
        Load session from JSON file.

        Args:
            session_id: Session ID to load
            directory: Directory to load from

        Returns:
            ConversationSession instance
        """
        if directory is None:
            directory = Path(settings.log_dir) / "sessions"

        filepath = Path(directory) / f"{session_id}.json"

        if not filepath.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reconstruct session
        session = cls(
            case_id=data["case_id"],
            model_name=data["model"],
            session_id=data["session_id"]
        )

        session.timestamp_start = data["timestamp_start"]
        session.timestamp_end = data["timestamp_end"]
        session.initial_input = data["initial_input"]
        session.workflow_steps = data["workflow_steps"]
        session.final_output = data["final_output"]
        session.current_step = data["metadata"]["total_steps"]
        session.total_tokens = data["metadata"]["total_tokens"]
        session.total_latency_ms = data["metadata"]["total_latency_ms"]
        session.pii_redacted = data["metadata"]["pii_redacted"]
        session.completed = data["completed"]

        return session

    def get_reasoning_chain(self) -> List[str]:
        """
        Extract reasoning chain from all steps.

        Returns:
            List of reasoning statements
        """
        chain = []
        for step in self.workflow_steps:
            if "reasoning" in step:
                chain.append(f"[{step['agent']}] {step['reasoning']}")
        return chain

    def get_cited_guidelines(self) -> List[Dict[str, Any]]:
        """
        Extract all cited guidelines.

        Returns:
            List of guideline citations with metadata
        """
        citations = []

        for step in self.workflow_steps:
            # Check for retrieved documents (research agent)
            if "retrieved_documents" in step:
                for doc in step["retrieved_documents"]:
                    citations.append({
                        "source": doc.get("source"),
                        "title": doc.get("title"),
                        "similarity": doc.get("similarity"),
                        "text_preview": doc.get("text", "")[:100] + "..."
                    })

            # Check for cited guidelines in output
            if "output" in step and "cited_guidelines" in step["output"]:
                for guideline in step["output"]["cited_guidelines"]:
                    if isinstance(guideline, str):
                        citations.append({"guideline": guideline})
                    else:
                        citations.append(guideline)

        return citations

    def __repr__(self):
        return (
            f"ConversationSession(id={self.session_id}, "
            f"case={self.case_id}, model={self.model_name}, "
            f"steps={self.current_step}, completed={self.completed})"
        )


class ConversationManager:
    """
    Manages multiple conversation sessions.

    Provides session lifecycle management and analytics.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize conversation manager.

        Args:
            storage_dir: Directory for session storage
        """
        if storage_dir is None:
            storage_dir = Path(settings.log_dir) / "sessions"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.active_sessions: Dict[str, ConversationSession] = {}

        logger.info(f"ConversationManager initialized (storage: {self.storage_dir})")

    def create_session(
        self,
        case_id: str,
        model_name: str
    ) -> ConversationSession:
        """
        Create new conversation session.

        Args:
            case_id: Case identifier
            model_name: Model name

        Returns:
            New ConversationSession
        """
        session = ConversationSession(
            case_id=case_id,
            model_name=model_name
        )

        self.active_sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Get active session by ID.

        Args:
            session_id: Session ID

        Returns:
            ConversationSession or None
        """
        return self.active_sessions.get(session_id)

    def complete_session(self, session_id: str, save: bool = True):
        """
        Complete and optionally save session.

        Args:
            session_id: Session ID
            save: Whether to save to disk
        """
        session = self.active_sessions.get(session_id)

        if not session:
            logger.warning(f"Session not found: {session_id}")
            return

        session.complete()

        if save:
            session.save(self.storage_dir)

        # Remove from active sessions
        del self.active_sessions[session_id]

    def list_sessions(
        self,
        case_id: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> List[str]:
        """
        List saved session IDs.

        Args:
            case_id: Filter by case ID
            model_name: Filter by model name

        Returns:
            List of session IDs
        """
        session_files = list(self.storage_dir.glob("session_*.json"))
        session_ids = []

        for filepath in session_files:
            # Quick load to check filters
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Apply filters
            if case_id and data.get("case_id") != case_id:
                continue
            if model_name and data.get("model") != model_name:
                continue

            session_ids.append(data["session_id"])

        return sorted(session_ids)

    def load_session(self, session_id: str) -> ConversationSession:
        """
        Load session from disk.

        Args:
            session_id: Session ID

        Returns:
            ConversationSession
        """
        return ConversationSession.load(session_id, self.storage_dir)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get manager statistics.

        Returns:
            Statistics dict
        """
        all_sessions = self.list_sessions()

        return {
            "total_sessions": len(all_sessions),
            "active_sessions": len(self.active_sessions),
            "storage_directory": str(self.storage_dir)
        }


# Global conversation manager instance
_conversation_manager = None


def get_conversation_manager() -> ConversationManager:
    """
    Get global conversation manager instance.

    Returns:
        ConversationManager singleton
    """
    global _conversation_manager

    if _conversation_manager is None:
        _conversation_manager = ConversationManager()

    return _conversation_manager
