"""
Logging utilities for MedGemma Clinical Robustness Assistant.
Provides structured, PII-safe logging for agent decisions and RAG operations.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from pythonjsonlogger import jsonlogger
from config.config import settings


class PIISafeFilter(logging.Filter):
    """Filter to redact potential PII from logs."""

    PII_KEYWORDS = ["patient_name", "name", "ssn", "dob", "date_of_birth", "phone", "email"]

    def filter(self, record):
        """Redact PII from log messages."""
        if hasattr(record, "msg"):
            msg = str(record.msg)
            for keyword in self.PII_KEYWORDS:
                if keyword in msg.lower():
                    record.msg = msg.replace(keyword, "[REDACTED]")
        return True


def setup_logger(name: str, log_level: str = None) -> logging.Logger:
    """
    Set up a structured JSON logger with PII protection.

    Args:
        name: Logger name (typically __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level or settings.log_level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create logs directory if it doesn't exist
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Console handler (human-readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    console_handler.addFilter(PIISafeFilter())

    # File handler (JSON structured)
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"medgemma_{timestamp}.json"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    json_format = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        timestamp=True
    )
    file_handler.setFormatter(json_format)
    file_handler.addFilter(PIISafeFilter())

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def log_agent_decision(logger: logging.Logger, agent_name: str, decision: dict):
    """
    Log an agent decision with structured data.

    Args:
        logger: Logger instance
        agent_name: Name of the agent making the decision
        decision: Dictionary containing decision details
    """
    logger.info(
        f"Agent decision from {agent_name}",
        extra={
            "agent": agent_name,
            "decision_type": decision.get("type", "unknown"),
            "reasoning": decision.get("reasoning", ""),
            "metadata": decision.get("metadata", {})
        }
    )


def log_retrieval(logger: logging.Logger, query: str, num_results: int, sources: list):
    """
    Log RAG retrieval operation.

    Args:
        logger: Logger instance
        query: Search query
        num_results: Number of documents retrieved
        sources: List of source identifiers
    """
    logger.info(
        f"RAG retrieval: {num_results} documents",
        extra={
            "query": query,
            "num_results": num_results,
            "sources": sources,
            "operation": "retrieval"
        }
    )


def log_model_call(logger: logging.Logger, model_name: str, prompt_length: int, response_length: int):
    """
    Log LLM API call (without exposing actual content).

    Args:
        logger: Logger instance
        model_name: Name of the model called
        prompt_length: Length of prompt in characters
        response_length: Length of response in characters
    """
    logger.debug(
        f"Model call to {model_name}",
        extra={
            "model": model_name,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "operation": "model_call"
        }
    )


def pii_filter(text: str) -> str:
    """
    Filter PII from text string.

    Args:
        text: Text to filter

    Returns:
        Filtered text with PII redacted
    """
    import re

    # Common PII patterns
    patterns = {
        r'\b\d{3}-\d{2}-\d{4}\b': '[REDACTED_SSN]',  # SSN
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[REDACTED_EMAIL]',  # Email
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b': '[REDACTED_PHONE]',  # Phone
        r'\b\d{4}-\d{2}-\d{2}\b': '[REDACTED_DATE]',  # Date YYYY-MM-DD
        r'\b\d{2}/\d{2}/\d{4}\b': '[REDACTED_DATE]',  # Date MM/DD/YYYY
    }

    filtered_text = text
    for pattern, replacement in patterns.items():
        filtered_text = re.sub(pattern, replacement, filtered_text)

    # Redact common PII keywords
    pii_keywords = PIISafeFilter.PII_KEYWORDS
    for keyword in pii_keywords:
        # Case-insensitive replacement
        filtered_text = re.sub(
            rf'({keyword})\s*[:\s]+([^\s,]+)',
            rf'\1: [REDACTED]',
            filtered_text,
            flags=re.IGNORECASE
        )

    return filtered_text
