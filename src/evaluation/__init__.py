"""
Evaluation module for MedGemma Clinical Robustness Assistant.

Provides tools to evaluate diagnostic robustness across context states.
"""

from .evaluator import (
    ClinicalCaseEvaluator,
    EvaluationResult,
    RobustnessMetrics,
    EvaluationReport
)

__all__ = [
    "ClinicalCaseEvaluator",
    "EvaluationResult",
    "RobustnessMetrics",
    "EvaluationReport"
]
