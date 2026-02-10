"""
Evaluation Module for MedGemma Clinical Robustness Assistant

Evaluates diagnostic robustness across 4 context states:
1. Original: Complete clinical data (history + exam + image)
2. History-Only: Patient history without exam or image
3. Image-Only: Clinical photo without history or exam
4. Exam-Restricted: Minimal physical exam findings

Metrics:
- Diagnostic accuracy (top-1 match with gold standard)
- Differential inclusion (gold standard in top-3)
- Confidence calibration
- Consistency across context states
- Agentic pause appropriateness (detects missing data)
- Robustness delta (accuracy change across states)
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re

from src.agents.adk_agents import create_workflow
from src.utils.schemas import ClinicalCase, ContextState
from config.config import settings

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result for a single case evaluation."""
    case_id: str
    context_state: str
    gold_standard: str
    predicted_diagnosis: Optional[str]
    differential_diagnoses: List[str]
    confidence: float
    agentic_pause_triggered: bool
    missing_data_detected: List[str]
    top1_correct: bool
    top3_correct: bool
    response_text: str
    execution_time_ms: float
    error: Optional[str] = None


@dataclass
class RobustnessMetrics:
    """Robustness metrics across context states."""
    total_cases: int
    context_state: str

    # Accuracy metrics
    top1_accuracy: float
    top3_accuracy: float
    avg_confidence: float

    # Agentic behavior metrics
    pause_rate: float  # % of cases where agent paused
    appropriate_pause_rate: float  # % of pauses that were appropriate

    # Consistency metrics
    avg_differential_overlap: float  # Overlap with original context state

    # Error analysis
    errors: int
    avg_execution_time_ms: float


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    model_name: str
    total_cases: int
    context_states: List[str]

    # Overall metrics
    overall_top1_accuracy: float
    overall_top3_accuracy: float

    # Per-context metrics
    metrics_by_context: Dict[str, RobustnessMetrics]

    # Robustness analysis
    robustness_delta: Dict[str, float]  # Accuracy drop from original to each state
    consistency_score: float  # How consistent diagnoses are across states

    # Detailed results
    individual_results: List[EvaluationResult]

    # Error summary
    total_errors: int
    error_details: List[Dict[str, Any]]


class ClinicalCaseEvaluator:
    """Evaluates MedGemma workflow on clinical cases."""

    def __init__(
        self,
        model_name: str = "gemini-pro-latest",
        use_medgemma: bool = True
    ):
        """
        Initialize evaluator.

        Args:
            model_name: Gemini model for orchestration
            use_medgemma: Whether to use MedGemma specialist
        """
        self.workflow = create_workflow(
            model_name=model_name,
            use_medgemma=use_medgemma
        )
        self.model_name = model_name
        self.use_medgemma = use_medgemma

        logger.info(f"ClinicalCaseEvaluator initialized with model: {model_name}")

    def load_evaluation_cases(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load evaluation cases from JSON file.

        Args:
            filepath: Path to evaluation_cases.json

        Returns:
            List of case dictionaries
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        cases = data.get("cases", [])
        logger.info(f"Loaded {len(cases)} evaluation cases from {filepath}")
        return cases

    def generate_context_states(self, original_case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate 4 context states from an original case.

        Context States:
        1. original: Complete data (history + exam + image)
        2. history_only: Only patient history
        3. image_only: Only clinical image
        4. exam_restricted: Minimal physical exam

        Args:
            original_case: Original case with complete data

        Returns:
            List of 4 case variants
        """
        base_id = original_case["case_id"]

        # 1. Original (complete)
        original = original_case.copy()
        original["context_state"] = "original"

        # 2. History-Only
        history_only = {
            "case_id": f"{base_id}_HISTORY_ONLY",
            "context_state": "history_only",
            "patient_age": original_case.get("patient_age"),
            "patient_gender": original_case.get("patient_gender"),
            "duration": original_case.get("duration"),
            "chief_complaint": original_case.get("chief_complaint"),
            "history": original_case.get("history"),
            "physical_exam": None,
            "image_path": None,
            "gold_standard_diagnosis": original_case["gold_standard_diagnosis"],
            "differential_diagnoses": original_case.get("differential_diagnoses", [])
        }

        # 3. Image-Only
        image_only = {
            "case_id": f"{base_id}_IMAGE_ONLY",
            "context_state": "image_only",
            "patient_age": None,
            "patient_gender": None,
            "duration": None,
            "chief_complaint": None,
            "history": None,
            "physical_exam": None,
            "image_path": original_case.get("image_path"),
            "gold_standard_diagnosis": original_case["gold_standard_diagnosis"],
            "differential_diagnoses": original_case.get("differential_diagnoses", [])
        }

        # 4. Exam-Restricted
        exam_text = original_case.get("physical_exam", "")
        # Extract only first sentence (minimal findings)
        restricted_exam = exam_text.split('.')[0] + '.' if exam_text else "Physical examination findings available."

        exam_restricted = {
            "case_id": f"{base_id}_EXAM_RESTRICTED",
            "context_state": "exam_restricted",
            "patient_age": original_case.get("patient_age"),
            "patient_gender": original_case.get("patient_gender"),
            "duration": original_case.get("duration"),
            "chief_complaint": original_case.get("chief_complaint"),
            "history": original_case.get("history"),
            "physical_exam": restricted_exam,
            "image_path": None,
            "gold_standard_diagnosis": original_case["gold_standard_diagnosis"],
            "differential_diagnoses": original_case.get("differential_diagnoses", [])
        }

        return [original, history_only, image_only, exam_restricted]

    async def evaluate_case(self, case_data: Dict[str, Any]) -> EvaluationResult:
        """
        Evaluate a single case.

        Args:
            case_data: Case dictionary

        Returns:
            EvaluationResult object
        """
        case_id = case_data["case_id"]
        context_state = case_data["context_state"]
        gold_standard = case_data["gold_standard_diagnosis"]

        logger.info(f"Evaluating {case_id} ({context_state})")

        try:
            # Build ClinicalCase object
            clinical_case = ClinicalCase(
                case_id=case_id,
                patient_age=case_data.get("patient_age"),
                patient_gender=case_data.get("patient_gender"),
                duration=case_data.get("duration"),
                chief_complaint=case_data.get("chief_complaint"),
                history=case_data.get("history"),
                physical_exam=case_data.get("physical_exam"),
                image_data=None,  # TODO: Load actual image if image_path provided
                context_state=context_state
            )

            # Run workflow
            start_time = datetime.now()
            result = await self.workflow.run_async(clinical_case)
            end_time = datetime.now()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000

            response_text = result.get("response", "")

            # Parse response for diagnosis and differentials
            predicted_diagnosis, differentials, confidence = self._parse_diagnosis(response_text)

            # Detect if agentic pause was triggered
            agentic_pause = self._detect_agentic_pause(response_text)
            missing_data = self._extract_missing_data(response_text)

            # Evaluate accuracy
            top1_correct = self._normalize_diagnosis(predicted_diagnosis) == self._normalize_diagnosis(gold_standard)
            top3_correct = top1_correct or any(
                self._normalize_diagnosis(gold_standard) in self._normalize_diagnosis(diff)
                for diff in differentials[:3]
            )

            return EvaluationResult(
                case_id=case_id,
                context_state=context_state,
                gold_standard=gold_standard,
                predicted_diagnosis=predicted_diagnosis,
                differential_diagnoses=differentials,
                confidence=confidence,
                agentic_pause_triggered=agentic_pause,
                missing_data_detected=missing_data,
                top1_correct=top1_correct,
                top3_correct=top3_correct,
                response_text=response_text,
                execution_time_ms=execution_time_ms,
                error=None
            )

        except Exception as e:
            logger.error(f"Error evaluating {case_id}: {e}")
            return EvaluationResult(
                case_id=case_id,
                context_state=context_state,
                gold_standard=gold_standard,
                predicted_diagnosis=None,
                differential_diagnoses=[],
                confidence=0.0,
                agentic_pause_triggered=False,
                missing_data_detected=[],
                top1_correct=False,
                top3_correct=False,
                response_text="",
                execution_time_ms=0.0,
                error=str(e)
            )

    def _parse_diagnosis(self, response: str) -> Tuple[Optional[str], List[str], float]:
        """
        Parse predicted diagnosis and differentials from response.

        Returns:
            (predicted_diagnosis, differential_list, confidence_score)
        """
        # Look for Assessment section in SOAP note
        assessment_match = re.search(r'Assessment.*?:(.*?)(?:Plan|$)', response, re.DOTALL | re.IGNORECASE)

        if not assessment_match:
            return None, [], 0.0

        assessment_text = assessment_match.group(1)

        # Extract primary diagnosis (often first line or "Primary:" or "Most likely:")
        primary_match = re.search(r'(?:Primary diagnosis|Most likely|Assessment):?\s*([^\n]+)', assessment_text, re.IGNORECASE)
        predicted = primary_match.group(1).strip() if primary_match else None

        # Extract differential diagnoses
        differentials = []
        diff_section = re.search(r'Differential.*?:(.*?)(?:\n\n|$)', assessment_text, re.DOTALL | re.IGNORECASE)
        if diff_section:
            diff_text = diff_section.group(1)
            # Extract numbered or bulleted items
            diff_items = re.findall(r'(?:^|\n)\s*[-•\d.]+\s*([^\n]+)', diff_text)
            differentials = [item.strip() for item in diff_items if item.strip()]

        # Extract confidence (if present)
        confidence = 0.7  # Default
        conf_match = re.search(r'confidence:?\s*([\d.]+)', response, re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1))

        return predicted, differentials, confidence

    def _detect_agentic_pause(self, response: str) -> bool:
        """Detect if agent paused for missing data."""
        response_lower = response.lower()

        # Check for questions without complete SOAP
        has_questions = "?" in response
        has_soap = "subjective" in response_lower and "assessment" in response_lower

        # Check for explicit missing data keywords
        missing_keywords = [
            "missing", "insufficient", "clarification",
            "please provide", "could you", "need more", "require additional"
        ]
        has_missing = any(kw in response_lower[:500] for kw in missing_keywords)

        return (has_questions and not has_soap) or has_missing

    def _extract_missing_data(self, response: str) -> List[str]:
        """Extract what data the agent is requesting."""
        missing_data = []
        response_lower = response.lower()

        if "history" in response_lower and ("missing" in response_lower or "provide" in response_lower):
            missing_data.append("history")
        if "exam" in response_lower and ("missing" in response_lower or "provide" in response_lower):
            missing_data.append("physical_exam")
        if "image" in response_lower and ("missing" in response_lower or "provide" in response_lower):
            missing_data.append("image")
        if "age" in response_lower and ("missing" in response_lower or "provide" in response_lower):
            missing_data.append("age")
        if "duration" in response_lower and ("missing" in response_lower or "provide" in response_lower):
            missing_data.append("duration")

        return missing_data

    def _normalize_diagnosis(self, diagnosis: Optional[str]) -> str:
        """Normalize diagnosis for comparison (lowercase, remove punctuation)."""
        if not diagnosis:
            return ""

        normalized = diagnosis.lower()
        # Remove common suffixes
        normalized = re.sub(r'\s*\(.*?\)', '', normalized)  # Remove parentheticals
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
        normalized = normalized.strip()

        return normalized

    def calculate_metrics(
        self,
        results: List[EvaluationResult],
        context_state: str
    ) -> RobustnessMetrics:
        """
        Calculate metrics for a specific context state.

        Args:
            results: List of evaluation results for this context state
            context_state: Context state name

        Returns:
            RobustnessMetrics object
        """
        total = len(results)

        # Filter out errors
        valid_results = [r for r in results if r.error is None]

        if not valid_results:
            return RobustnessMetrics(
                total_cases=total,
                context_state=context_state,
                top1_accuracy=0.0,
                top3_accuracy=0.0,
                avg_confidence=0.0,
                pause_rate=0.0,
                appropriate_pause_rate=0.0,
                avg_differential_overlap=0.0,
                errors=total,
                avg_execution_time_ms=0.0
            )

        # Accuracy metrics
        top1_correct = sum(1 for r in valid_results if r.top1_correct)
        top3_correct = sum(1 for r in valid_results if r.top3_correct)
        top1_accuracy = top1_correct / len(valid_results)
        top3_accuracy = top3_correct / len(valid_results)

        # Average confidence
        avg_confidence = sum(r.confidence for r in valid_results) / len(valid_results)

        # Agentic pause metrics
        pauses = sum(1 for r in valid_results if r.agentic_pause_triggered)
        pause_rate = pauses / len(valid_results)

        # Appropriate pause: pause when context_state != "original"
        appropriate_pauses = sum(
            1 for r in valid_results
            if r.agentic_pause_triggered and r.context_state != "original"
        )
        appropriate_pause_rate = appropriate_pauses / pauses if pauses > 0 else 0.0

        # Execution time
        avg_execution_time_ms = sum(r.execution_time_ms for r in valid_results) / len(valid_results)

        return RobustnessMetrics(
            total_cases=total,
            context_state=context_state,
            top1_accuracy=top1_accuracy,
            top3_accuracy=top3_accuracy,
            avg_confidence=avg_confidence,
            pause_rate=pause_rate,
            appropriate_pause_rate=appropriate_pause_rate,
            avg_differential_overlap=0.0,  # TODO: Calculate overlap with original
            errors=total - len(valid_results),
            avg_execution_time_ms=avg_execution_time_ms
        )

    def calculate_robustness_delta(
        self,
        metrics_by_context: Dict[str, RobustnessMetrics]
    ) -> Dict[str, float]:
        """
        Calculate accuracy delta from original to each context state.

        Robustness Delta = Original Accuracy - Context State Accuracy
        Lower delta = more robust

        Args:
            metrics_by_context: Metrics for each context state

        Returns:
            Dict mapping context_state to accuracy delta
        """
        original_accuracy = metrics_by_context.get("original")
        if not original_accuracy:
            return {}

        deltas = {}
        for state, metrics in metrics_by_context.items():
            if state != "original":
                deltas[state] = original_accuracy.top1_accuracy - metrics.top1_accuracy

        return deltas

    async def run_evaluation(
        self,
        cases_filepath: str,
        output_dir: str = "logs/evaluation"
    ) -> EvaluationReport:
        """
        Run complete evaluation on all cases.

        Args:
            cases_filepath: Path to evaluation_cases.json
            output_dir: Directory to save evaluation reports

        Returns:
            EvaluationReport object
        """
        logger.info("Starting evaluation run")

        # Load cases
        original_cases = self.load_evaluation_cases(cases_filepath)

        # Generate context states for each case
        all_cases = []
        for original_case in original_cases:
            context_variants = self.generate_context_states(original_case)
            all_cases.extend(context_variants)

        logger.info(f"Generated {len(all_cases)} case variants from {len(original_cases)} original cases")

        # Evaluate all cases
        results = []
        for i, case_data in enumerate(all_cases):
            logger.info(f"Evaluating case {i+1}/{len(all_cases)}: {case_data['case_id']}")
            result = await self.evaluate_case(case_data)
            results.append(result)

            # Log progress every 10 cases
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(all_cases)} cases evaluated")

        # Calculate metrics by context state
        context_states = ["original", "history_only", "image_only", "exam_restricted"]
        metrics_by_context = {}

        for state in context_states:
            state_results = [r for r in results if r.context_state == state]
            metrics = self.calculate_metrics(state_results, state)
            metrics_by_context[state] = metrics

            logger.info(f"{state}: Top-1 Accuracy = {metrics.top1_accuracy:.2%}, Pause Rate = {metrics.pause_rate:.2%}")

        # Calculate robustness deltas
        robustness_delta = self.calculate_robustness_delta(metrics_by_context)

        # Calculate overall metrics
        overall_top1 = sum(1 for r in results if r.top1_correct and r.error is None) / len(results)
        overall_top3 = sum(1 for r in results if r.top3_correct and r.error is None) / len(results)

        # Consistency score: how often does the diagnosis stay the same across states?
        # Group results by base case ID
        from collections import defaultdict
        by_base_case = defaultdict(list)
        for r in results:
            base_id = r.case_id.split('_')[0] + '_' + r.case_id.split('_')[1]  # EVAL_001
            by_base_case[base_id].append(r)

        consistency_scores = []
        for base_id, case_results in by_base_case.items():
            diagnoses = [r.predicted_diagnosis for r in case_results if r.predicted_diagnosis]
            if len(diagnoses) > 1:
                # Count how many match the original
                original_diag = next((r.predicted_diagnosis for r in case_results if r.context_state == "original"), None)
                if original_diag:
                    matches = sum(1 for d in diagnoses if self._normalize_diagnosis(d) == self._normalize_diagnosis(original_diag))
                    consistency_scores.append(matches / len(diagnoses))

        consistency_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0

        # Error summary
        errors = [r for r in results if r.error is not None]
        error_details = [
            {"case_id": r.case_id, "context_state": r.context_state, "error": r.error}
            for r in errors
        ]

        # Create report
        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            model_name=self.model_name,
            total_cases=len(original_cases),
            context_states=context_states,
            overall_top1_accuracy=overall_top1,
            overall_top3_accuracy=overall_top3,
            metrics_by_context=metrics_by_context,
            robustness_delta=robustness_delta,
            consistency_score=consistency_score,
            individual_results=results,
            total_errors=len(errors),
            error_details=error_details
        )

        # Save report
        self.save_report(report, output_dir)

        logger.info(f"Evaluation complete: {overall_top1:.2%} top-1 accuracy, {overall_top3:.2%} top-3 accuracy")

        return report

    def save_report(self, report: EvaluationReport, output_dir: str):
        """
        Save evaluation report to disk.

        Args:
            report: EvaluationReport object
            output_dir: Directory to save reports
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON report (full details)
        json_file = output_path / f"evaluation_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            # Convert dataclasses to dicts
            report_dict = {
                "timestamp": report.timestamp,
                "model_name": report.model_name,
                "total_cases": report.total_cases,
                "context_states": report.context_states,
                "overall_top1_accuracy": report.overall_top1_accuracy,
                "overall_top3_accuracy": report.overall_top3_accuracy,
                "metrics_by_context": {k: asdict(v) for k, v in report.metrics_by_context.items()},
                "robustness_delta": report.robustness_delta,
                "consistency_score": report.consistency_score,
                "individual_results": [asdict(r) for r in report.individual_results],
                "total_errors": report.total_errors,
                "error_details": report.error_details
            }
            json.dump(report_dict, f, indent=2)

        logger.info(f"Saved JSON report to {json_file}")

        # Save Markdown summary
        md_file = output_path / f"evaluation_summary_{timestamp}.md"
        with open(md_file, 'w') as f:
            f.write(self._generate_markdown_summary(report))

        logger.info(f"Saved Markdown summary to {md_file}")

    def _generate_markdown_summary(self, report: EvaluationReport) -> str:
        """Generate Markdown summary of evaluation report."""
        md = f"""# MedGemma Clinical Robustness Evaluation Report

**Generated:** {report.timestamp}
**Model:** {report.model_name}
**Total Cases:** {report.total_cases}
**Total Evaluations:** {report.total_cases * 4} (4 context states per case)

---

## Overall Performance

- **Top-1 Accuracy:** {report.overall_top1_accuracy:.2%}
- **Top-3 Accuracy:** {report.overall_top3_accuracy:.2%}
- **Consistency Score:** {report.consistency_score:.2%}
- **Total Errors:** {report.total_errors}

---

## Performance by Context State

| Context State | Top-1 Accuracy | Top-3 Accuracy | Avg Confidence | Pause Rate | Appropriate Pause Rate | Errors |
|---------------|----------------|----------------|----------------|------------|------------------------|--------|
"""

        for state in report.context_states:
            metrics = report.metrics_by_context[state]
            md += f"| {state} | {metrics.top1_accuracy:.2%} | {metrics.top3_accuracy:.2%} | "
            md += f"{metrics.avg_confidence:.3f} | {metrics.pause_rate:.2%} | "
            md += f"{metrics.appropriate_pause_rate:.2%} | {metrics.errors} |\n"

        md += "\n---\n\n## Robustness Analysis\n\n"
        md += "**Accuracy Delta from Original Context:**\n\n"

        for state, delta in report.robustness_delta.items():
            md += f"- **{state}:** -{delta:.2%} (lower is better)\n"

        md += "\n---\n\n## Key Insights\n\n"

        # Identify best/worst context states
        best_state = min(report.robustness_delta, key=report.robustness_delta.get)
        worst_state = max(report.robustness_delta, key=report.robustness_delta.get)

        md += f"- **Most Robust Context:** {best_state} (only {report.robustness_delta[best_state]:.2%} accuracy drop)\n"
        md += f"- **Least Robust Context:** {worst_state} ({report.robustness_delta[worst_state]:.2%} accuracy drop)\n"

        # Agentic behavior analysis
        original_metrics = report.metrics_by_context["original"]
        md += f"- **Agentic Pause Rate (Original):** {original_metrics.pause_rate:.2%}\n"
        md += f"- **Agentic Pause Rate (Incomplete):** {sum(report.metrics_by_context[s].pause_rate for s in ['history_only', 'image_only', 'exam_restricted']) / 3:.2%}\n"

        md += "\n---\n\n## Recommendations\n\n"

        if report.robustness_delta.get("image_only", 0) > 0.2:
            md += "- **Image-Only Context** shows significant accuracy drop. Consider enhancing visual reasoning.\n"

        if original_metrics.pause_rate > 0.1:
            md += "- **High pause rate on complete cases** suggests overcautious behavior. Review triage logic.\n"

        if report.consistency_score < 0.7:
            md += "- **Low consistency across contexts** indicates high variability. Review guideline retrieval.\n"

        md += "\n---\n\n## Error Details\n\n"

        if report.error_details:
            for error in report.error_details[:10]:  # Show first 10 errors
                md += f"- **{error['case_id']}** ({error['context_state']}): {error['error']}\n"
        else:
            md += "*No errors encountered during evaluation.*\n"

        md += "\n---\n\n*Generated by MedGemma Clinical Robustness Evaluator*\n"

        return md


# CLI entry point
async def main():
    """Run evaluation from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate MedGemma Clinical Robustness")
    parser.add_argument(
        "--cases",
        default="data/cases/evaluation_cases.json",
        help="Path to evaluation cases JSON"
    )
    parser.add_argument(
        "--output",
        default="logs/evaluation",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--model",
        default="gemini-pro-latest",
        help="Model to use for orchestration"
    )

    args = parser.parse_args()

    evaluator = ClinicalCaseEvaluator(model_name=args.model, use_medgemma=True)
    report = await evaluator.run_evaluation(args.cases, args.output)

    print(f"\nEvaluation Complete!")
    print(f"Top-1 Accuracy: {report.overall_top1_accuracy:.2%}")
    print(f"Top-3 Accuracy: {report.overall_top3_accuracy:.2%}")
    print(f"Consistency Score: {report.consistency_score:.2%}")
    print(f"Reports saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
