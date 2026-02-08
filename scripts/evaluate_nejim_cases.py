"""
Evaluate NEJIM cases directly without conversion.

This script:
1. Loads cases from NEJIM/image_challenge_input folder
2. Runs evaluation on each context state INDEPENDENTLY
3. NO user interaction needed - fully automatic
4. Saves results to logs/evaluation/

Usage:
    python scripts/evaluate_nejim_cases.py
    python scripts/evaluate_nejim_cases.py --input NEJIM/image_challenge_input --max-cases 5
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import re

from src.agents.adk_agents import create_workflow
from src.utils.schemas import ClinicalCase
from config.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NEJIMEvaluator:
    """Evaluator for NEJIM Image Challenge cases."""

    def __init__(self, model_name: str = "gemini-pro-latest", agent_model: str = "medgemma"):
        """
        Initialize evaluator.

        Args:
            model_name: Gemini model for ADK orchestration
            agent_model: Model for clinical reasoning ("medgemma", "medgemma-vertex", "gemini")
        """
        self.workflow = create_workflow(
            model_name=model_name,
            use_medgemma=True
        )
        # Override agent model choice if specified
        if agent_model != "medgemma":
            self.workflow.agent_model = agent_model
            import src.agents.adk_agents as adk_mod
            adk_mod._agent_model_choice = agent_model
        self.model_name = model_name
        self.agent_model = agent_model

    def find_nejim_cases(self, nejim_dir: str) -> List[str]:
        """Find all case IDs in NEJIM directory."""
        nejim_path = Path(nejim_dir)
        original_files = list(nejim_path.glob("*_original.txt"))

        case_ids = []
        for f in sorted(original_files):
            match = re.match(r'(\d+_\d+_\d+)', f.name)
            if match:
                case_ids.append(match.group(1))

        logger.info(f"Found {len(case_ids)} cases in {nejim_dir}")
        return case_ids

    def load_case_variant(
        self,
        nejim_dir: str,
        case_id: str,
        variant: str
    ) -> Dict[str, Any]:
        """
        Load a specific variant of a case.

        Args:
            nejim_dir: Path to NEJIM folder
            case_id: Case ID (e.g., "01_02_25")
            variant: One of: "original", "history_only", "image_only", "exam_only", "exam_restricted"

        Returns:
            Dict with case data
        """
        nejim_path = Path(nejim_dir)

        # Read the appropriate text file
        if variant == "original":
            text_file = nejim_path / f"{case_id}_original.txt"
        elif variant == "history_only":
            text_file = nejim_path / f"{case_id}_history.txt"
        elif variant == "image_only":
            text_file = nejim_path / f"{case_id}_image_only.txt"
        elif variant == "exam_only":
            text_file = nejim_path / f"{case_id}_exam.txt"
        elif variant == "exam_restricted":
            text_file = nejim_path / f"{case_id}_exam_restricted.txt"
        else:
            raise ValueError(f"Unknown variant: {variant}")

        if not text_file.exists():
            logger.warning(f"File not found: {text_file}, using original")
            text_file = nejim_path / f"{case_id}_original.txt"

        text_content = text_file.read_text().strip()

        # Find image file
        image_path = None
        for ext in ['.jpeg', '.jpg', '.jfif', '.png']:
            img_file = nejim_path / f"{case_id}_img{ext}"
            if img_file.exists():
                image_path = str(img_file)
                break

        # For image_only variant, we have image + minimal text
        # For other variants, we may or may not have image
        use_image = variant in ("image_only", "original") and image_path is not None

        return {
            "case_id": f"NEJIM_{case_id}_{variant}",
            "text": text_content,
            "image_path": image_path if use_image else None,
            "variant": variant,
            "original_case_id": case_id
        }

    async def evaluate_case_variant(
        self,
        case_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single case variant.

        Returns result with:
        - response_text
        - agentic_pause_triggered
        - execution_time_ms
        - error (if any)
        """
        case_id = case_data["case_id"]
        variant = case_data["variant"]

        logger.info(f"Evaluating {case_id}")

        try:
            # Build ClinicalCase
            clinical_case = ClinicalCase(
                case_id=case_id,
                history=case_data["text"],  # Use the text content as history
                physical_exam=None,
                image_data=None,  # TODO: Load actual image if needed
                context_state=variant
            )

            # Run workflow
            start_time = datetime.now()
            result = await self.workflow.run_async(clinical_case)
            end_time = datetime.now()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000

            response_text = result.get("response", "")

            # Detect agentic pause
            agentic_pause = self._detect_pause(response_text)

            return {
                "case_id": case_id,
                "variant": variant,
                "original_case_id": case_data["original_case_id"],
                "response_text": response_text,
                "agentic_pause_triggered": agentic_pause,
                "execution_time_ms": execution_time_ms,
                "error": None,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error evaluating {case_id}: {e}")
            return {
                "case_id": case_id,
                "variant": variant,
                "original_case_id": case_data["original_case_id"],
                "response_text": "",
                "agentic_pause_triggered": False,
                "execution_time_ms": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _detect_pause(self, response: str) -> bool:
        """Detect if agent paused for missing data."""
        response_lower = response.lower()

        # Check for questions without complete SOAP
        has_questions = "?" in response
        has_soap = "subjective" in response_lower and "assessment" in response_lower

        # Check for missing data keywords in first 500 chars,
        # but exclude false positives like "Missing Items: None"
        first_500 = response_lower[:500]

        # Phrases that indicate NO missing data (false positive patterns)
        # Note: first_500 is already lowercased so "None"/"NONE" are matched as "none"
        no_missing_phrases = [
            "missing items: none", "missing items**: none",
            "missing: none", "missing data: none",
            "missing items:** none", "no missing",
            "missing items:\n- none", "missing items:\n*   none",
            "missing items:**\n- none", "missing items:**\n*   none",
            "missing items:\n-   none", "missing items:\n* none",
            "missing data identified**: `patient_age` (note:",
            "flagged it as missing. this was overridden",
            "flagged `patient_age` as missing, but",
            "flagged as missing, but the specialist noted",
            "sufficient data to proceed",
            "sufficient data for analysis",
            "case data is sufficient",
            "has sufficient information",
            "case is complete",
            "data is complete",
        ]
        has_no_missing_statement = any(p in first_500 for p in no_missing_phrases)

        # If triage says "Missing Items: None" or similar AND we have a full SOAP,
        # it's not a real pause
        if has_soap and has_no_missing_statement:
            return False

        # Only flag as missing if keywords appear AND it's not a "none" statement
        pause_keywords = [
            "insufficient", "please provide", "could you provide",
            "need more", "require additional",
            "request clarification", "cannot proceed"
        ]
        has_pause_keyword = any(kw in first_500 for kw in pause_keywords)

        # Check for "missing" keyword only if not negated
        has_missing_flag = "missing" in first_500 and not has_no_missing_statement

        has_missing = has_pause_keyword or has_missing_flag

        return (has_questions and not has_soap) or has_missing

    async def evaluate_all_cases(
        self,
        nejim_dir: str,
        output_dir: str = "logs/evaluation",
        max_cases: int = None
    ):
        """
        Evaluate all NEJIM cases across 4 context states.

        Args:
            nejim_dir: Path to NEJIM folder
            output_dir: Where to save results
            max_cases: Limit to first N cases (for testing)
        """
        # Find all cases
        case_ids = self.find_nejim_cases(nejim_dir)

        if max_cases:
            case_ids = case_ids[:max_cases]
            logger.info(f"Limiting to first {max_cases} cases")

        # Variants to test
        variants = ["original", "history_only", "image_only", "exam_only", "exam_restricted"]

        results = []
        total_evaluations = len(case_ids) * len(variants)

        logger.info(f"Starting evaluation: {len(case_ids)} cases × {len(variants)} variants = {total_evaluations} evaluations")
        logger.info("This will take approximately 1-2 hours. NO USER INTERACTION NEEDED.")

        eval_num = 0
        for case_id in case_ids:
            for variant in variants:
                eval_num += 1
                logger.info(f"\n[{eval_num}/{total_evaluations}] Case {case_id} - {variant}")

                # Load case variant
                case_data = self.load_case_variant(nejim_dir, case_id, variant)

                # Evaluate
                result = await self.evaluate_case_variant(case_data)
                results.append(result)

                # Log progress
                if result["agentic_pause_triggered"]:
                    logger.info(f"  ✓ Agentic pause triggered (expected for incomplete variants)")
                else:
                    logger.info(f"  → Diagnosis provided")

                # Save partial results every 10 cases
                if eval_num % 10 == 0:
                    self._save_partial_results(results, output_dir)

        # Save final results
        self._save_results(results, output_dir, case_ids, variants)

        logger.info(f"\nEvaluation complete! Results saved to {output_dir}")
        return results

    def _save_partial_results(self, results: List[Dict], output_dir: str):
        """Save partial results during evaluation."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        partial_file = output_path / "evaluation_partial.json"
        with open(partial_file, 'w') as f:
            json.dump({"results": results, "count": len(results)}, f, indent=2)

        logger.info(f"  Saved partial results ({len(results)} evaluations)")

    def _save_results(
        self,
        results: List[Dict],
        output_dir: str,
        case_ids: List[str],
        variants: List[str]
    ):
        """Save final results with analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Calculate metrics
        by_variant = {v: [] for v in variants}
        for r in results:
            by_variant[r["variant"]].append(r)

        metrics = {}
        for variant, var_results in by_variant.items():
            successful = [r for r in var_results if r["error"] is None]
            pauses = sum(1 for r in successful if r["agentic_pause_triggered"])
            avg_time = sum(r["execution_time_ms"] for r in successful) / len(successful) if successful else 0

            metrics[variant] = {
                "total": len(var_results),
                "successful": len(successful),
                "errors": len(var_results) - len(successful),
                "pause_count": pauses,
                "pause_rate": pauses / len(successful) if successful else 0,
                "avg_execution_time_ms": avg_time
            }

        # Save JSON
        json_file = output_path / f"nejim_evaluation_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "orchestrator_model": self.model_name,
                    "agent_model": self.agent_model,
                    "total_cases": len(case_ids),
                    "total_evaluations": len(results),
                    "variants": variants
                },
                "metrics": metrics,
                "results": results
            }, f, indent=2)

        logger.info(f"Saved JSON: {json_file}")

        # Save Markdown summary
        md_file = output_path / f"nejim_summary_{timestamp}.md"
        with open(md_file, 'w') as f:
            f.write(f"# NEJIM Evaluation Results\n\n")
            f.write(f"**Date:** {datetime.now().isoformat()}\n")
            f.write(f"**Orchestrator Model:** {self.model_name}\n")
            f.write(f"**Agent Model (Clinical Reasoning):** {self.agent_model}\n")
            f.write(f"**Total Cases:** {len(case_ids)}\n")
            f.write(f"**Total Evaluations:** {len(results)}\n\n")

            f.write("## Metrics by Context State\n\n")
            f.write("| Variant | Evaluations | Successful | Errors | Pause Rate | Avg Time (ms) |\n")
            f.write("|---------|-------------|------------|--------|------------|---------------|\n")

            for variant in variants:
                m = metrics[variant]
                f.write(f"| {variant} | {m['total']} | {m['successful']} | {m['errors']} | ")
                f.write(f"{m['pause_rate']:.1%} | {m['avg_execution_time_ms']:.0f} |\n")

            f.write("\n## Key Findings\n\n")

            original_pause_rate = metrics["original"]["pause_rate"]
            incomplete_variants = ["history_only", "image_only", "exam_only", "exam_restricted"]
            incomplete_pause_rate = sum(
                metrics[v]["pause_rate"] for v in incomplete_variants
            ) / len(incomplete_variants)

            f.write(f"- **Original cases pause rate:** {original_pause_rate:.1%} (should be low)\n")
            f.write(f"- **Incomplete cases pause rate:** {incomplete_pause_rate:.1%} (should be high)\n")

            if original_pause_rate < 0.1 and incomplete_pause_rate > 0.6:
                f.write("\n✓ **GOOD:** Agent appropriately pauses on incomplete data!\n")
            elif original_pause_rate > 0.2:
                f.write("\n⚠ **WARNING:** Agent pausing too often on complete cases\n")
            elif incomplete_pause_rate < 0.5:
                f.write("\n⚠ **WARNING:** Agent not pausing enough on incomplete cases\n")

        logger.info(f"Saved Markdown: {md_file}")


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate NEJIM cases")
    parser.add_argument(
        "--input",
        default="NEJIM/image_challenge_input",
        help="Path to NEJIM folder"
    )
    parser.add_argument(
        "--output",
        default="logs/evaluation",
        help="Output directory"
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Limit to first N cases (for testing)"
    )
    parser.add_argument(
        "--model",
        default="gemini-pro-latest",
        help="Model for orchestration"
    )
    parser.add_argument(
        "--agent-model",
        default="medgemma",
        help="Model for clinical reasoning: medgemma, medgemma-vertex, gemini (default: medgemma)"
    )

    args = parser.parse_args()

    # Build output dir with model name for easy comparison
    agent_model = args.agent_model
    output_dir = args.output
    if output_dir == "logs/evaluation":
        # Auto-name by agent model (e.g., logs/evaluation_medgemma-27b-it)
        model_slug = agent_model.replace("/", "-").replace(" ", "-")
        output_dir = f"logs/evaluation_{model_slug}"

    evaluator = NEJIMEvaluator(model_name=args.model, agent_model=agent_model)
    await evaluator.evaluate_all_cases(
        nejim_dir=args.input,
        output_dir=output_dir,
        max_cases=args.max_cases
    )


if __name__ == "__main__":
    asyncio.run(main())
