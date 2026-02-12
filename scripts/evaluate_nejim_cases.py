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
import time
from typing import Dict, Any, List
from datetime import datetime
import re

from src.agents.adk_agents import create_workflow
from src.agents.registry import MODEL_REGISTRY
from src.utils.schemas import ClinicalCase
from config.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Orchestrator models to try in order. Each has its own daily quota (~100 requests/day).
# When one model's quota is exhausted (429), we automatically switch to the next.
# Ordered: Pro models first (best orchestration quality), then Flash models as fallback.
# Each model has its own daily quota (~100 requests/day = ~25 evals).
ORCHESTRATOR_FALLBACK_MODELS = [
    "gemini-2.5-pro",                     # Pro - best orchestration quality
    "gemini-pro-latest",                  # Pro - proven to work well
    "gemini-3-pro-preview",               # Pro - newest
    "gemini-2.5-flash",                   # Flash - good quality
    "gemini-3-flash-preview",             # Flash - newest
    "gemini-2.5-flash-preview-09-2025",   # Flash - backup
    "gemini-flash-latest",                # Flash - backup
    "gemini-2.0-flash-001",              # Flash - backup
    "gemini-2.0-flash",                   # Flash - lightweight (may loop)
]


class NEJIMEvaluator:
    """Evaluator for NEJIM Image Challenge cases."""

    def __init__(self, model_name: str = "gemini-2.0-flash", agent_model: str = "medgemma"):
        """
        Initialize evaluator.

        Args:
            model_name: Starting Gemini model for ADK orchestration
            agent_model: Model for clinical reasoning ("medgemma", "medgemma-4b", "medgemma-vertex", "gemini")
        """
        self.agent_model = agent_model
        self.current_model_index = 0

        # Always start from the first model in the fallback list
        # The --model arg is ignored; we use the ordered fallback list instead
        self.model_name = ORCHESTRATOR_FALLBACK_MODELS[0]
        self._create_workflow(self.model_name)

        logger.info(f"Orchestrator model: {self.model_name}")
        logger.info(f"Fallback models ({len(ORCHESTRATOR_FALLBACK_MODELS)}): {ORCHESTRATOR_FALLBACK_MODELS}")

    def _create_workflow(self, model_name: str):
        """Create or recreate the workflow with a specific orchestrator model."""
        self.workflow = create_workflow(
            model_name=model_name,
            use_medgemma=True
        )
        if self.agent_model != "medgemma":
            self.workflow.agent_model = self.agent_model
            import src.agents.adk_agents as adk_mod
            adk_mod._agent_model_choice = self.agent_model
        self.model_name = model_name

    def _switch_to_next_model(self) -> bool:
        """Switch to next fallback orchestrator model. Returns False if no more models."""
        self.current_model_index += 1
        if self.current_model_index >= len(ORCHESTRATOR_FALLBACK_MODELS):
            logger.error("All orchestrator models exhausted! No more fallback models available.")
            return False

        next_model = ORCHESTRATOR_FALLBACK_MODELS[self.current_model_index]
        logger.warning(f"Switching orchestrator: {self.model_name} → {next_model}")
        self._create_workflow(next_model)
        return True

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
        case_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate a single case variant with dynamic model fallback on 429 errors.

        If the current orchestrator model hits a daily quota limit (429),
        automatically switches to the next model in ORCHESTRATOR_FALLBACK_MODELS
        and retries. Each model has its own separate daily quota.

        Returns result with:
        - response_text
        - agentic_pause_triggered
        - execution_time_ms
        - orchestrator_model (which model was used)
        - error (if any)
        """
        case_id = case_data["case_id"]
        variant = case_data["variant"]

        logger.info(f"Evaluating {case_id} (orchestrator: {self.model_name})")

        # Build ClinicalCase once (reused across retries)
        clinical_case = ClinicalCase(
            case_id=case_id,
            history=case_data["text"],
            physical_exam=None,
            image_data=None,
            context_state=variant
        )

        # Try current model, then fallback models on 429
        while True:
            try:
                start_time = datetime.now()
                result = await self.workflow.run_async(clinical_case)
                end_time = datetime.now()
                execution_time_ms = (end_time - start_time).total_seconds() * 1000

                response_text = result.get("response", "")

                # Check if the response itself contains a 429 error
                if "429" in response_text and "RESOURCE_EXHAUSTED" in response_text:
                    raise Exception(f"429 RESOURCE_EXHAUSTED in response")

                # Detect agentic pause
                agentic_pause = self._detect_pause(response_text)

                return {
                    "case_id": case_id,
                    "variant": variant,
                    "original_case_id": case_data["original_case_id"],
                    "response_text": response_text,
                    "agentic_pause_triggered": agentic_pause,
                    "execution_time_ms": execution_time_ms,
                    "orchestrator_model": self.model_name,
                    "error": None,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(kw in error_str for kw in [
                    "429", "resource_exhausted", "rate limit",
                    "quota", "too many requests"
                ])

                if is_rate_limit:
                    logger.warning(
                        f"  Quota exhausted for orchestrator '{self.model_name}' on {case_id}"
                    )
                    # Try switching to next model
                    if self._switch_to_next_model():
                        logger.info(f"  Retrying {case_id} with new orchestrator: {self.model_name}")
                        time.sleep(5)  # Brief pause before trying new model
                        continue
                    else:
                        # All models exhausted
                        logger.error(f"  All orchestrator models exhausted on {case_id}")
                        return {
                            "case_id": case_id,
                            "variant": variant,
                            "original_case_id": case_data["original_case_id"],
                            "response_text": "",
                            "agentic_pause_triggered": False,
                            "execution_time_ms": 0.0,
                            "orchestrator_model": self.model_name,
                            "error": f"All orchestrator models exhausted: {str(e)}",
                            "timestamp": datetime.now().isoformat()
                        }
                else:
                    # Non-rate-limit error — don't retry
                    logger.error(f"Error evaluating {case_id}: {e}")
                    return {
                        "case_id": case_id,
                        "variant": variant,
                        "original_case_id": case_data["original_case_id"],
                        "response_text": "",
                        "agentic_pause_triggered": False,
                        "execution_time_ms": 0.0,
                        "orchestrator_model": self.model_name,
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

    def _load_previous_results(self, output_dir: str) -> Dict[str, Dict]:
        """Load previously completed results from partial file for resuming."""
        partial_file = Path(output_dir) / "evaluation_partial.json"
        if not partial_file.exists():
            return {}

        try:
            with open(partial_file) as f:
                data = json.load(f)
            results = data.get("results", [])

            # Index by case_id (e.g., "NEJIM_01_02_25_original") -> result
            completed = {}
            for r in results:
                if r.get("error") is None:
                    completed[r["case_id"]] = r

            logger.info(f"Loaded {len(completed)} previously completed evaluations from {partial_file}")
            return completed
        except Exception as e:
            logger.warning(f"Could not load previous results: {e}")
            return {}

    async def evaluate_all_cases(
        self,
        nejim_dir: str,
        output_dir: str = "logs/evaluation",
        max_cases: int = None,
        skip_cases: int = 0,
        resume: bool = False
    ):
        """
        Evaluate all NEJIM cases across 5 context states.

        Args:
            nejim_dir: Path to NEJIM folder
            output_dir: Where to save results
            max_cases: Limit to first N cases (for testing)
            skip_cases: Skip first N cases (for resuming after quota hit)
            resume: If True, skip cases that already passed in previous partial results
        """
        # Find all cases
        case_ids = self.find_nejim_cases(nejim_dir)

        if skip_cases > 0:
            logger.info(f"Skipping first {skip_cases} cases")
            case_ids = case_ids[skip_cases:]

        if max_cases:
            case_ids = case_ids[:max_cases]
            logger.info(f"Limiting to {max_cases} cases")

        # Load previous results if resuming
        previously_completed = {}
        if resume:
            previously_completed = self._load_previous_results(output_dir)

        # Variants to test
        variants = ["original", "history_only", "image_only", "exam_only", "exam_restricted"]

        results = []
        # Pre-load previously passed results so final output is complete
        if resume and previously_completed:
            results = list(previously_completed.values())
            logger.info(f"Starting with {len(results)} previously completed results")

        total_evaluations = len(case_ids) * len(variants)
        skipped_count = 0

        logger.info(f"Starting evaluation: {len(case_ids)} cases × {len(variants)} variants = {total_evaluations} evaluations")
        if resume:
            logger.info("Resume mode ON — skipping previously completed evaluations")

        eval_num = 0
        for case_id in case_ids:
            for variant in variants:
                eval_num += 1

                # Build the full case_id to check
                full_case_id = f"NEJIM_{case_id}_{variant}"

                # Skip if already completed in previous run
                if resume and full_case_id in previously_completed:
                    skipped_count += 1
                    logger.info(f"[{eval_num}/{total_evaluations}] {full_case_id} — SKIPPED (already passed)")
                    continue

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

                # Save partial results every 5 evaluations
                if (eval_num - skipped_count) % 5 == 0:
                    self._save_partial_results(results, output_dir)

        if skipped_count > 0:
            logger.info(f"\nSkipped {skipped_count} previously completed evaluations")

        # Save final results — use all case_ids for complete report
        all_case_ids = self.find_nejim_cases(nejim_dir)
        self._save_results(results, output_dir, all_case_ids, variants)

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
        # Get full model ID from registry
        agent_model_id = MODEL_REGISTRY.get(self.agent_model, {}).get("model_id", self.agent_model)
        with open(md_file, 'w') as f:
            f.write(f"# NEJIM Evaluation Results\n\n")
            f.write(f"**Date:** {datetime.now().isoformat()}\n")
            f.write(f"**Orchestrator Model:** {self.model_name}\n")
            f.write(f"**Agent Model (Clinical Reasoning):** {agent_model_id}\n")
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
        "--skip-cases",
        type=int,
        default=0,
        help="Skip first N cases (for resuming after quota hit)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run: skip cases that already passed in partial results"
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Model for orchestration (default: gemini-2.0-flash)"
    )
    parser.add_argument(
        "--agent-model",
        default="medgemma",
        help="Model for clinical reasoning: medgemma, medgemma-4b, medgemma-vertex, gemini (default: medgemma)"
    )

    args = parser.parse_args()

    # Build output dir with model name for easy comparison
    agent_model = args.agent_model
    output_dir = args.output
    if output_dir == "logs/evaluation":
        # Auto-name by actual model_id from registry (e.g., logs/evaluation_medgemma-1.5-4b-it)
        model_id = MODEL_REGISTRY.get(agent_model, {}).get("model_id", agent_model)
        model_slug = model_id.replace("/", "-").replace(" ", "-").replace("google/", "")
        output_dir = f"logs/evaluation_{model_slug}"

    evaluator = NEJIMEvaluator(model_name=args.model, agent_model=agent_model)
    await evaluator.evaluate_all_cases(
        nejim_dir=args.input,
        output_dir=output_dir,
        max_cases=args.max_cases,
        skip_cases=args.skip_cases,
        resume=args.resume
    )


if __name__ == "__main__":
    asyncio.run(main())
