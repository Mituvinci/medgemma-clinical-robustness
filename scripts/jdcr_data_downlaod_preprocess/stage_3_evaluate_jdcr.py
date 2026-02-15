"""
Stage 3: Evaluate JDCR cases using MedGemma multi-agent workflow.

Based on scripts/evaluate_nejim_cases.py but adapted for JDCR:
  - Handles multiple images per case
  - Uses JDCR_ prefix for case IDs
  - Reads from any input directory with NEJM-style case files

Usage:
    # Test with 1 case
    python stage_3_evaluate_jdcr.py \\
        --input ./jaadcr_input \\
        --agent-model medgemma \\
        --max-cases 1

    # Full evaluation (without options)
    python stage_3_evaluate_jdcr.py \\
        --input ./jaadcr_input \\
        --agent-model medgemma \\
        --output ./logs/evaluation_jdcr

    # Full evaluation (with options)
    python stage_3_evaluate_jdcr.py \\
        --input ./jaadcr_input_with_options \\
        --agent-model medgemma-vertex \\
        --output ./logs/evaluation_jdcr_with_options

Run AFTER stage_1 and stage_2.
"""

import sys
import os
import asyncio
import json
import logging
import time
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Determine project root dynamically: walk up from this script until we find src/
_script_path = Path(__file__).resolve()
_project_root = None
for parent in _script_path.parents:
    if (parent / "src").is_dir() and (parent / "config").is_dir():
        _project_root = parent
        break
if _project_root is None:
    _project_root = _script_path.parent.parent.parent
sys.path.insert(0, str(_project_root))

from src.agents.adk_agents import create_workflow
from src.agents.registry import MODEL_REGISTRY
from src.utils.schemas import ClinicalCase
from config.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Orchestrator fallback chain (rotated on 429 quota errors)
ORCHESTRATOR_FALLBACK_MODELS = [
    "gemini-2.5-pro",
    "gemini-pro-latest",
    "gemini-3-pro-preview",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-flash-latest",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash",
]


class JDCREvaluator:
    """Evaluator for JDCR (JAAD Case Reports) cases."""

    def __init__(self, model_name: str = "gemini-2.0-flash", agent_model: str = "medgemma"):
        self.agent_model = agent_model
        self.current_model_index = 0
        self.model_name = ORCHESTRATOR_FALLBACK_MODELS[0]
        self._create_workflow(self.model_name)

        logger.info(f"Orchestrator model: {self.model_name}")
        logger.info(f"Agent model: {self.agent_model}")
        logger.info(
            f"Fallback models ({len(ORCHESTRATOR_FALLBACK_MODELS)}): "
            f"{ORCHESTRATOR_FALLBACK_MODELS}"
        )

    def _create_workflow(self, model_name: str):
        """Create or recreate the workflow with a specific orchestrator model."""
        self.workflow = create_workflow(model_name=model_name, use_medgemma=True)
        if self.agent_model != "medgemma":
            self.workflow.agent_model = self.agent_model
            import src.agents.adk_agents as adk_mod
            adk_mod._agent_model_choice = self.agent_model
        self.model_name = model_name

    def _switch_to_next_model(self) -> bool:
        """Switch to next fallback orchestrator model."""
        self.current_model_index += 1
        if self.current_model_index >= len(ORCHESTRATOR_FALLBACK_MODELS):
            logger.error("All orchestrator models exhausted!")
            return False
        next_model = ORCHESTRATOR_FALLBACK_MODELS[self.current_model_index]
        logger.warning(f"Switching orchestrator: {self.model_name} -> {next_model}")
        self._create_workflow(next_model)
        return True

    def find_jdcr_cases(self, input_dir: str) -> List[str]:
        """Find all case IDs in input directory."""
        input_path = Path(input_dir)
        original_files = list(input_path.glob("*_original.txt"))
        case_ids = []
        for f in sorted(original_files):
            match = re.match(r"(\d+_\d+_\d+)", f.name)
            if match:
                case_ids.append(match.group(1))
        logger.info(f"Found {len(case_ids)} cases in {input_dir}")
        return case_ids

    def load_case_variant(
        self, input_dir: str, case_id: str, variant: str
    ) -> Dict[str, Any]:
        """Load a specific variant of a case, including multiple images."""
        input_path = Path(input_dir)
        variant_to_file = {
            "original": f"{case_id}_original.txt",
            "history_only": f"{case_id}_history.txt",
            "image_only": f"{case_id}_image_only.txt",
            "exam_only": f"{case_id}_exam.txt",
            "exam_restricted": f"{case_id}_exam_restricted.txt",
        }

        text_file = input_path / variant_to_file.get(variant, f"{case_id}_original.txt")
        if not text_file.exists():
            logger.warning(f"File not found: {text_file}, using original")
            text_file = input_path / f"{case_id}_original.txt"

        text_content = text_file.read_text().strip()

        # Find ALL image files for this case
        image_paths = []
        for ext in [".jpeg", ".jpg", ".jfif", ".png"]:
            first_img = input_path / f"{case_id}_img{ext}"
            if first_img.exists():
                image_paths.append(str(first_img))
            for i in range(2, 10):
                extra_img = input_path / f"{case_id}_img{i}{ext}"
                if extra_img.exists():
                    image_paths.append(str(extra_img))

        use_images = variant in ("image_only", "original") and len(image_paths) > 0

        return {
            "case_id": f"JDCR_{case_id}_{variant}",
            "text": text_content,
            "image_paths": image_paths if use_images else [],
            "image_path": image_paths[0] if use_images and image_paths else None,
            "variant": variant,
            "original_case_id": case_id,
            "image_count": len(image_paths),
        }

    async def evaluate_case_variant(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single case variant with model fallback on 429 errors."""
        case_id = case_data["case_id"]
        variant = case_data["variant"]

        logger.info(
            f"Evaluating {case_id} (orchestrator: {self.model_name}, "
            f"images: {case_data['image_count']})"
        )

        clinical_case = ClinicalCase(
            case_id=case_id,
            history=case_data["text"],
            physical_exam=None,
            image_data=None,
            image_path=case_data.get("image_path"),
            context_state=variant,
        )

        while True:
            try:
                start_time = datetime.now()
                result = await self.workflow.run_async(clinical_case)
                end_time = datetime.now()
                execution_time_ms = (end_time - start_time).total_seconds() * 1000

                response_text = result.get("response", "")
                if "429" in response_text and "RESOURCE_EXHAUSTED" in response_text:
                    raise Exception("429 RESOURCE_EXHAUSTED in response")

                agentic_pause = self._detect_pause(response_text)

                return {
                    "case_id": case_id,
                    "variant": variant,
                    "original_case_id": case_data["original_case_id"],
                    "response_text": response_text,
                    "agentic_pause_triggered": agentic_pause,
                    "execution_time_ms": execution_time_ms,
                    "orchestrator_model": self.model_name,
                    "image_count": case_data["image_count"],
                    "error": None,
                    "timestamp": datetime.now().isoformat(),
                }

            except Exception as e:
                error_str = str(e).lower()
                if "429" in str(e) or "resource_exhausted" in error_str or "quota" in error_str:
                    logger.warning(f"  Quota hit: {e}")
                    if not self._switch_to_next_model():
                        return {
                            "case_id": case_id,
                            "variant": variant,
                            "original_case_id": case_data["original_case_id"],
                            "response_text": "",
                            "agentic_pause_triggered": False,
                            "execution_time_ms": 0,
                            "orchestrator_model": self.model_name,
                            "image_count": case_data["image_count"],
                            "error": f"All models exhausted: {e}",
                            "timestamp": datetime.now().isoformat(),
                        }
                    continue
                else:
                    logger.error(f"  Error: {e}")
                    return {
                        "case_id": case_id,
                        "variant": variant,
                        "original_case_id": case_data["original_case_id"],
                        "response_text": "",
                        "agentic_pause_triggered": False,
                        "execution_time_ms": 0,
                        "orchestrator_model": self.model_name,
                        "image_count": case_data["image_count"],
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }

    def _detect_pause(self, response: str) -> bool:
        """Detect agentic pause (same logic as NEJM evaluator)."""
        if not response or len(response) < 10:
            return True

        response_lower = response.lower()
        has_questions = "?" in response
        has_soap = "subjective" in response_lower and "assessment" in response_lower

        first_500 = response_lower[:500]

        no_missing_phrases = [
            "missing items: none", "missing items**: none",
            "missing: none", "missing data: none",
            "missing items:** none", "no missing",
            "missing items:\n- none", "missing items:\n*   none",
            "sufficient data to proceed", "sufficient data for analysis",
            "case data is sufficient", "has sufficient information",
            "case is complete", "data is complete",
        ]
        has_no_missing_statement = any(p in first_500 for p in no_missing_phrases)

        if has_soap and has_no_missing_statement:
            return False

        pause_keywords = [
            "insufficient", "please provide", "could you provide",
            "need more", "require additional",
            "request clarification", "cannot proceed",
        ]
        has_pause_keyword = any(kw in first_500 for kw in pause_keywords)
        has_missing_flag = "missing" in first_500 and not has_no_missing_statement
        has_missing = has_pause_keyword or has_missing_flag

        return (has_questions and not has_soap) or has_missing

    def _load_previous_results(self, output_dir: str) -> Dict[str, Dict]:
        """Load previously completed results for resuming."""
        partial_file = Path(output_dir) / "evaluation_partial.json"
        if not partial_file.exists():
            return {}
        try:
            with open(partial_file) as f:
                data = json.load(f)
            completed = {}
            for r in data.get("results", []):
                if r.get("error") is None:
                    completed[r["case_id"]] = r
            logger.info(f"Loaded {len(completed)} previously completed evaluations")
            return completed
        except Exception as e:
            logger.warning(f"Could not load previous results: {e}")
            return {}

    async def evaluate_all_cases(
        self,
        input_dir: str,
        output_dir: str = "logs/evaluation_jdcr",
        max_cases: int = None,
        skip_cases: int = 0,
        resume: bool = False,
    ):
        """Evaluate all cases across 5 context variants."""
        case_ids = self.find_jdcr_cases(input_dir)

        if skip_cases > 0:
            logger.info(f"Skipping first {skip_cases} cases")
            case_ids = case_ids[skip_cases:]

        if max_cases:
            case_ids = case_ids[:max_cases]
            logger.info(f"Limiting to {max_cases} cases")

        previously_completed = {}
        if resume:
            previously_completed = self._load_previous_results(output_dir)

        variants = ["original", "history_only", "image_only", "exam_only", "exam_restricted"]

        results = []
        if resume and previously_completed:
            results = list(previously_completed.values())
            logger.info(f"Starting with {len(results)} previously completed results")

        total_evaluations = len(case_ids) * len(variants)
        skipped_count = 0

        logger.info(
            f"Starting JDCR evaluation: {len(case_ids)} cases x "
            f"{len(variants)} variants = {total_evaluations} evaluations"
        )

        eval_num = 0
        for case_id in case_ids:
            for variant in variants:
                eval_num += 1
                full_case_id = f"JDCR_{case_id}_{variant}"

                if resume and full_case_id in previously_completed:
                    skipped_count += 1
                    logger.info(
                        f"[{eval_num}/{total_evaluations}] {full_case_id} -- SKIPPED (already done)"
                    )
                    continue

                logger.info(f"\n[{eval_num}/{total_evaluations}] Case {case_id} - {variant}")

                case_data = self.load_case_variant(input_dir, case_id, variant)
                result = await self.evaluate_case_variant(case_data)
                results.append(result)

                if result["agentic_pause_triggered"]:
                    logger.info("  -> Agentic pause triggered")
                else:
                    logger.info("  -> Diagnosis provided")

                if (eval_num - skipped_count) % 5 == 0:
                    self._save_partial_results(results, output_dir)

        if skipped_count > 0:
            logger.info(f"\nSkipped {skipped_count} previously completed evaluations")

        all_case_ids = self.find_jdcr_cases(input_dir)
        self._save_results(results, output_dir, all_case_ids, variants)

        logger.info(f"\nJDCR evaluation complete! Results saved to {output_dir}")
        return results

    def _save_partial_results(self, results: List[Dict], output_dir: str):
        """Save partial results during evaluation."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        partial_file = output_path / "evaluation_partial.json"
        with open(partial_file, "w") as f:
            json.dump({"results": results, "count": len(results)}, f, indent=2)
        logger.info(f"  Saved partial results ({len(results)} evaluations)")

    def _save_results(
        self,
        results: List[Dict],
        output_dir: str,
        case_ids: List[str],
        variants: List[str],
    ):
        """Save final results with analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        by_variant = {v: [] for v in variants}
        for r in results:
            by_variant[r["variant"]].append(r)

        metrics = {}
        for variant, var_results in by_variant.items():
            successful = [r for r in var_results if r["error"] is None]
            pauses = sum(1 for r in successful if r["agentic_pause_triggered"])
            avg_time = (
                sum(r["execution_time_ms"] for r in successful) / len(successful)
                if successful
                else 0
            )
            metrics[variant] = {
                "total": len(var_results),
                "successful": len(successful),
                "errors": len(var_results) - len(successful),
                "pause_count": pauses,
                "pause_rate": pauses / len(successful) if successful else 0,
                "avg_execution_time_ms": avg_time,
            }

        # Save JSON
        json_file = output_path / f"jdcr_evaluation_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "dataset": "JAADCR",
                        "orchestrator_model": self.model_name,
                        "agent_model": self.agent_model,
                        "total_cases": len(case_ids),
                        "total_evaluations": len(results),
                        "variants": variants,
                    },
                    "metrics": metrics,
                    "results": results,
                },
                f,
                indent=2,
            )
        logger.info(f"Saved JSON: {json_file}")

        # Save Markdown summary
        md_file = output_path / f"jdcr_summary_{timestamp}.md"
        agent_model_id = MODEL_REGISTRY.get(self.agent_model, {}).get(
            "model_id", self.agent_model
        )
        with open(md_file, "w") as f:
            f.write("# JDCR (JAADCR) Evaluation Results\n\n")
            f.write(f"**Date:** {datetime.now().isoformat()}\n")
            f.write(f"**Orchestrator Model:** {self.model_name}\n")
            f.write(f"**Agent Model (Clinical Reasoning):** {agent_model_id}\n")
            f.write(f"**Total Cases:** {len(case_ids)}\n")
            f.write(f"**Total Evaluations:** {len(results)}\n\n")

            f.write("## Metrics by Context State\n\n")
            f.write(
                "| Variant | Evaluations | Successful | Errors | Pause Rate | Avg Time (ms) |\n"
            )
            f.write(
                "|---------|-------------|------------|--------|------------|---------------|\n"
            )

            for variant in variants:
                m = metrics[variant]
                f.write(
                    f"| {variant} | {m['total']} | {m['successful']} | {m['errors']} | "
                    f"{m['pause_rate']:.1%} | {m['avg_execution_time_ms']:.0f} |\n"
                )

            f.write("\n## Key Findings\n\n")
            original_pause_rate = metrics["original"]["pause_rate"]
            incomplete_variants = ["history_only", "image_only", "exam_only", "exam_restricted"]
            incomplete_pause_rate = sum(
                metrics[v]["pause_rate"] for v in incomplete_variants
            ) / len(incomplete_variants)

            f.write(f"- **Original cases pause rate:** {original_pause_rate:.1%} (should be low)\n")
            f.write(
                f"- **Incomplete cases pause rate:** {incomplete_pause_rate:.1%} (should be high)\n"
            )

            if original_pause_rate < 0.1 and incomplete_pause_rate > 0.6:
                f.write("\n**GOOD:** Agent appropriately pauses on incomplete data!\n")
            elif original_pause_rate > 0.2:
                f.write("\n**WARNING:** Agent pausing too often on complete cases\n")
            elif incomplete_pause_rate < 0.5:
                f.write("\n**WARNING:** Agent not pausing enough on incomplete cases\n")

        logger.info(f"Saved Markdown: {md_file}")


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stage 3: Evaluate JDCR cases with MedGemma multi-agent workflow"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JDCR input folder containing case variant files",
    )
    parser.add_argument(
        "--output",
        default="logs/evaluation_jdcr",
        help="Output directory for results (default: logs/evaluation_jdcr)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Limit to first N cases (for testing)",
    )
    parser.add_argument(
        "--skip-cases",
        type=int,
        default=0,
        help="Skip first N cases",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous partial results",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash",
        help="Orchestrator model (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--agent-model",
        default="medgemma",
        help="Clinical reasoning model: medgemma, medgemma-4b, medgemma-vertex, "
             "medgemma-4b-it-vertex, etc.",
    )
    args = parser.parse_args()

    agent_model = args.agent_model
    output_dir = args.output
    if output_dir == "logs/evaluation_jdcr":
        model_id = MODEL_REGISTRY.get(agent_model, {}).get("model_id", agent_model)
        model_slug = model_id.replace("/", "-").replace(" ", "-").replace("google/", "")
        output_dir = f"logs/evaluation_jaadcr_{model_slug}"

    evaluator = JDCREvaluator(model_name=args.model, agent_model=agent_model)
    await evaluator.evaluate_all_cases(
        input_dir=args.input,
        output_dir=output_dir,
        max_cases=args.max_cases,
        skip_cases=args.skip_cases,
        resume=args.resume,
    )


if __name__ == "__main__":
    asyncio.run(main())
