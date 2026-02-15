"""
Stage 2: Build ground truth CSV from stage_1 metadata.

Reads the per-case metadata JSON files produced by stage_1 and generates a
ground truth CSV in the same format as NEJM:
    Date,Ground_Truth
    03_01_24,Leprosy

Usage:
    python stage_2_build_ground_truth.py --metadata-dir ./output/case_metadata --output-csv ./output/JAADCR_Groundtruth.csv
    python stage_2_build_ground_truth.py --metadata-dir ./output/case_metadata --output-csv ./output/JAADCR_Groundtruth.csv --list-all

Run AFTER stage_1_extract_and_split.py
"""

import sys
import json
import csv
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Build ground truth CSV from stage_1 metadata"
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        required=True,
        help="Directory containing per-case metadata JSON files from stage_1",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path for the output ground truth CSV file",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=0,
        help="Minimum cases with ground truth required (0=no check)",
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="List all cases with their ground truth status",
    )
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir)
    output_csv = Path(args.output_csv)

    if not metadata_dir.exists():
        logger.error(f"Metadata directory not found: {metadata_dir}")
        logger.error("Run stage_1_extract_and_split.py first!")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("STAGE 2: Build ground truth CSV from stage_1 metadata")
    logger.info(f"  Metadata dir: {metadata_dir}")
    logger.info(f"  Output CSV:   {output_csv}")
    logger.info("=" * 70)

    meta_files = sorted(metadata_dir.glob("*.json"))
    meta_files = [f for f in meta_files if f.name != "all_cases_metadata.json"]

    if not meta_files:
        logger.error(f"No metadata files found in {metadata_dir}")
        sys.exit(1)

    logger.info(f"Found {len(meta_files)} case metadata files")

    cases_with_gt = []
    cases_without_gt = []

    for meta_file in meta_files:
        data = json.loads(meta_file.read_text())
        case_id = data["case_id"]
        has_gt = data.get("has_ground_truth", False)
        correct_letter = data.get("correct_letter")
        correct_text = data.get("correct_answer_text", "")

        if has_gt and correct_letter and correct_text:
            cases_with_gt.append({
                "case_id": case_id,
                "correct_letter": correct_letter,
                "correct_answer_text": correct_text,
            })
        else:
            cases_without_gt.append(case_id)

    logger.info(f"Cases WITH ground truth: {len(cases_with_gt)}")
    logger.info(f"Cases WITHOUT ground truth: {len(cases_without_gt)}")

    if args.list_all:
        logger.info("")
        logger.info("--- Cases with ground truth ---")
        for c in cases_with_gt:
            logger.info(f"  {c['case_id']}: {c['correct_letter']}) {c['correct_answer_text']}")
        logger.info("")
        logger.info("--- Cases without ground truth ---")
        for c_id in cases_without_gt:
            logger.info(f"  {c_id}")

    if args.min_cases > 0 and len(cases_with_gt) < args.min_cases:
        logger.error(
            f"Only {len(cases_with_gt)} cases with ground truth, need {args.min_cases}"
        )
        sys.exit(1)

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Ground_Truth"])
        for case in cases_with_gt:
            writer.writerow([case["case_id"], case["correct_answer_text"]])

    logger.info(f"\nGround truth CSV written: {output_csv}")
    logger.info(f"  Total entries: {len(cases_with_gt)}")

    if cases_without_gt:
        logger.info(f"\nNote: {len(cases_without_gt)} cases lack ground truth (not in CSV)")

    logger.info("")
    logger.info("=" * 70)
    logger.info("STAGE 2 COMPLETE")
    logger.info(f"  Ground truth CSV: {output_csv}")
    logger.info(f"  Cases ready for evaluation: {len(cases_with_gt)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
