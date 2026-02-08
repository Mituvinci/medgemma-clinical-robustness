"""
Load NEJIM cases into evaluation format.

Converts NEJIM folder structure to evaluation_cases.json format.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import os
from typing import Dict, Any, List
import re


def parse_case_id(filename: str) -> str:
    """Extract case ID from filename like 01_02_25_original.txt"""
    match = re.match(r'(\d+_\d+_\d+)', filename)
    return match.group(1) if match else "unknown"


def load_nejim_cases(nejim_dir: str) -> List[Dict[str, Any]]:
    """
    Load all NEJIM cases from directory.

    Expected structure:
    - MM_DD_YY_original.txt (complete case)
    - MM_DD_YY_history.txt (history only)
    - MM_DD_YY_exam.txt (exam details)
    - MM_DD_YY_exam_restricted.txt (minimal exam)
    - MM_DD_YY_image_only.txt (just image prompt)
    - MM_DD_YY_img.jpeg or .jfif (image file)

    Args:
        nejim_dir: Path to NEJIM/image_challenge_input folder

    Returns:
        List of case dictionaries
    """
    cases = []
    nejim_path = Path(nejim_dir)

    # Find all original.txt files (one per case)
    original_files = list(nejim_path.glob("*_original.txt"))

    print(f"Found {len(original_files)} NEJIM cases in {nejim_dir}")

    for original_file in sorted(original_files):
        case_id = parse_case_id(original_file.name)

        # Read all variants
        original_text = original_file.read_text().strip()

        history_file = nejim_path / f"{case_id}_history.txt"
        history_text = history_file.read_text().strip() if history_file.exists() else None

        exam_file = nejim_path / f"{case_id}_exam.txt"
        exam_text = exam_file.read_text().strip() if exam_file.exists() else None

        exam_restricted_file = nejim_path / f"{case_id}_exam_restricted.txt"
        exam_restricted_text = exam_restricted_file.read_text().strip() if exam_restricted_file.exists() else None

        image_only_file = nejim_path / f"{case_id}_image_only.txt"
        image_only_text = image_only_file.read_text().strip() if image_only_file.exists() else None

        # Find image file
        image_file = None
        for ext in ['.jpeg', '.jpg', '.jfif', '.png']:
            img_path = nejim_path / f"{case_id}_img{ext}"
            if img_path.exists():
                image_file = str(img_path)
                break

        # Extract diagnosis from filename if available
        # (You might need to manually add gold standard diagnoses)
        gold_diagnosis = "Unknown"  # TODO: Add manually or from answer key

        case = {
            "case_id": f"NEJIM_{case_id}",
            "original_text": original_text,
            "history_text": history_text,
            "exam_text": exam_text,
            "exam_restricted_text": exam_restricted_text,
            "image_only_text": image_only_text,
            "image_path": image_file,
            "gold_standard_diagnosis": gold_diagnosis,
            "differential_diagnoses": [],
            "source": "NEJIM_image_challenge"
        }

        cases.append(case)
        print(f"  Loaded: {case_id} (image: {'yes' if image_file else 'no'})")

    return cases


def save_evaluation_cases(cases: List[Dict[str, Any]], output_file: str):
    """Save cases to evaluation_cases.json format."""
    output_data = {
        "cases": cases,
        "metadata": {
            "version": "1.0",
            "created_date": "2026-02-07",
            "description": "NEJIM Image Challenge cases for MedGemma evaluation",
            "total_cases": len(cases),
            "source": "NEJIM/image_challenge_input",
            "context_variations": [
                "original",
                "history_only",
                "image_only",
                "exam_restricted"
            ]
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(cases)} cases to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load NEJIM cases for evaluation")
    parser.add_argument(
        "--input",
        default="NEJIM/image_challenge_input",
        help="Path to NEJIM input folder"
    )
    parser.add_argument(
        "--output",
        default="data/cases/nejim_evaluation_cases.json",
        help="Output JSON file"
    )

    args = parser.parse_args()

    cases = load_nejim_cases(args.input)
    save_evaluation_cases(cases, args.output)

    print("\nNext steps:")
    print("1. Add gold standard diagnoses to the JSON file")
    print(f"2. Run evaluation: python src/evaluation/evaluator.py --cases {args.output}")
