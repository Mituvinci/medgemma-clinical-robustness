"""
Stage 1: Extract and split JDCR cases into NEJM-style evaluation format.

Uses Gemini API to read raw text extracted from each JDCR Case Challenge PDF
and produce structured variants matching the evaluation format.

Generates TWO output folders:
  - <output>/jaadcr_input/               (without MCQ options)
  - <output>/jaadcr_input_with_options/   (with MCQ options)

Each folder contains per-case files:
  {case_id}_original.txt
  {case_id}_history.txt
  {case_id}_exam.txt
  {case_id}_exam_restricted.txt
  {case_id}_image_only.txt
  {case_id}_img.jpeg  (copied from extracted images)

Also saves a metadata JSON per case for stage_2 ground truth extraction.

Usage:
    python stage_1_extract_and_split.py --extracted-dir ./backup_extracted --output-dir ./output
    python stage_1_extract_and_split.py --extracted-dir ./backup_extracted --output-dir ./output --max-cases 5
    python stage_1_extract_and_split.py --extracted-dir ./backup_extracted --output-dir ./output --resume
"""

import sys
import os
import json
import time
import shutil
import logging
import argparse
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

import google.generativeai as genai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Excluded prefixes (PII files from Elsevier) ---
EXCLUDED_PREFIXES = ("PIIS2352",)

CASE_PROMPT_SUFFIX = "Act as a dermatologist and determine what is the diagnosis."

# --- Gemini fallback models (rotate on 429 quota errors) ---
GEMINI_MODELS = [
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

SYSTEM_PROMPT = """You are a medical text extraction assistant. Given raw text extracted from a JAAD Case Reports (JDCR) PDF, extract the following structured information.

IMPORTANT RULES:
1. Extract ONLY from the text provided. Do not invent information.
2. The "history" should include patient demographics, presenting complaint, duration, relevant medical history, and physical exam findings as described in the HISTORY section.
3. The "exam_findings" should be ONLY the physical examination findings (what was seen on exam, biopsy results, etc.) WITHOUT patient demographics or history.
4. The "exam_restricted" should be a very brief, vague version of exam findings — remove specific descriptors, colors, patterns. Keep it minimal.
5. For Question 1 ONLY: extract the question stem, all options (A-E), and identify which option is marked as correct (look for "Correct" or "e Correct" near an option).
6. If you cannot find a clear correct answer for Question 1, set correct_letter to null.
7. Extract ONLY Question 1. Ignore Question 2, Question 3, etc.

Return a JSON object with EXACTLY this structure (no markdown fences):
{
    "history": "Full history text including demographics, complaint, duration, exam findings",
    "history_only": "Patient demographics and complaint ONLY, no exam findings. Remove specific skin descriptions.",
    "exam_findings": "Physical exam and biopsy findings only, no patient demographics",
    "exam_restricted": "Very brief vague exam description, stripped of specific details",
    "question_stem": "The Question 1 text exactly as written",
    "options": {
        "A": "Option A text",
        "B": "Option B text",
        "C": "Option C text",
        "D": "Option D text",
        "E": "Option E text"
    },
    "correct_letter": "D",
    "correct_answer_text": "The full text of the correct option",
    "has_ground_truth": true
}
"""


def get_gemini_model(api_key: str, model_idx: int = 0) -> Any:
    """Initialize Gemini model with fallback."""
    genai.configure(api_key=api_key)
    model_name = GEMINI_MODELS[model_idx % len(GEMINI_MODELS)]
    logger.info(f"Using Gemini model: {model_name}")
    return genai.GenerativeModel(
        model_name,
        system_instruction=SYSTEM_PROMPT,
    ), model_idx


def call_gemini_with_fallback(raw_text: str, current_model_idx: int, api_key: str) -> tuple:
    """Call Gemini with automatic model fallback on 429 errors."""
    attempts = 0
    model_idx = current_model_idx

    while attempts < len(GEMINI_MODELS) * 2:
        try:
            model, model_idx = get_gemini_model(api_key, model_idx)
            config = genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=4096,
            )
            response = model.generate_content(
                f"Extract structured data from this JDCR case PDF text:\n\n{raw_text}",
                generation_config=config,
            )
            return response.text, model_idx
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower() or "resource" in error_str.lower():
                logger.warning(
                    f"Quota hit on {GEMINI_MODELS[model_idx % len(GEMINI_MODELS)]}, switching model..."
                )
                model_idx += 1
                attempts += 1
                time.sleep(2)
            else:
                logger.error(f"Gemini error: {e}")
                raise

    raise RuntimeError("All Gemini models exhausted")


def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON from Gemini response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return json.loads(text)


def find_cases(extracted_dir: Path) -> List[str]:
    """Find all valid case folder names, excluding PII files."""
    cases = []
    for d in sorted(extracted_dir.iterdir()):
        if not d.is_dir():
            continue
        if any(d.name.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            logger.info(f"Skipping excluded case: {d.name}")
            continue
        raw_files = list((d / "text").glob("*all_pages_raw.txt"))
        if raw_files:
            cases.append(d.name)
        else:
            logger.warning(f"No all_pages_raw.txt found in {d.name}, skipping")
    return cases


def find_all_images(case_dir: Path) -> List[Path]:
    """Find all images from a case's images folder, deduplicated by file size."""
    images_dir = case_dir / "images"
    if not images_dir.exists():
        return []

    images = []
    for ext in ["*.jpeg", "*.jpg", "*.png"]:
        images.extend(images_dir.glob(ext))

    if not images:
        return []

    images.sort(key=lambda p: p.name)

    # Deduplicate by file size
    seen_sizes = set()
    unique = []
    for img in images:
        size = img.stat().st_size
        if size not in seen_sizes:
            seen_sizes.add(size)
            unique.append(img)

    return unique


def case_id_from_folder(folder_name: str) -> str:
    """Convert folder name like '01_2023_JDCR' to case_id like '01_01_23'.

    JDCR provides month and year only. We use the 1st of the month as day.
    Output format: MM_DD_YY to match the evaluator's regex: r'(\\d+_\\d+_\\d+)'
    """
    m = re.match(r"^(\d{2})_(\d{4})_JDCR$", folder_name)
    if m:
        month = m.group(1)
        year = m.group(2)[2:]  # 2023 -> 23
        return f"{month}_01_{year}"
    return folder_name


def write_case_files(
    case_id: str,
    data: Dict[str, Any],
    image_paths: List[Path],
    output_dir: Path,
    output_dir_options: Path,
):
    """Write all variant files for a case in both output folders."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_options.mkdir(parents=True, exist_ok=True)

    history = data.get("history", "")
    history_only = data.get("history_only", "")
    exam_findings = data.get("exam_findings", "")
    exam_restricted = data.get("exam_restricted", "")
    options = data.get("options", {})

    # Format MCQ options
    options_text = ""
    if options:
        lines = []
        for letter in ["A", "B", "C", "D", "E"]:
            if letter in options:
                lines.append(f"{letter}) {options[letter]}")
        options_text = "\n" + "\n".join(lines)

    # === WITHOUT OPTIONS ===
    variants_no_opts = {
        "original": f"{history} {CASE_PROMPT_SUFFIX}",
        "history": f"{history_only} {CASE_PROMPT_SUFFIX}",
        "exam": f"A patient was noted to have skin changes. {exam_findings} {CASE_PROMPT_SUFFIX}",
        "exam_restricted": f"A patient was noted to have skin changes. {exam_restricted} {CASE_PROMPT_SUFFIX}",
        "image_only": CASE_PROMPT_SUFFIX,
    }

    for variant_name, text in variants_no_opts.items():
        filepath = output_dir / f"{case_id}_{variant_name}.txt"
        filepath.write_text(text.strip())

    # === WITH OPTIONS ===
    variants_with_opts = {
        "original": f"{history} {CASE_PROMPT_SUFFIX}{options_text}",
        "history": f"{history_only} {CASE_PROMPT_SUFFIX}{options_text}",
        "exam": f"A patient was noted to have skin changes. {exam_findings} {CASE_PROMPT_SUFFIX}{options_text}",
        "exam_restricted": f"A patient was noted to have skin changes. {exam_restricted} {CASE_PROMPT_SUFFIX}{options_text}",
        "image_only": f"{CASE_PROMPT_SUFFIX}{options_text}",
    }

    for variant_name, text in variants_with_opts.items():
        filepath = output_dir_options / f"{case_id}_{variant_name}.txt"
        filepath.write_text(text.strip())

    # === Copy ALL images to both output folders ===
    for out_dir in [output_dir, output_dir_options]:
        for idx, img_path in enumerate(image_paths):
            ext = img_path.suffix
            if idx == 0:
                dest = out_dir / f"{case_id}_img{ext}"
            else:
                dest = out_dir / f"{case_id}_img{idx + 1}{ext}"
            if not dest.exists():
                shutil.copy2(img_path, dest)


def process_case(
    folder_name: str,
    extracted_dir: Path,
    model_idx: int,
    api_key: str,
    output_dir: Path,
    output_dir_options: Path,
) -> tuple:
    """Process a single case. Returns (metadata_dict, new_model_idx)."""
    case_dir = extracted_dir / folder_name
    case_id = case_id_from_folder(folder_name)

    raw_files = list((case_dir / "text").glob("*all_pages_raw.txt"))
    raw_text = raw_files[0].read_text()

    logger.info(f"Processing {folder_name} -> case_id={case_id} ({len(raw_text)} chars)")

    response_text, model_idx = call_gemini_with_fallback(raw_text, model_idx, api_key)
    data = parse_json_response(response_text)

    image_paths = find_all_images(case_dir)
    write_case_files(case_id, data, image_paths, output_dir, output_dir_options)

    metadata = {
        "case_id": case_id,
        "folder_name": folder_name,
        "has_ground_truth": data.get("has_ground_truth", False),
        "correct_letter": data.get("correct_letter"),
        "correct_answer_text": data.get("correct_answer_text", ""),
        "question_stem": data.get("question_stem", ""),
        "options": data.get("options", {}),
        "image_files": [p.name for p in image_paths],
        "image_count": len(image_paths),
    }

    return metadata, model_idx


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Extract and split JDCR cases into NEJM-style evaluation format"
    )
    parser.add_argument(
        "--extracted-dir",
        type=str,
        required=True,
        help="Path to extracted case folders (each with text/ and images/ subdirs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory. Creates jaadcr_input/ and jaadcr_input_with_options/ inside it",
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default=None,
        help="Directory for per-case metadata JSON files (default: <output-dir>/case_metadata)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("GOOGLE_API_KEY", ""),
        help="Google API key for Gemini (default: GOOGLE_API_KEY env var)",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="Limit to N cases (0=all)")
    parser.add_argument("--resume", action="store_true", help="Skip already processed cases")
    parser.add_argument("--skip-cases", type=int, default=0, help="Skip first N cases")
    args = parser.parse_args()

    if not args.api_key:
        logger.error("GOOGLE_API_KEY not set. Use --api-key or set the environment variable.")
        sys.exit(1)

    extracted_dir = Path(args.extracted_dir)
    if not extracted_dir.exists():
        logger.error(f"Extracted directory not found: {extracted_dir}")
        sys.exit(1)

    base_output = Path(args.output_dir)
    output_dir = base_output / "jaadcr_input"
    output_dir_options = base_output / "jaadcr_input_with_options"
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else base_output / "case_metadata"

    logger.info("=" * 70)
    logger.info("STAGE 1: Extract and split JDCR cases into NEJM-style format")
    logger.info(f"  Extracted dir: {extracted_dir}")
    logger.info(f"  Output dir:    {base_output}")
    logger.info(f"  Metadata dir:  {metadata_dir}")
    logger.info("=" * 70)

    cases = find_cases(extracted_dir)
    logger.info(f"Found {len(cases)} valid cases")

    if args.skip_cases > 0:
        cases = cases[args.skip_cases:]
        logger.info(f"Skipping first {args.skip_cases}, {len(cases)} remaining")

    if args.max_cases > 0:
        cases = cases[: args.max_cases]
        logger.info(f"Limiting to {len(cases)} cases")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_options.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    already_done = set()
    if args.resume:
        for f in metadata_dir.glob("*.json"):
            already_done.add(f.stem)
        logger.info(f"Resume mode: {len(already_done)} cases already done")

    model_idx = 0
    all_metadata = []
    success = 0
    failed = 0

    for i, folder_name in enumerate(cases):
        case_id = case_id_from_folder(folder_name)

        if args.resume and case_id in already_done:
            logger.info(f"[{i+1}/{len(cases)}] Skipping {case_id} (already done)")
            meta_file = metadata_dir / f"{case_id}.json"
            if meta_file.exists():
                all_metadata.append(json.loads(meta_file.read_text()))
            continue

        logger.info(f"[{i+1}/{len(cases)}] Processing: {folder_name}")

        try:
            metadata, model_idx = process_case(
                folder_name, extracted_dir, model_idx, args.api_key,
                output_dir, output_dir_options,
            )

            meta_file = metadata_dir / f"{case_id}.json"
            meta_file.write_text(json.dumps(metadata, indent=2))
            all_metadata.append(metadata)

            success += 1
            logger.info(
                f"  -> OK: ground_truth={'YES' if metadata['has_ground_truth'] else 'NO'}, "
                f"answer={metadata['correct_letter'] or 'N/A'}"
            )

        except Exception as e:
            failed += 1
            logger.error(f"  -> FAILED: {e}")

        time.sleep(1)

    # Save combined metadata
    combined_file = metadata_dir / "all_cases_metadata.json"
    combined_file.write_text(json.dumps(all_metadata, indent=2))

    gt_count = sum(1 for m in all_metadata if m.get("has_ground_truth"))
    logger.info("")
    logger.info("=" * 70)
    logger.info("STAGE 1 COMPLETE")
    logger.info(f"  Total cases processed: {success}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Cases with ground truth: {gt_count}")
    logger.info(f"  Output (without options): {output_dir}")
    logger.info(f"  Output (with options):    {output_dir_options}")
    logger.info(f"  Metadata:                 {metadata_dir}")
    logger.info("")
    logger.info("Next step: python stage_2_build_ground_truth.py --metadata-dir <metadata-dir> --output-csv <path>")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
