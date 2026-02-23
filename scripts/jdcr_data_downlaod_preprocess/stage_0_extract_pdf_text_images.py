"""
Step 0: Extract Raw Text and Images from JDCR Case Challenge PDFs.

Uses PyMuPDF (fitz) to extract the full text and all embedded clinical images
from each JDCR Case Challenge PDF. This is the first step in the JDCR data
pipeline and produces the `backup_extracted/` folder that is consumed by
stage_1_extract_and_split.py.

Input:
    A folder of JDCR Case Challenge PDFs, named in the format MM_YYYY_JDCR.pdf
    Example:
        pdf_input/
            01_2023_JDCR.pdf
            02_2023_JDCR.pdf
            ...

Output (one subfolder per PDF inside --output-dir):
    backup_extracted/
        01_2023_JDCR/
            text/
                01_2023_JDCR_all_pages_raw.txt    <- full PDF text (all pages combined)
                01_2023_JDCR_page01_raw.txt        <- per-page text (for debugging)
                01_2023_JDCR_page02_raw.txt
                ...
            images/
                01_2023_JDCR_p01_img01.jpeg        <- clinical images extracted from PDF
                01_2023_JDCR_p02_img01.jpeg
                ...
        02_2023_JDCR/
            text/  ...
            images/  ...

The combined *_all_pages_raw.txt file is what stage_1_extract_and_split.py reads.
Images are filtered by size (>= 15 KB) and deduplicated by content hash to
skip logos, icons, and duplicate embedded copies.

Requirements:
    pip install PyMuPDF

Usage:
    python 0_extract_pdf_text_images.py --input ./pdf_input --output ./backup_extracted
    python 0_extract_pdf_text_images.py --input ./pdf_input --output ./backup_extracted --min-image-bytes 20000

Next step:
    python stage_1_extract_and_split.py --extracted-dir ./backup_extracted --output-dir ./output
"""

import os
import re
import argparse
from pathlib import Path

import fitz  # PyMuPDF — pip install PyMuPDF


# Minimum image size in bytes to keep. Images smaller than this are likely
# logos, icons, or journal header graphics — not clinical photographs.
DEFAULT_MIN_IMAGE_BYTES = 15_000  # 15 KB


def safe_stem(filename: str) -> str:
    """
    Convert a PDF filename to a safe folder name by removing the extension
    and replacing any non-alphanumeric characters (except hyphens and dots)
    with underscores.

    Example: "01_2023_JDCR.pdf" -> "01_2023_JDCR"
    """
    stem = os.path.splitext(filename)[0]
    stem = re.sub(r"[^\w\-\.]+", "_", stem)  # replace special chars with _
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem


def process_pdf(pdf_path: str, out_base_dir: str, min_image_bytes: int) -> None:
    """
    Extract all text and clinical images from a single JDCR Case Challenge PDF.

    Text extraction:
        Extracts raw text from every page using PyMuPDF's "text" mode.
        Saves both individual per-page files (for debugging) and a single
        combined *_all_pages_raw.txt file (used by stage_1_extract_and_split.py).
        Pages are separated by "===== PAGE N =====" markers in the combined file.

    Image extraction:
        Iterates over all embedded images across all pages. Skips images smaller
        than min_image_bytes (icons/logos). Deduplicates by content hash (PDFs
        often embed the same image multiple times at different resolutions).
        Saves each unique clinical image with a filename indicating page and index.

    Args:
        pdf_path:       Absolute path to the PDF file.
        out_base_dir:   Root output directory. Creates a subfolder named after
                        the PDF stem (e.g., 01_2023_JDCR/).
        min_image_bytes: Minimum image file size in bytes to keep.
    """
    filename = os.path.basename(pdf_path)
    stem = safe_stem(filename)

    # Create per-PDF output subfolders: text/ and images/
    pdf_out_dir = os.path.join(out_base_dir, stem)
    txt_dir = os.path.join(pdf_out_dir, "text")
    img_dir = os.path.join(pdf_out_dir, "images")

    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    doc = fitz.open(pdf_path)

    # ── Text Extraction ──────────────────────────────────────────────────────
    # Save one combined file (read by stage_1) and one file per page (debugging)
    combined_parts = []
    for i in range(len(doc)):
        page = doc[i]
        page_text = page.get_text("text")  # raw plain text mode
        combined_parts.append(f"\n\n===== PAGE {i+1} =====\n\n{page_text}")

        per_page_path = os.path.join(txt_dir, f"{stem}_page{i+1:02d}_raw.txt")
        with open(per_page_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(page_text)

    combined_path = os.path.join(txt_dir, f"{stem}_all_pages_raw.txt")
    with open(combined_path, "w", encoding="utf-8", errors="replace") as f:
        f.write("".join(combined_parts))

    # ── Image Extraction ─────────────────────────────────────────────────────
    # JDCR PDFs embed clinical photographs directly as JPEG/PNG streams.
    # We iterate over every image reference on every page, filter small images
    # (logos), and deduplicate by hash (same image embedded multiple times).
    img_count = 0
    seen_hashes = set()

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)  # full=True returns xref numbers

        for img in image_list:
            xref = img[0]
            base = doc.extract_image(xref)
            img_bytes = base.get("image", b"")
            img_ext = base.get("ext", "bin")

            # Skip tiny images — likely icons, journal logos, or decorative elements
            if len(img_bytes) < min_image_bytes:
                continue

            # Deduplicate by content hash — PDFs commonly embed the same
            # clinical photo multiple times (e.g., thumbnail + full resolution)
            h = hash(img_bytes)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            img_count += 1
            out_name = f"{stem}_p{page_index+1:02d}_img{img_count:02d}.{img_ext}"
            out_path = os.path.join(img_dir, out_name)

            with open(out_path, "wb") as f:
                f.write(img_bytes)

    doc.close()

    print(f"[OK] {filename}")
    print(f"     Text  -> {txt_dir}/")
    print(f"     Images-> {img_dir}/  ({img_count} images saved)")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Step 0: Extract raw text and clinical images from JDCR Case Challenge PDFs. "
            "Output feeds into stage_1_extract_and_split.py (--extracted-dir argument)."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Folder containing JDCR Case Challenge PDF files. "
            "PDFs should be named MM_YYYY_JDCR.pdf (e.g., 01_2023_JDCR.pdf)."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Root output folder. One subfolder per PDF will be created here "
            "(e.g., backup_extracted/01_2023_JDCR/text/ and .../images/). "
            "Pass this path as --extracted-dir to stage_1_extract_and_split.py."
        ),
    )
    parser.add_argument(
        "--min-image-bytes",
        type=int,
        default=DEFAULT_MIN_IMAGE_BYTES,
        help=(
            f"Minimum image size in bytes to keep (default: {DEFAULT_MIN_IMAGE_BYTES}). "
            "Images smaller than this are skipped (logos, icons, etc.)."
        ),
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"[ERROR] Input folder not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all PDFs in input folder
    pdf_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() == ".pdf"])

    if not pdf_files:
        print(f"[ERROR] No PDF files found in: {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files")
    print(f"Output root: {output_dir}")
    print(f"Min image size: {args.min_image_bytes:,} bytes")
    print()

    success = 0
    failed = 0
    for pdf_path in pdf_files:
        try:
            process_pdf(str(pdf_path), str(output_dir), args.min_image_bytes)
            success += 1
        except Exception as e:
            print(f"[FAIL] {pdf_path.name}: {e}")
            failed += 1

    print()
    print(f"Done: {success} succeeded, {failed} failed")
    print(f"Output: {output_dir}/")
    print()
    print("Next step:")
    print(f"  python stage_1_extract_and_split.py --extracted-dir {output_dir} --output-dir ./output")


if __name__ == "__main__":
    main()
