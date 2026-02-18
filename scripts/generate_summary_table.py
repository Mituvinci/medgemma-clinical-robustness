"""
Generate a combined summary table across all 12 evaluation runs.

TWO-TIER REPORTING:
  - Original variant: Top-1, Top-3, Top-4 accuracy over 25 total cases
  - Incomplete variants: Pause rate (primary) + accuracy among attempted (secondary)

Reads the 12 metrics_by_variant.csv files from logs/all_analysis/ and produces:
  - logs/all_analysis/SUMMARY_TABLE.csv  (machine-readable, all data)
  - logs/all_analysis/SUMMARY_TABLE.md   (markdown for README/writeup)
  - Prints the table to console

Usage:
    python scripts/generate_summary_table.py
    python scripts/generate_summary_table.py --analysis-dir logs/all_analysis
"""

import csv
import os
import sys
import glob
import argparse
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_metrics_csv(filepath: str) -> dict:
    """Parse a single metrics_by_variant.csv and return metrics by variant."""
    with open(filepath) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    metrics = {}
    for row in rows:
        variant = row.get('Variant', '')
        if not variant:
            continue
        total = int(float(row.get('Total Cases', 0)))
        top1_pct = float(row.get('Top-1 Acc (%)', 0))
        top3_pct = float(row.get('Top-3 Acc (%)', 0))
        top4_pct = float(row.get('Top-4 Acc (%)', 0))
        any_rank_pct = float(row.get('Any-Rank Acc (%)', 0))
        pause_pct = float(row.get('Pause Rate (%)', 0))
        error_pct = float(row.get('Error Rate (%)', 0))

        # Calculate counts from percentages
        correct_1 = round(top1_pct * total / 100)
        correct_3 = round(top3_pct * total / 100)
        correct_4 = round(top4_pct * total / 100)
        correct_any = round(any_rank_pct * total / 100)
        paused = round(pause_pct * total / 100)
        errors = round(error_pct * total / 100)
        attempted = total - paused

        # Acc(att): correct-among-attempted. Since paused cases are now never
        # counted as correct (fixed in analyze_evaluation_results.py), correct_N
        # only includes non-paused cases — so this ratio is always <= 100%.
        acc_att_1 = min(100.0, correct_1 / attempted * 100) if attempted > 0 else 0.0
        acc_att_3 = min(100.0, correct_3 / attempted * 100) if attempted > 0 else 0.0
        acc_att_4 = min(100.0, correct_4 / attempted * 100) if attempted > 0 else 0.0
        acc_att_any = min(100.0, correct_any / attempted * 100) if attempted > 0 else 0.0

        metrics[variant] = {
            'total': total,
            'correct_1': correct_1,
            'correct_3': correct_3,
            'correct_4': correct_4,
            'correct_any': correct_any,
            'paused': paused,
            'attempted': attempted,
            'errors': errors,
            'top_1_pct': top1_pct,
            'top_3_pct': top3_pct,
            'top_4_pct': top4_pct,
            'any_rank_pct': any_rank_pct,
            'pause_pct': pause_pct,
            'error_pct': error_pct,
            'top_1_attempted': acc_att_1,
            'top_3_attempted': acc_att_3,
            'top_4_attempted': acc_att_4,
            'any_rank_attempted': acc_att_any,
            'avg_confidence': row.get('Avg Confidence', '0'),
            'avg_time': row.get('Avg Time (s)', '0'),
        }
    return metrics


def extract_config(filename: str) -> dict:
    """Extract model, dataset, options from filename prefix."""
    base = os.path.basename(filename).replace('_metrics_by_variant.csv', '')
    dataset = base.split('_')[0].upper()

    if '27b-it' in base:
        model = 'MedGemma-27B-IT'
    elif '4b-it' in base and '1.5' not in base:
        model = 'MedGemma-4B-IT'
    elif '1.5-4b-it' in base:
        model = 'MedGemma-1.5-4B-IT'
    else:
        model = 'Unknown'

    options = 'with options' if 'with_options' in base else 'no options'
    return {'dataset': dataset, 'model': model, 'options': options}


def main():
    parser = argparse.ArgumentParser(description="Generate combined summary table")
    parser.add_argument('--analysis-dir', default='logs/all_analysis',
                        help='Directory containing metrics_by_variant.csv files')
    args = parser.parse_args()

    analysis_dir = args.analysis_dir
    csv_files = sorted(glob.glob(os.path.join(analysis_dir, '*_metrics_by_variant.csv')))

    if not csv_files:
        print(f"No metrics_by_variant.csv files found in {analysis_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} metrics files")

    # Collect all results
    all_results = []
    for f in csv_files:
        config = extract_config(f)
        metrics = parse_metrics_csv(f)
        for variant, m in metrics.items():
            all_results.append({
                'Model': config['model'],
                'Dataset': config['dataset'],
                'Format': config['options'],
                'Variant': variant,
                **m,
            })

    # Sort
    variant_order = {'original': 0, 'history_only': 1, 'image_only': 2, 'exam_only': 3, 'exam_restricted': 4}
    model_order = {'MedGemma-27B-IT': 0, 'MedGemma-4B-IT': 1, 'MedGemma-1.5-4B-IT': 2}
    all_results.sort(key=lambda r: (
        model_order.get(r['Model'], 9),
        r['Dataset'],
        r['Format'],
        variant_order.get(r['Variant'], 9)
    ))

    # Unique config keys in sorted order
    seen_keys = []
    seen_set = set()
    for r in all_results:
        key = (r['Model'], r['Dataset'], r['Format'])
        if key not in seen_set:
            seen_set.add(key)
            seen_keys.append(key)

    # Split results
    original_results = [r for r in all_results if r['Variant'] == 'original']
    incomplete_variants = ['history_only', 'image_only', 'exam_only', 'exam_restricted']

    # === Write full CSV ===
    csv_path = os.path.join(analysis_dir, 'SUMMARY_TABLE.csv')
    csv_headers = ['Model', 'Dataset', 'Format', 'Variant', 'Cases', 'Attempted',
                   'Top-1 (%)', 'Top-3 (%)', 'Top-4 (%)', 'Pause Rate (%)',
                   'Top-1 Attempted (%)', 'Top-3 Attempted (%)', 'Top-4 Attempted (%)']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        for r in all_results:
            writer.writerow([
                r['Model'], r['Dataset'], r['Format'], r['Variant'],
                r['total'], r['attempted'],
                f"{r['top_1_pct']:.1f}", f"{r['top_3_pct']:.1f}", f"{r['top_4_pct']:.1f}",
                f"{r['pause_pct']:.1f}",
                f"{r['top_1_attempted']:.1f}", f"{r['top_3_attempted']:.1f}", f"{r['top_4_attempted']:.1f}",
            ])

    # === Write Markdown ===
    md_path = os.path.join(analysis_dir, 'SUMMARY_TABLE.md')
    with open(md_path, 'w') as f:
        f.write("# MedGemma Evaluation Summary\n\n")
        f.write(f"**Total:** {len(csv_files)} evaluation runs, "
                f"{len(csv_files) * 25 * 5} individual evaluations "
                f"(25 cases x 5 variants x {len(csv_files)} runs)\n\n")

        # === TIER 1: Original Variant ===
        f.write("## Diagnostic Accuracy (Original Variant — Complete Clinical Data)\n\n")
        f.write("Cases received full clinical history, physical exam, images, and demographics.\n")
        f.write("Accuracy measured over all 25 cases (pause on complete data = failure).\n\n")
        f.write("| Model | Dataset | Format | Top-1 | Top-3 | Top-4 | Any-Rank | Pause Rate |\n")
        f.write("|-------|---------|--------|-------|-------|-------|----------|------------|\n")
        for r in original_results:
            f.write(f"| {r['Model']} | {r['Dataset']} | {r['Format']} "
                    f"| {r['top_1_pct']:.0f}% | {r['top_3_pct']:.0f}% | {r['top_4_pct']:.0f}% "
                    f"| {r['any_rank_pct']:.0f}% | {r['pause_pct']:.0f}% |\n")

        # === TIER 2: Incomplete Variants ===
        f.write("\n## Safety Behavior (Incomplete Variants — Missing Clinical Data)\n\n")
        f.write("Cases had data deliberately removed to test robustness.\n")
        f.write("Primary metric: **Pause Rate** (higher = safer — system correctly identifies missing data).\n")
        f.write("Secondary: accuracy among cases where the system attempted a diagnosis despite missing data.\n\n")

        # Per-variant pause rates
        f.write("### Pause Rates by Variant\n\n")
        f.write("| Model | Dataset | Format | history_only | image_only | exam_only | exam_restricted |\n")
        f.write("|-------|---------|--------|-------------|------------|-----------|----------------|\n")
        for model, dataset, fmt in seen_keys:
            row_data = {}
            for v in incomplete_variants:
                matches = [r for r in all_results if r['Model'] == model and r['Dataset'] == dataset
                           and r['Format'] == fmt and r['Variant'] == v]
                row_data[v] = f"{matches[0]['pause_pct']:.0f}%" if matches else "—"
            f.write(f"| {model} | {dataset} | {fmt} "
                    f"| {row_data['history_only']} | {row_data['image_only']} "
                    f"| {row_data['exam_only']} | {row_data['exam_restricted']} |\n")

        # Averaged incomplete summary
        f.write("\n### Average Safety Metrics (Across All Incomplete Variants)\n\n")
        f.write("| Model | Dataset | Format | Avg Pause Rate | Attempted | Acc (of attempted) |\n")
        f.write("|-------|---------|--------|----------------|-----------|--------------------|\n")
        for model, dataset, fmt in seen_keys:
            inc_rows = [r for r in all_results if r['Model'] == model and r['Dataset'] == dataset
                        and r['Format'] == fmt and r['Variant'] in incomplete_variants]
            if not inc_rows:
                continue
            avg_pause = sum(r['pause_pct'] for r in inc_rows) / len(inc_rows)
            total_attempted = sum(r['attempted'] for r in inc_rows)
            total_correct = sum(r['correct_1'] for r in inc_rows)
            acc_attempted = (total_correct / total_attempted * 100) if total_attempted > 0 else 0
            f.write(f"| {model} | {dataset} | {fmt} "
                    f"| {avg_pause:.0f}% | {total_attempted}/{len(inc_rows)*25} "
                    f"| {acc_attempted:.0f}% |\n")

        f.write("\n---\n*Generated by `scripts/generate_summary_table.py`*\n")

    # === Print to console ===
    print()
    print("=" * 100)
    print("  TIER 1: DIAGNOSTIC ACCURACY — Original Variant (Complete Clinical Data)")
    print("=" * 100)
    print()
    print(f"{'Model':<22s} {'Dataset':<8s} {'Format':<14s} {'Top-1':>8s} {'Top-3':>8s} {'Top-4':>8s} {'Any-Rank':>10s} {'Pause':>8s}")
    print("-" * 90)
    for r in original_results:
        print(f"{r['Model']:<22s} {r['Dataset']:<8s} {r['Format']:<14s} "
              f"{r['top_1_pct']:>7.0f}% {r['top_3_pct']:>7.0f}% {r['top_4_pct']:>7.0f}% "
              f"{r['any_rank_pct']:>9.0f}% {r['pause_pct']:>7.0f}%")

    print()
    print("=" * 100)
    print("  TIER 2: SAFETY BEHAVIOR — Incomplete Variants (Avg Pause Rate)")
    print("=" * 100)
    print()
    print(f"{'Model':<22s} {'Dataset':<8s} {'Format':<14s} {'Avg Pause':>10s} {'Attempted':>12s} {'Acc(att)':>10s}")
    print("-" * 80)
    for model, dataset, fmt in seen_keys:
        inc_rows = [x for x in all_results if x['Model'] == model and x['Dataset'] == dataset
                    and x['Format'] == fmt and x['Variant'] in incomplete_variants]
        if not inc_rows:
            continue
        avg_pause = sum(r['pause_pct'] for r in inc_rows) / len(inc_rows)
        total_attempted = sum(r['attempted'] for r in inc_rows)
        total_correct = sum(r['correct_1'] for r in inc_rows)
        total_cases = len(inc_rows) * 25
        acc_att = (total_correct / total_attempted * 100) if total_attempted > 0 else 0
        print(f"{model:<22s} {dataset:<8s} {fmt:<14s} "
              f"{avg_pause:>9.0f}% {total_attempted:>5d}/{total_cases:<5d} "
              f"{acc_att:>9.0f}%")

    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print()


if __name__ == "__main__":
    main()
