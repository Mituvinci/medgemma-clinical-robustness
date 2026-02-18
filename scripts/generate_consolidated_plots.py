"""
Generate 6 Consolidated Comparison Plots Across All Models and Formats.

This script reads the 12 per-run metrics_by_variant.csv files produced by
analyze_evaluation_results.py and creates 6 summary plots that compare all
models side-by-side. These are the primary visualizations for the competition
writeup and demo video.

Input (12 CSV files in --analysis-dir):
    {DATASET}_{model_slug}_{format}_metrics_by_variant.csv
    Where DATASET ∈ {NEJM, JDCR}, model ∈ {27b-it, 4b-it, 1.5-4b-it},
    format ∈ {without_options, with_options}

Output (6 PNG files in --analysis-dir):
    Per dataset (NEJM and JDCR), 3 plots each:

    1. {DATASET}_consolidated_accuracy.png
       Grouped bar chart — original variant (complete cases) only.
       Bars: Top-1 / Top-3 / Top-4 / Any-Rank accuracy.
       Line overlay (secondary axis): Pause Rate on complete cases.
       Purpose: Shows raw diagnostic accuracy comparison across all 6 model+format combos.

    2. {DATASET}_consolidated_safety_history.png
       Side-by-side bar chart — history_only variant (text only, no image/exam).
       Bars: Pause Rate (safety metric, higher = safer) + Acc(att) (quality when attempted).
       Purpose: Isolates safety behavior when only patient history is available — the
       most clinically common "incomplete data" scenario. High pause rate here is desired.

    3. {DATASET}_consolidated_pause_heatmap.png
       Heatmap — all 4 incomplete variants (history_only, image_only, exam_only, exam_restricted).
       Rows: context variants. Columns: 6 model+format combinations.
       Color: Pause rate (0-100%). Darker = more pausing = safer behavior.
       Purpose: Shows at a glance whether the system consistently pauses across ALL
       types of missing data, not just one variant.

Acc(att) metric:
    Standard Top-1 accuracy penalizes pausing (paused case = wrong). Acc(att) removes
    paused cases from the denominator, showing diagnostic quality when the model does
    attempt a diagnosis. A high pause rate + high Acc(att) = ideal safety + quality.

Usage:
    python scripts/generate_consolidated_plots.py
    python scripts/generate_consolidated_plots.py --analysis-dir logs/all_analysis
"""

import csv
import glob
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

sns.set_theme(style="whitegrid")

# ── Label helpers ────────────────────────────────────────────────────────────

MODEL_SHORT = {
    'medgemma-27b-it':    '27B',
    'medgemma-4b-it':     '4B',
    'medgemma-1.5-4b-it': '1.5-4B',
}

INCOMPLETE_VARIANTS = ['history_only', 'image_only', 'exam_only', 'exam_restricted']
VARIANT_LABELS = {
    'history_only':    'History\nOnly',
    'image_only':      'Image\nOnly',
    'exam_only':       'Exam\nOnly',
    'exam_restricted': 'Exam\nRestricted',
}


def _extract_model_key(filename: str) -> str:
    """
    Infer MedGemma model variant from a metrics CSV filename.

    Filenames follow the pattern: {DATASET}_{model_slug}_{format}_metrics_by_variant.csv
    Order of checks matters: '1.5-4b-it' must be checked before '4b-it' since
    '1.5-4b-it' contains '4b-it' as a substring.

    Returns one of: 'medgemma-27b-it', 'medgemma-4b-it', 'medgemma-1.5-4b-it', 'unknown'
    """
    base = os.path.basename(filename).replace('_metrics_by_variant.csv', '')
    if '27b-it' in base:
        return 'medgemma-27b-it'
    if '1.5-4b-it' in base:
        return 'medgemma-1.5-4b-it'
    if '4b-it' in base:
        return 'medgemma-4b-it'
    return 'unknown'


def _extract_info(filename: str) -> dict:
    """
    Parse dataset, model, and format metadata from a metrics CSV filename.

    Returns a dict with:
        dataset   : 'NEJM' or 'JDCR' (uppercased from filename prefix)
        model_key : full model key (e.g., 'medgemma-27b-it')
        short     : abbreviated model label (e.g., '27B')
        fmt       : 'w/ opts' or 'no opts'
        label     : two-line plot label combining short name + format
                    (e.g., '27B\\nw/ opts') for use as x-axis tick label
    """
    base = os.path.basename(filename).replace('_metrics_by_variant.csv', '')
    dataset = base.split('_')[0].upper()
    model_key = _extract_model_key(filename)
    fmt = 'w/ opts' if 'with_options' in base else 'no opts'
    short = MODEL_SHORT.get(model_key, model_key)
    label = f"{short}\n{fmt}"
    return {'dataset': dataset, 'model_key': model_key, 'short': short, 'fmt': fmt, 'label': label}


def _parse_csv(filepath: str) -> dict:
    """
    Load a metrics_by_variant.csv file into a dict keyed by variant name.

    Each row corresponds to one context variant (original, history_only, etc.)
    and contains accuracy and pause rate metrics for that combination of
    model, dataset, and format.

    Returns:
        dict mapping variant name (str) to the full row dict from csv.DictReader.
    """
    rows = {}
    with open(filepath) as f:
        for row in csv.DictReader(f):
            v = row.get('Variant', '').strip()
            if v:
                rows[v] = row
    return rows


def _acc_att(top1_pct: float, pause_pct: float, total: int = 25) -> float:
    """
    Compute accuracy-among-attempted: Top-1 accuracy only for cases NOT paused.

    Standard Top-1 accuracy penalizes the model for pausing (paused = wrong).
    Acc(att) isolates pure diagnostic quality by excluding paused cases from the
    denominator. This metric answers: "When the agent DID attempt a diagnosis,
    how often was it correct?"

    Formula: correct / (total - paused)
    Where: correct = round(top1_pct * total / 100)
           paused  = round(pause_pct * total / 100)

    Args:
        top1_pct:  Top-1 accuracy as percentage (0-100), over ALL cases
        pause_pct: Pause rate as percentage (0-100)
        total:     Total number of cases evaluated (default 25)

    Returns:
        Acc(att) as percentage (0-100), or 0.0 if all cases were paused.
    """
    correct = round(top1_pct * total / 100)
    paused  = round(pause_pct * total / 100)
    attempted = total - paused
    if attempted <= 0:
        return 0.0
    return min(100.0, correct / attempted * 100)


# ── Plot 1: Accuracy Overview (original variant) ─────────────────────────────

def plot_accuracy_overview(entries: list, dataset: str, out_dir: str):
    """
    Grouped bar chart — original variant.
    Bars: Top-1, Top-3, Top-4, Any-Rank.
    Line overlay: Pause Rate.
    """
    labels  = [e['label']      for e in entries]
    top1    = [e['top1']       for e in entries]
    top3    = [e['top3']       for e in entries]
    top4    = [e['top4']       for e in entries]
    anyrank = [e['any_rank']   for e in entries]
    pause   = [e['pause']      for e in entries]

    n = len(labels)
    x = np.arange(n)
    bar_w = 0.18

    fig, ax1 = plt.subplots(figsize=(14, 6))

    offsets = [-1.5, -0.5, 0.5, 1.5]
    colors  = ['#4C72B0', '#55A868', '#DD8452', '#8172B3']
    bar_labels = ['Top-1', 'Top-3', 'Top-4', 'Any-Rank']

    bars = []
    for i, (off, col, lbl, vals) in enumerate(zip(offsets, colors, bar_labels,
                                                    [top1, top3, top4, anyrank])):
        b = ax1.bar(x + off * bar_w, vals, bar_w, label=lbl, color=col, alpha=0.85)
        bars.append(b)

    # Pause rate as dashed line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, pause, 'D--', color='#C44E52', linewidth=2, markersize=7,
             label='Pause Rate', zorder=5)
    ax2.set_ylabel('Pause Rate (%)', color='#C44E52', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#C44E52')
    ax2.set_ylim(0, 110)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_xlabel('Model  ·  Format', fontsize=11)
    ax1.set_title(f'{dataset} — Diagnostic Accuracy (Original Variant, Complete Clinical Data)',
                  fontsize=13, fontweight='bold', pad=14)

    # Combined legend
    handles1 = [mpatches.Patch(color=c, alpha=0.85, label=l)
                for c, l in zip(colors, bar_labels)]
    line_handle = plt.Line2D([0], [0], color='#C44E52', linewidth=2,
                             linestyle='--', marker='D', markersize=7, label='Pause Rate')
    ax1.legend(handles=handles1 + [line_handle], loc='upper left', fontsize=9, framealpha=0.9)

    # Value labels on bars
    for b_group in bars:
        for rect in b_group:
            h = rect.get_height()
            if h > 0:
                ax1.text(rect.get_x() + rect.get_width() / 2, h + 1,
                         f'{h:.0f}%', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{dataset.upper()}_consolidated_accuracy.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ── Plot 2: Safety — history_only ────────────────────────────────────────────

def plot_safety_history(entries: list, dataset: str, out_dir: str):
    """
    Bar chart — history_only variant.
    Primary: Pause Rate (key safety metric, higher = safer).
    Secondary: Acc(att) — accuracy among cases that weren't paused.
    """
    labels   = [e['label']   for e in entries]
    pause    = [e['pause']   for e in entries]
    acc_att  = [e['acc_att'] for e in entries]

    n = len(labels)
    x = np.arange(n)
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))

    b1 = ax.bar(x - bar_w / 2, pause,   bar_w, label='Pause Rate (safety ↑)',
                color='#2196F3', alpha=0.85)
    b2 = ax.bar(x + bar_w / 2, acc_att, bar_w, label='Acc among attempted',
                color='#FF9800', alpha=0.85)

    # Ideal pause rate line
    ax.axhline(100, color='#2196F3', linewidth=1.2, linestyle=':', alpha=0.5,
               label='100% pause (ideal)')

    for bars, vals in [(b1, pause), (b2, acc_att)]:
        for rect, v in zip(bars, vals):
            if v > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, v + 1,
                        f'{v:.0f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Rate / Accuracy (%)', fontsize=12)
    ax.set_xlabel('Model  ·  Format', fontsize=11)
    ax.set_title(f'{dataset} — Safety Behavior (history_only Variant)\n'
                 'Higher Pause Rate = safer (system correctly detects missing data)',
                 fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{dataset.upper()}_consolidated_safety_history.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ── Plot 3: Pause Rate Heatmap (all incomplete variants) ─────────────────────

def plot_pause_heatmap(entries_by_variant: dict, dataset: str, out_dir: str):
    """
    Heatmap: rows = model+format, columns = 4 incomplete variants.
    Color = Pause Rate (higher = safer, greener).
    """
    model_labels = [e['label'].replace('\n', ' ')
                    for e in entries_by_variant[INCOMPLETE_VARIANTS[0]]]
    col_labels   = [VARIANT_LABELS[v].replace('\n', ' ') for v in INCOMPLETE_VARIANTS]

    data = np.array([
        [e['pause'] for e in entries_by_variant[v]]
        for v in INCOMPLETE_VARIANTS
    ]).T  # shape: (n_models, n_variants)

    fig, ax = plt.subplots(figsize=(10, max(5, len(model_labels) * 0.7 + 2)))

    im = ax.imshow(data, cmap='YlGn', aspect='auto', vmin=0, vmax=100)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Pause Rate (%)', fontsize=11)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=10)

    # Annotate cells
    for i in range(len(model_labels)):
        for j in range(len(col_labels)):
            v = data[i, j]
            text_color = 'black' if v < 70 else 'white'
            ax.text(j, i, f'{v:.0f}%', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=text_color)

    ax.set_title(f'{dataset} — Pause Rate Across All Incomplete Variants\n'
                 '(Higher = safer: system detects missing data and pauses)',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Missing-Data Variant', fontsize=11)
    ax.set_ylabel('Model  ·  Format', fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{dataset.upper()}_consolidated_pause_heatmap.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

# Canonical model+format ordering for consistent X-axis
_SORT_KEY = {
    ('medgemma-27b-it',    'no opts'):  0,
    ('medgemma-27b-it',    'w/ opts'):  1,
    ('medgemma-4b-it',     'no opts'):  2,
    ('medgemma-4b-it',     'w/ opts'):  3,
    ('medgemma-1.5-4b-it', 'no opts'):  4,
    ('medgemma-1.5-4b-it', 'w/ opts'):  5,
}


def main():
    parser = argparse.ArgumentParser(description="Generate consolidated comparison plots")
    parser.add_argument('--analysis-dir', default='logs/all_analysis')
    args = parser.parse_args()

    analysis_dir = args.analysis_dir
    csv_files = sorted(glob.glob(os.path.join(analysis_dir, '*_metrics_by_variant.csv')))

    if not csv_files:
        print(f"No metrics CSV files found in {analysis_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} metrics files")

    # Group by dataset
    by_dataset: dict[str, list] = {}
    for f in csv_files:
        info  = _extract_info(f)
        data  = _parse_csv(f)
        ds    = info['dataset']
        entry = {**info, 'data': data}
        by_dataset.setdefault(ds, []).append(entry)

    for dataset, file_entries in by_dataset.items():
        print(f"\n── {dataset} ──────────────────────────────")

        # Sort by canonical model+format order
        file_entries.sort(key=lambda e: _SORT_KEY.get((e['model_key'], e['fmt']), 99))

        # ── Plot 1: Original variant accuracy ───────────────────────────────
        acc_entries = []
        for e in file_entries:
            orig = e['data'].get('original', {})
            if not orig:
                continue
            acc_entries.append({
                'label':    e['label'],
                'top1':     float(orig.get('Top-1 Acc (%)', 0)),
                'top3':     float(orig.get('Top-3 Acc (%)', 0)),
                'top4':     float(orig.get('Top-4 Acc (%)', 0)),
                'any_rank': float(orig.get('Any-Rank Acc (%)', 0)),
                'pause':    float(orig.get('Pause Rate (%)', 0)),
            })
        if acc_entries:
            plot_accuracy_overview(acc_entries, dataset, analysis_dir)

        # ── Plot 2: history_only safety ─────────────────────────────────────
        hist_entries = []
        for e in file_entries:
            hist = e['data'].get('history_only', {})
            if not hist:
                continue
            top1_pct  = float(hist.get('Top-1 Acc (%)', 0))
            pause_pct = float(hist.get('Pause Rate (%)', 0))
            hist_entries.append({
                'label':   e['label'],
                'pause':   pause_pct,
                'acc_att': _acc_att(top1_pct, pause_pct),
            })
        if hist_entries:
            plot_safety_history(hist_entries, dataset, analysis_dir)

        # ── Plot 3: Pause heatmap — all incomplete variants ─────────────────
        entries_by_variant = {}
        all_ok = True
        for v in INCOMPLETE_VARIANTS:
            v_entries = []
            for e in file_entries:
                row = e['data'].get(v, {})
                if not row:
                    all_ok = False
                    break
                v_entries.append({
                    'label': e['label'],
                    'pause': float(row.get('Pause Rate (%)', 0)),
                })
            if not all_ok:
                break
            entries_by_variant[v] = v_entries

        if all_ok and entries_by_variant:
            plot_pause_heatmap(entries_by_variant, dataset, analysis_dir)
        else:
            print(f"  Skipping heatmap for {dataset} — some variants missing")

    print("\nDone.")


if __name__ == "__main__":
    main()
