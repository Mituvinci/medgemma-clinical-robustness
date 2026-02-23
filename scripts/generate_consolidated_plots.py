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

Output (per dataset, 3 plots each saved as JPEG + PDF):
    {DATASET}_consolidated_accuracy.{jpg,pdf}
    {DATASET}_consolidated_safety_history.{jpg,pdf}
    {DATASET}_consolidated_pause_heatmap.{jpg,pdf}

Style: Publication-quality. Minimum font size 14. 300 dpi JPEG + vector PDF.

Acc(att) metric:
    Standard Top-1 accuracy penalizes pausing (paused = wrong). Acc(att) removes
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Publication style ─────────────────────────────────────────────────────────
# Uniform large fonts across all plots for print/PDF readability.
plt.rcParams.update({
    'font.size':          15,
    'axes.titlesize':     17,
    'axes.labelsize':     15,
    'xtick.labelsize':    14,
    'ytick.labelsize':    14,
    'legend.fontsize':    13,
    'axes.titleweight':   'bold',
    'figure.dpi':         150,
    'savefig.dpi':        300,
})
sns.set_theme(style="whitegrid", font_scale=1.25)

# ── Label helpers ─────────────────────────────────────────────────────────────

# Correct MedGemma variant names: lowercase b, lowercase -it suffix
MODEL_SHORT = {
    'medgemma-27b-it':    '27b-it',
    'medgemma-4b-it':     '4b-it',
    'medgemma-1.5-4b-it': '1.5-4b-it',
}

INCOMPLETE_VARIANTS = ['history_only', 'image_only', 'exam_only', 'exam_restricted']
VARIANT_LABELS = {
    'history_only':    'History\nOnly',
    'image_only':      'Image\nOnly',
    'exam_only':       'Exam\nOnly',
    'exam_restricted': 'Exam\nRestricted',
}


def _save_fig(base_path: str) -> None:
    """
    Save the current matplotlib figure as both JPEG (300 dpi) and PDF (vector).

    Args:
        base_path: Full path without extension (e.g. 'logs/all_analysis/JDCR_consolidated_accuracy')
    """
    jpg_path = base_path + '.jpg'
    pdf_path = base_path + '.pdf'
    plt.savefig(jpg_path, dpi=300, bbox_inches='tight', format='jpeg')
    plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"  Saved: {jpg_path}")
    print(f"  Saved: {pdf_path}")


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
        short     : abbreviated model label (e.g., '27b-it')
        fmt       : 'w/ opts' or 'no opts'
        label     : two-line plot label combining short name + format
                    (e.g., '27b-it\\nw/ opts') for use as x-axis tick label
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


# ── Plot 1: Accuracy Overview (original variant) ──────────────────────────────

def plot_accuracy_overview(entries: list, dataset: str, out_dir: str):
    """
    Grouped bar chart -- original variant (publication quality).
    Bars: Top-1, Top-3, Top-4, Any-Rank.  Line overlay: Pause Rate.
    """
    labels  = [e['label']    for e in entries]
    top1    = [e['top1']     for e in entries]
    top3    = [e['top3']     for e in entries]
    top4    = [e['top4']     for e in entries]
    anyrank = [e['any_rank'] for e in entries]
    pause   = [e['pause']    for e in entries]

    n = len(labels)
    x = np.arange(n)
    bar_w = 0.19

    fig, ax1 = plt.subplots(figsize=(12, 6))

    offsets    = [-1.5, -0.5, 0.5, 1.5]
    colors     = ['#4C72B0', '#55A868', '#DD8452', '#8172B3']
    bar_labels = ['Top-1', 'Top-3', 'Top-4', 'Any-Rank']

    bars = []
    for off, col, lbl, vals in zip(offsets, colors, bar_labels,
                                   [top1, top3, top4, anyrank]):
        b = ax1.bar(x + off * bar_w, vals, bar_w, label=lbl, color=col, alpha=0.88)
        bars.append(b)

    ax2 = ax1.twinx()
    ax2.plot(x, pause, 'D--', color='#C44E52', linewidth=2.5, markersize=9,
             label='Pause Rate', zorder=5)
    ax2.set_ylabel('Pause Rate (%)', color='#C44E52', fontsize=15)
    ax2.tick_params(axis='y', labelcolor='#C44E52', labelsize=14)
    ax2.set_ylim(0, 110)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=14)
    ax1.set_ylim(0, 110)
    ax1.set_ylabel('Accuracy (%)', fontsize=15)
    ax1.set_xlabel('MedGemma Model Variants', fontsize=15, labelpad=8)
    ax1.set_title(f'{dataset} -- Diagnostic Accuracy (Complete Clinical Data)',
                  fontsize=17, fontweight='bold', pad=12)

    handles1 = [mpatches.Patch(color=c, alpha=0.88, label=l)
                for c, l in zip(colors, bar_labels)]
    line_handle = plt.Line2D([0], [0], color='#C44E52', linewidth=2.5,
                             linestyle='--', marker='D', markersize=9, label='Pause Rate')
    ax1.legend(handles=handles1 + [line_handle], loc='upper left', fontsize=13,
               framealpha=0.92, edgecolor='#cccccc')

    for b_group in bars:
        for rect in b_group:
            h = rect.get_height()
            if h > 0:
                ax1.text(rect.get_x() + rect.get_width() / 2, h + 1,
                         f'{h:.0f}%', ha='center', va='bottom', fontsize=10,
                         fontweight='bold')

    plt.tight_layout()
    base = os.path.join(out_dir, f'{dataset.upper()}_consolidated_accuracy')
    _save_fig(base)


# ── Plot 2: Safety — history_only ─────────────────────────────────────────────

def plot_safety_history(entries: list, dataset: str, out_dir: str):
    """
    Bar chart -- history_only variant (publication quality).
    Primary: Pause Rate (key safety metric, higher = safer).
    Secondary: Acc(att) -- accuracy among cases that weren't paused.
    """
    labels  = [e['label']   for e in entries]
    pause   = [e['pause']   for e in entries]
    acc_att = [e['acc_att'] for e in entries]

    n = len(labels)
    x = np.arange(n)
    bar_w = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    b1 = ax.bar(x - bar_w / 2, pause,   bar_w, label='Pause Rate',
                color='#2196F3', alpha=0.88)
    b2 = ax.bar(x + bar_w / 2, acc_att, bar_w, label='Acc(att)',
                color='#FF9800', alpha=0.88)

    ax.axhline(100, color='#2196F3', linewidth=1.2, linestyle=':', alpha=0.4)

    for bars, vals in [(b1, pause), (b2, acc_att)]:
        for rect, v in zip(bars, vals):
            if v > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, v + 1,
                        f'{v:.0f}%', ha='center', va='bottom',
                        fontsize=12, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Percentage (%)', fontsize=15)
    ax.set_xlabel('MedGemma Model Variants', fontsize=15, labelpad=8)
    ax.set_title(f'{dataset} -- Safety Behavior (History-Only Variant)',
                 fontsize=17, fontweight='bold', pad=12)
    ax.legend(loc='lower right', fontsize=13, framealpha=0.92, edgecolor='#cccccc')

    plt.tight_layout()
    base = os.path.join(out_dir, f'{dataset.upper()}_consolidated_safety_history')
    _save_fig(base)


# ── Plot 3: Pause Rate Heatmap (all incomplete variants) ──────────────────────

def plot_pause_heatmap(entries_by_variant: dict, dataset: str, out_dir: str):
    """
    Publication-quality heatmap: rows = model+format, columns = 4 incomplete variants.
    Color = Pause Rate (higher = safer, greener).

    Design choices for publication quality:
    - seaborn heatmap with clean cell borders (no grid intersection confusion)
    - Large annotation font (16pt bold) for readability in print
    - Fixed 0-100 color scale for cross-figure consistency
    - Concise title (no subtitle explanation)
    - Square cells with consistent aspect ratio
    """
    model_labels = [e['label'].replace('\n', ' ')
                    for e in entries_by_variant[INCOMPLETE_VARIANTS[0]]]
    col_labels   = [VARIANT_LABELS[v].replace('\n', ' ') for v in INCOMPLETE_VARIANTS]

    data = np.array([
        [e['pause'] for e in entries_by_variant[v]]
        for v in INCOMPLETE_VARIANTS
    ]).T  # shape: (n_models, n_variants)

    n_rows = len(model_labels)
    n_cols = len(col_labels)
    fig, ax = plt.subplots(figsize=(n_cols * 2.4 + 3, n_rows * 0.9 + 2))

    # Use seaborn heatmap for clean cell rendering
    sns.heatmap(
        data,
        ax=ax,
        annot=False,           # manual annotations below for finer control
        cmap='YlGn',
        vmin=0,
        vmax=100,
        linewidths=2.0,        # clean white borders between cells
        linecolor='white',
        square=True,           # uniform cell shape
        cbar_kws={
            'label': 'Pause Rate (%)',
            'shrink': 0.8,
            'pad': 0.02,
        },
    )

    # Style the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Pause Rate (%)', fontsize=15, labelpad=10)

    # Manual cell annotations -- large, bold, high-contrast
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            text_color = '#ffffff' if v >= 65 else '#1a1a1a'
            ax.text(j + 0.5, i + 0.5, f'{v:.0f}%',
                    ha='center', va='center',
                    fontsize=16, fontweight='bold', color=text_color)

    # Axis labels
    ax.set_xticklabels(col_labels, fontsize=15, rotation=0, ha='center')
    ax.set_yticklabels(model_labels, fontsize=15, rotation=0, va='center')
    ax.set_xlabel('Context Variant (Incomplete Data)', fontsize=15, labelpad=10)
    ax.set_ylabel('MedGemma Model Variants', fontsize=15, labelpad=10)

    # Concise title -- no multi-line subtitle
    ax.set_title(f'{dataset} -- Pause Rate on Incomplete Variants',
                 fontsize=17, fontweight='bold', pad=14)

    # Remove top/right spines for cleaner look
    ax.tick_params(top=False, right=False, left=False, bottom=False)

    plt.tight_layout()
    base = os.path.join(out_dir, f'{dataset.upper()}_consolidated_pause_heatmap')
    _save_fig(base)


# ── Plot 4: Safety Governor — Quadrant Scatter Map ────────────────────────────

def plot_safety_governor_scatter(file_entries: list, dataset: str, out_dir: str):
    """
    Quadrant Safety Map (publication quality): scatter plot showing the inverse
    relationship between diagnostic accuracy on complete data (X-axis) and
    average pause rate on incomplete data (Y-axis).
    """
    from matplotlib.lines import Line2D

    SHORT_TAG = {
        ('medgemma-27b-it',    'no opts'):  '27b no opts',
        ('medgemma-27b-it',    'w/ opts'):  '27b opts',
        ('medgemma-4b-it',     'no opts'):  '4b no opts',
        ('medgemma-4b-it',     'w/ opts'):  '4b opts',
        ('medgemma-1.5-4b-it', 'no opts'):  '1.5b no opts',
        ('medgemma-1.5-4b-it', 'w/ opts'):  '1.5b opts',
    }

    # Manually tuned label anchor positions to avoid overlaps.
    LABEL_XY = {
        ('medgemma-27b-it',    'no opts'):  (6,   90,  'left'),
        ('medgemma-4b-it',     'no opts'):  (24,  97,  'left'),
        ('medgemma-1.5-4b-it', 'no opts'):  (40,  83,  'right'),
        ('medgemma-27b-it',    'w/ opts'):  (44,  95,  'right'),
        ('medgemma-4b-it',     'w/ opts'):  (68,  80,  'left'),
        ('medgemma-1.5-4b-it', 'w/ opts'):  (64,  86,  'right'),
    }

    acc_complete     = []
    pause_incomplete = []
    model_keys       = []
    fmt_flags        = []

    for e in file_entries:
        model_keys.append(e['model_key'])
        fmt_flags.append(e['fmt'])

        orig = e['data'].get('original', {})
        acc_complete.append(float(orig.get('Top-1 Acc (%)', 0)))

        inc_pauses = []
        for v in INCOMPLETE_VARIANTS:
            row = e['data'].get(v, {})
            if row:
                inc_pauses.append(float(row.get('Pause Rate (%)', 0)))
        pause_incomplete.append(sum(inc_pauses) / len(inc_pauses) if inc_pauses else 0.0)

    fig, ax = plt.subplots(figsize=(9, 8))

    # ── Threshold guide lines ──────────────────────────────────────────────────
    ax.axhline(85, color='#999999', linewidth=1.2, linestyle=':', alpha=0.7, zorder=2)
    ax.text(2, 86.2, 'Safety threshold (85%)', fontsize=12, color='#666666', alpha=0.85)

    ax.axvline(50, color='#999999', linewidth=1.2, linestyle=':', alpha=0.7, zorder=2)
    ax.text(51.5, 4, 'Utility\nthreshold\n(50%)', fontsize=11, color='#666666', alpha=0.85)

    # ── Diagonal: x + y = 100 ─────────────────────────────────────────────────
    x_line = np.linspace(0, 100, 300)
    y_line = 100 - x_line
    ax.plot(x_line, y_line, '--', color='#777777', linewidth=1.5, alpha=0.5, zorder=2)
    ax.text(63, 33, 'x + y = 100',
            fontsize=11, color='#666666', alpha=0.7, rotation=-42, ha='center')

    # ── Ideal zone shading ─────────────────────────────────────────────────────
    ax.fill_between(x_line, np.maximum(y_line, 85), 100,
                    where=(x_line >= 50), alpha=0.10, color='#4CAF50', zorder=1)
    ax.text(66, 100.5,
            'Ideal Zone',
            fontsize=13, color='#1B5E20', fontweight='bold',
            alpha=0.90, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.7, edgecolor='none'))

    # ── Problematic zone label ─────────────────────────────────────────────────
    ax.text(18, 18, 'Problematic Zone', fontsize=12, color='#B71C1C',
            fontstyle='italic', alpha=0.55, ha='center')

    # ── Colors per model family ────────────────────────────────────────────────
    model_colors = {
        'medgemma-27b-it':    '#1565C0',
        'medgemma-4b-it':     '#2E7D32',
        'medgemma-1.5-4b-it': '#B71C1C',
    }

    for acc, pause, mkey, fmt in zip(acc_complete, pause_incomplete,
                                     model_keys, fmt_flags):
        color  = model_colors.get(mkey, '#555555')
        marker = 's' if fmt == 'w/ opts' else 'o'
        ax.scatter(acc, pause, color=color, marker=marker,
                   s=280, zorder=5, edgecolors='white', linewidths=1.2)

        tag = SHORT_TAG.get((mkey, fmt), mkey)
        tx, ty, ha = LABEL_XY.get((mkey, fmt), (acc + 4, pause + 2, 'left'))

        ax.annotate(
            tag, xy=(acc, pause), xycoords='data',
            xytext=(tx, ty), textcoords='data',
            fontsize=12, color=color, fontweight='bold', ha=ha, va='center',
            arrowprops=dict(arrowstyle='-', color=color, lw=0.8, alpha=0.4),
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 104)
    ax.set_xlabel('Top-1 Accuracy on Complete Cases (%)', fontsize=15, labelpad=8)
    ax.set_ylabel('Avg Pause Rate on Incomplete Cases (%)', fontsize=15, labelpad=8)
    ax.set_title(f'{dataset} -- Safety Governor',
                 fontsize=17, fontweight='bold', pad=12)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555',
               markersize=12, label='No options'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#555555',
               markersize=12, label='With MCQ options'),
    ]
    ax.legend(handles=legend_handles, loc='lower left', fontsize=13,
              framealpha=0.92, edgecolor='#cccccc')

    plt.tight_layout()
    base = os.path.join(out_dir, f'{dataset.upper()}_safety_governor_scatter')
    _save_fig(base)


# ── Main ──────────────────────────────────────────────────────────────────────

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
    parser = argparse.ArgumentParser(description="Generate consolidated comparison plots (publication style)")
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

        # NEJM "with options" run was incomplete (orchestrator exhausted for 1.5-4b-it).
        # Only show "no options" results for NEJM plots to avoid misleading numbers.
        if dataset == 'NEJM':
            file_entries = [e for e in file_entries if e['fmt'] == 'no opts']

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

        # ── Plot 4: Safety governor scatter (JDCR only — official evaluation) ─
        if dataset == 'JDCR':
            plot_safety_governor_scatter(file_entries, dataset, analysis_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
