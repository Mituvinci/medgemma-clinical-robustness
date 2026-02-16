#!/bin/bash
#
# Run Analysis on ALL Evaluation Results
#
# Generates for each evaluation run (all files prefixed with model_dataset_options):
#   - {prefix}_side_by_side_results.csv (Case_ID | Ground_Truth | 5 variants + full responses)
#   - {prefix}_metrics_by_variant.csv (accuracy, pause rate, confidence per variant)
#   - {prefix}_detailed_results.csv (per-case per-variant breakdown)
#   - {prefix}_accuracy_by_variant.png
#   - {prefix}_pause_rate_by_variant.png
#   - {prefix}_confidence_vs_accuracy.png
#   - {prefix}_execution_time_by_variant.png
#   - {prefix}_evaluation_report.html
#
# Covers:
#   NEJM:  3 models x 2 data paths (with/without options) = 6 runs
#   JDCR:  3 models x 2 data paths (with/without options) = 6 runs
#   Total: 12 analysis runs
#
# Usage:
#   bash bin/run_all_analysis.sh
#

set -e

PROJECT_DIR="/users/ha00014/Halimas_projects/MedGemma"
cd "$PROJECT_DIR"

SCRIPT="scripts/analyze_evaluation_results.py"
NEJM_GT="NEJIM/NEJM_Groundtruth.csv"
JDCR_GT="JAADCR/JAADCR_Groundtruth.csv"

echo "========================================================================"
echo "  MedGemma - Run All Evaluation Analysis"
echo "  12 analysis runs (6 NEJM + 6 JDCR)"
echo "========================================================================"
echo ""
echo "Started: $(date)"
echo ""

# Activate conda if not already active
if [ -z "$CONDA_PREFIX" ] || [[ "$CONDA_PREFIX" != *"pytorch"* ]]; then
    eval "$(conda shell.bash hook)"
    conda activate pytorch
    echo "Activated conda: pytorch"
else
    echo "Conda already active: $CONDA_PREFIX"
fi

echo ""
echo "========================================================================"
echo "  NEJM EVALUATIONS (6 runs)"
echo "========================================================================"

# --- NEJM: MedGemma-1.5-4B-IT ---
echo ""
echo "--- [1/12] NEJM: MedGemma-1.5-4B-IT WITHOUT options ---"
python "$SCRIPT" \
    --results logs/nejm_evaluated/evaluation_medgemma-1.5-4b-it_without_options/nejim_evaluation_20260212_000001.json \
    --groundtruth "$NEJM_GT" \
    --output-dir logs/nejm_evaluated/analysis_1.5-4b-it_without_options \
    --prefix nejm_medgemma-1.5-4b-it_without_options

echo ""
echo "--- [2/12] NEJM: MedGemma-1.5-4B-IT WITH options ---"
python "$SCRIPT" \
    --results logs/nejm_evaluated/evaluation_medgemma-1.5-4b-it_with_options/nejim_evaluation_20260211_235104.json \
    --groundtruth "$NEJM_GT" \
    --output-dir logs/nejm_evaluated/analysis_1.5-4b-it_with_options \
    --prefix nejm_medgemma-1.5-4b-it_with_options

# --- NEJM: MedGemma-27B-IT ---
echo ""
echo "--- [3/12] NEJM: MedGemma-27B-IT WITHOUT options ---"
python "$SCRIPT" \
    --results logs/nejm_evaluated/evaluation_medgemma-27b-it-vertex_without_options/nejim_evaluation_20260212_230548.json \
    --groundtruth "$NEJM_GT" \
    --output-dir logs/nejm_evaluated/analysis_27b-it_without_options \
    --prefix nejm_medgemma-27b-it_without_options

echo ""
echo "--- [4/12] NEJM: MedGemma-27B-IT WITH options ---"
python "$SCRIPT" \
    --results logs/nejm_evaluated/evaluation_medgemma-27b-it-vertex_with_options/nejim_evaluation_20260213_010034.json \
    --groundtruth "$NEJM_GT" \
    --output-dir logs/nejm_evaluated/analysis_27b-it_with_options \
    --prefix nejm_medgemma-27b-it_with_options

# --- NEJM: MedGemma-4B-IT ---
echo ""
echo "--- [5/12] NEJM: MedGemma-4B-IT WITHOUT options ---"
python "$SCRIPT" \
    --results logs/nejm_evaluated/evaluation_medgemma-4b-it-vertex_without_options/nejim_evaluation_20260214_230330.json \
    --groundtruth "$NEJM_GT" \
    --output-dir logs/nejm_evaluated/analysis_4b-it_without_options \
    --prefix nejm_medgemma-4b-it_without_options

echo ""
echo "--- [6/12] NEJM: MedGemma-4B-IT WITH options ---"
python "$SCRIPT" \
    --results logs/nejm_evaluated/evaluation_medgemma-4b-it-vertex_with_options/nejim_evaluation_20260215_005151.json \
    --groundtruth "$NEJM_GT" \
    --output-dir logs/nejm_evaluated/analysis_4b-it_with_options \
    --prefix nejm_medgemma-4b-it_with_options

echo ""
echo "========================================================================"
echo "  JDCR EVALUATIONS (6 runs)"
echo "========================================================================"

# --- JDCR: MedGemma-1.5-4B-IT ---
echo ""
echo "--- [7/12] JDCR: MedGemma-1.5-4B-IT WITHOUT options ---"
python "$SCRIPT" \
    --results logs/jaadcr_evaluated/evaluation_medgemma-1.5-4b-it-vertex_without_options/jdcr_evaluation_20260215_173426.json \
    --groundtruth "$JDCR_GT" \
    --output-dir logs/jaadcr_evaluated/analysis_1.5-4b-it_without_options \
    --prefix jdcr_medgemma-1.5-4b-it_without_options

echo ""
echo "--- [8/12] JDCR: MedGemma-1.5-4B-IT WITH options ---"
python "$SCRIPT" \
    --results logs/jaadcr_evaluated/evaluation_medgemma-1.5-4b-it-vertex_with_options/jdcr_evaluation_20260215_192156.json \
    --groundtruth "$JDCR_GT" \
    --output-dir logs/jaadcr_evaluated/analysis_1.5-4b-it_with_options \
    --prefix jdcr_medgemma-1.5-4b-it_with_options

# --- JDCR: MedGemma-27B-IT ---
echo ""
echo "--- [9/12] JDCR: MedGemma-27B-IT WITHOUT options ---"
python "$SCRIPT" \
    --results logs/jaadcr_evaluated/evaluation_medgemma-27b-it-vertex_without_options/jdcr_evaluation_20260215_095311.json \
    --groundtruth "$JDCR_GT" \
    --output-dir logs/jaadcr_evaluated/analysis_27b-it_without_options \
    --prefix jdcr_medgemma-27b-it_without_options

echo ""
echo "--- [10/12] JDCR: MedGemma-27B-IT WITH options ---"
python "$SCRIPT" \
    --results logs/jaadcr_evaluated/evaluation_medgemma-27b-it-vertex_with_options/jdcr_evaluation_20260215_114957.json \
    --groundtruth "$JDCR_GT" \
    --output-dir logs/jaadcr_evaluated/analysis_27b-it_with_options \
    --prefix jdcr_medgemma-27b-it_with_options

# --- JDCR: MedGemma-4B-IT ---
echo ""
echo "--- [11/12] JDCR: MedGemma-4B-IT WITHOUT options ---"
python "$SCRIPT" \
    --results logs/jaadcr_evaluated/evaluation_medgemma-4b-it-vertex_without_options/jdcr_evaluation_20260215_024018.json \
    --groundtruth "$JDCR_GT" \
    --output-dir logs/jaadcr_evaluated/analysis_4b-it_without_options \
    --prefix jdcr_medgemma-4b-it_without_options

echo ""
echo "--- [12/12] JDCR: MedGemma-4B-IT WITH options ---"
python "$SCRIPT" \
    --results logs/jaadcr_evaluated/evaluation_medgemma-4b-it-vertex_with_options/jdcr_evaluation_20260215_043416.json \
    --groundtruth "$JDCR_GT" \
    --output-dir logs/jaadcr_evaluated/analysis_4b-it_with_options \
    --prefix jdcr_medgemma-4b-it_with_options

echo ""
echo "========================================================================"
echo "  ALL 12 ANALYSIS RUNS COMPLETE"
echo "========================================================================"
echo "Finished: $(date)"
echo ""
echo "Output files per analysis folder (example for NEJM 27B without options):"
echo "  nejm_medgemma-27b-it_without_options_side_by_side_results.csv"
echo "  nejm_medgemma-27b-it_without_options_metrics_by_variant.csv"
echo "  nejm_medgemma-27b-it_without_options_detailed_results.csv"
echo "  nejm_medgemma-27b-it_without_options_accuracy_by_variant.png"
echo "  nejm_medgemma-27b-it_without_options_pause_rate_by_variant.png"
echo "  nejm_medgemma-27b-it_without_options_confidence_vs_accuracy.png"
echo "  nejm_medgemma-27b-it_without_options_execution_time_by_variant.png"
echo "  nejm_medgemma-27b-it_without_options_evaluation_report.html"
echo ""
echo "NEJM analysis folders:"
echo "  logs/nejm_evaluated/analysis_1.5-4b-it_without_options/"
echo "  logs/nejm_evaluated/analysis_1.5-4b-it_with_options/"
echo "  logs/nejm_evaluated/analysis_27b-it_without_options/"
echo "  logs/nejm_evaluated/analysis_27b-it_with_options/"
echo "  logs/nejm_evaluated/analysis_4b-it_without_options/"
echo "  logs/nejm_evaluated/analysis_4b-it_with_options/"
echo ""
echo "JDCR analysis folders:"
echo "  logs/jaadcr_evaluated/analysis_1.5-4b-it_without_options/"
echo "  logs/jaadcr_evaluated/analysis_1.5-4b-it_with_options/"
echo "  logs/jaadcr_evaluated/analysis_27b-it_without_options/"
echo "  logs/jaadcr_evaluated/analysis_27b-it_with_options/"
echo "  logs/jaadcr_evaluated/analysis_4b-it_without_options/"
echo "  logs/jaadcr_evaluated/analysis_4b-it_with_options/"
