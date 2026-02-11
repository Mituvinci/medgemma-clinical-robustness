#!/bin/bash
#
# Run NEJM Evaluations - Vertex AI Model (Cloud API)
#
# 250 total: 25 cases x 5 variants x 2 data paths x 1 model
#
# Model:
#   MedGemma-1.5-4B-IT (Vertex AI endpoint - no local GPU needed)
#
# PREREQUISITE: Deploy MedGemma-1.5-4B-IT on Vertex AI and update:
#   1. endpoint_id in src/agents/registry.py (line ~93)
#   2. region in src/agents/registry.py (line ~92) if different from us-central1
#
# Usage (no GPU needed - can run from login node):
#   bash bin/run_evaluation_vertex.sh
#
# Usage (batch job - CPU-only partition is fine):
#   sbatch bin/run_evaluation_vertex.sh
#
#SBATCH --partition=shared
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/evaluation_vertex_job_%j.log
#SBATCH --job-name=medgemma_eval_vertex

set -e

PROJECT_DIR="/users/ha00014/Halimas_projects/MedGemma"
cd "$PROJECT_DIR"

echo "========================================================================"
echo "  MedGemma Evaluation Pipeline - Vertex AI Model (Cloud API)"
echo "  250 evaluations: 25 cases x 5 variants x 2 data paths x 1 model"
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

# DO NOT set HF_HUB_OFFLINE - Vertex AI needs internet access
echo "Online mode (Vertex AI requires internet for API calls)"

# Verify Google Cloud auth
if gcloud auth application-default print-access-token &>/dev/null; then
    echo "Google Cloud auth: OK"
else
    echo "ERROR: Google Cloud auth not configured."
    echo "Run: gcloud auth application-default login"
    exit 1
fi

# Verify project
PROJECT=$(gcloud config get-value project 2>/dev/null)
echo "Google Cloud project: $PROJECT"

# Verify endpoint is configured (lightweight check, no torch import)
ENDPOINT=$(grep 'endpoint_id' src/agents/registry.py | grep -o '"mg-endpoint[^"]*"' | tr -d '"')
echo "Vertex endpoint: ${ENDPOINT:-NOT SET}"
if [ -z "$ENDPOINT" ]; then
    echo "ERROR: endpoint_id not configured in src/agents/registry.py"
    echo "Deploy MedGemma-1.5 first and update the endpoint_id."
    exit 1
fi
echo ""

echo "========================================================================"
echo "  RUN 1/2: MedGemma-1.5-4B-IT (Vertex AI) + without options"
echo "========================================================================"
echo "Started: $(date)"
python scripts/evaluate_nejim_cases.py \
    --input NEJIM/image_challenge_input \
    --agent-model medgemma-vertex \
    --output logs/evaluation_medgemma-1.5-4b-it_without_options
echo "Finished: $(date)"
echo ""

echo "========================================================================"
echo "  RUN 2/2: MedGemma-1.5-4B-IT (Vertex AI) + with options (MCQ)"
echo "========================================================================"
echo "Started: $(date)"
python scripts/evaluate_nejim_cases.py \
    --input NEJIM/image_challenge_input_with_options \
    --agent-model medgemma-vertex \
    --output logs/evaluation_medgemma-1.5-4b-it_with_options
echo "Finished: $(date)"
echo ""

echo "========================================================================"
echo "  VERTEX AI EVALUATIONS COMPLETE"
echo "========================================================================"
echo "Finished: $(date)"
echo ""
echo "Results saved to:"
echo "  logs/evaluation_medgemma-1.5-4b-it_without_options/"
echo "  logs/evaluation_medgemma-1.5-4b-it_with_options/"
