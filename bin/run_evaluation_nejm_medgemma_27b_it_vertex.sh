#!/bin/bash
#
# Run NEJM Evaluations - MedGemma-27B-IT Vertex AI Endpoint
#
# 250 total: 25 cases x 5 variants x 2 data paths x 1 model
#
# Model:
#   MedGemma-27B-IT (Vertex AI endpoint - no local GPU needed)
#
# IMPORTANT: Do NOT run this at the same time as other evaluation scripts!
#   All scripts use Gemini Pro (Google ADK) for orchestration.
#   Running them in parallel will exhaust the Gemini API daily quota (429 errors).
#   Run this script AFTER other evaluations have completed, or vice versa.
#
# FEATURES:
#   - RESUME SUPPORT: Use --resume flag to continue from last checkpoint
#   - GEMINI FALLBACK: Auto-switches between 9 Gemini models when quota hits
#   - NO GPU NEEDED: Can run from login node (cloud endpoint)
#
# Usage (no GPU needed - can run from login node):
#   bash bin/run_evaluation_27b_vertex.sh
#
# Usage (batch job - CPU-only partition is fine):
#   sbatch bin/run_evaluation_27b_vertex.sh
#
#SBATCH --partition=shared
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/evaluation_27b_vertex_job_%j.log
#SBATCH --job-name=medgemma_27b_eval_vertex

set -e

PROJECT_DIR="/users/ha00014/Halimas_projects/MedGemma"
cd "$PROJECT_DIR"

echo "========================================================================"
echo "  MedGemma Evaluation Pipeline - MedGemma-27B-IT Vertex AI"
echo "  250 evaluations: 25 cases x 5 variants x 2 data paths x 1 model"
echo "  Competition Model: Multimodal (image+text)"
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

# Verify 27B endpoint is configured
ENDPOINT_27B=$(grep -A 10 '"medgemma-27b-it-vertex"' src/agents/registry.py | grep 'endpoint_id' | grep -o '"mg-endpoint[^"]*"' | tr -d '"')
echo "Vertex 27B endpoint: ${ENDPOINT_27B:-NOT SET}"
if [ -z "$ENDPOINT_27B" ]; then
    echo "ERROR: endpoint_id for medgemma-27b-it-vertex not configured in src/agents/registry.py"
    echo "Deploy MedGemma-27B-IT first and update the endpoint_id."
    exit 1
fi
echo ""

echo "========================================================================"
echo "  RUN 1/2: MedGemma-27B-IT (Vertex AI) + without options"
echo "========================================================================"
echo "Started: $(date)"
python scripts/evaluate_nejim_cases.py \
    --input NEJIM/image_challenge_input \
    --agent-model medgemma-27b-it-vertex \
    --resume \
    --output logs/evaluation_medgemma-27b-it-vertex_without_options
echo "Finished: $(date)"
echo ""
echo "Waiting 60 seconds between runs to avoid Gemini API rate limits..."
sleep 60

echo "========================================================================"
echo "  RUN 2/2: MedGemma-27B-IT (Vertex AI) + with options (MCQ)"
echo "========================================================================"
echo "Started: $(date)"
python scripts/evaluate_nejim_cases.py \
    --input NEJIM/image_challenge_input_with_options \
    --agent-model medgemma-27b-it-vertex \
    --resume \
    --output logs/evaluation_medgemma-27b-it-vertex_with_options
echo "Finished: $(date)"
echo ""

echo "========================================================================"
echo "  MedGemma-27B-IT VERTEX AI EVALUATIONS COMPLETE"
echo "========================================================================"
echo "Finished: $(date)"
echo ""
echo "Results saved to:"
echo "  logs/evaluation_medgemma-27b-it-vertex_without_options/"
echo "  logs/evaluation_medgemma-27b-it-vertex_with_options/"
echo ""
echo "Summary:"
echo "  - 250 total evaluations (125 + 125)"
echo "  - Model: MedGemma-27B-IT (Vertex AI endpoint)"
echo "  - Multimodal: image+text (instruction-tuned)"
echo "  - Orchestrator: Gemini Pro with 9-model fallback"
