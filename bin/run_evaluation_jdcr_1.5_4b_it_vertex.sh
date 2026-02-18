#!/bin/bash
#
# Run JAADCR Evaluations - MedGemma-1.5-4B-IT Vertex AI Endpoint
#
# 250 total: 25 cases x 5 variants x 2 data paths x 1 model
#
# Model:
#   MedGemma-1.5-4B-IT (Vertex AI endpoint - no local GPU needed)
#
# BEFORE RUNNING:
#   1. Deploy MedGemma-1.5-4B-IT on Vertex AI Model Garden
#   2. Update src/agents/registry.py:
#      - "medgemma-vertex" -> status: "active"
#      - Update region and endpoint_id with your new endpoint
#   3. Run this script
#   4. AFTER completion: undeploy & delete endpoint to stop charges
#
# IMPORTANT: Do NOT run at the same time as other evaluation scripts!
#   All scripts use Gemini Pro (Google ADK) for orchestration.
#   Running them in parallel will exhaust the Gemini API daily quota.
#
# Usage (no GPU needed):
#   bash bin/run_evaluation_jdcr_1.5_4b_vertex.sh
#

set -e

PROJECT_DIR="/users/ha00014/Halimas_projects/MedGemma"
cd "$PROJECT_DIR"

echo "========================================================================"
echo "  MedGemma Evaluation Pipeline - MedGemma-1.5-4B-IT Vertex AI (JAADCR)"
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

# Verify 1.5-4B endpoint is configured (not placeholder)
ENDPOINT_1_5=$(grep -A 10 '"medgemma-vertex"' src/agents/registry.py | grep 'endpoint_id' | head -1)
if echo "$ENDPOINT_1_5" | grep -q "UPDATE_AFTER_DEPLOY"; then
    echo ""
    echo "ERROR: endpoint_id for medgemma-vertex is not configured!"
    echo "  1. Deploy MedGemma-1.5-4B-IT on Vertex AI Model Garden"
    echo "  2. Update src/agents/registry.py with the new endpoint_id and region"
    echo "  3. Set status to 'active'"
    exit 1
fi

STATUS_1_5=$(grep -A 3 '"medgemma-vertex"' src/agents/registry.py | grep 'status' | head -1)
if echo "$STATUS_1_5" | grep -q "stub"; then
    echo ""
    echo "ERROR: medgemma-vertex status is 'stub'!"
    echo "  Update status to 'active' in src/agents/registry.py"
    exit 1
fi

echo "1.5-4B-IT Vertex endpoint: configured"
echo ""

echo "========================================================================"
echo "  RUN 1/2: MedGemma-1.5-4B-IT (Vertex AI) + JAADCR without options"
echo "========================================================================"
echo "Started: $(date)"
python scripts/evaluate_jdcr_cases.py \
    --input JAADCR/jaadcr_input \
    --agent-model medgemma-vertex \
    --resume \
    --max-cases 25 \
    --output logs/jaadcr_evaluated/evaluation_medgemma-1.5-4b-it-vertex_without_options
echo "Finished: $(date)"
echo ""
echo "Waiting 60 seconds between runs to avoid Gemini API rate limits..."
sleep 60

echo "========================================================================"
echo "  RUN 2/2: MedGemma-1.5-4B-IT (Vertex AI) + JAADCR with options (MCQ)"
echo "========================================================================"
echo "Started: $(date)"
python scripts/evaluate_jdcr_cases.py \
    --input JAADCR/jaadcr_input_with_options \
    --agent-model medgemma-vertex \
    --resume \
    --max-cases 25 \
    --output logs/jaadcr_evaluated/evaluation_medgemma-1.5-4b-it-vertex_with_options
echo "Finished: $(date)"
echo ""

echo "========================================================================"
echo "  MedGemma-1.5-4B-IT JAADCR EVALUATIONS COMPLETE"
echo "========================================================================"
echo "Finished: $(date)"
echo ""
echo "Results saved to:"
echo "  logs/jaadcr_evaluated/evaluation_medgemma-1.5-4b-it-vertex_without_options/"
echo "  logs/jaadcr_evaluated/evaluation_medgemma-1.5-4b-it-vertex_with_options/"
echo ""
echo "REMINDER: Undeploy & delete the endpoint to stop hourly charges!"
