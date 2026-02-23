#!/bin/bash
#
# Run MedGemma Gradio Demo — Vertex AI Mode
#
# Uses MedGemma via Vertex AI endpoint (configured in .env).
# No local GPU required — runs on login node with internet access.
# Gemini orchestration via Gemini API (also requires internet).
#
# Usage:
#   bash bin/run_gradio_demo.sh
#

set -e

PROJECT_DIR="/users/ha00014/Halimas_projects/MedGemma"
cd "$PROJECT_DIR"

# Read AGENT_MODEL from .env
AGENT_MODEL=$(grep -E '^AGENT_MODEL=' .env | cut -d= -f2)

echo "========================================================================"
echo "  MedGemma Gradio Demo — Vertex AI Mode"
echo "  Model: ${AGENT_MODEL:-medgemma-vertex}"
echo "  No GPU required — runs on login node"
echo "========================================================================"
echo ""

# Activate conda
if [ -z "$CONDA_PREFIX" ] || [[ "$CONDA_PREFIX" != *"pytorch"* ]]; then
    eval "$(conda shell.bash hook)"
    conda activate pytorch
fi
echo "✓ Conda environment: $(basename $CONDA_PREFIX)"

# Ensure internet mode
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
echo "✓ Online mode enabled (Vertex AI requires internet)"
echo ""

# Check API keys
python -c "
from config.config import settings
errors = []
if not settings.google_api_key:
    errors.append('GOOGLE_API_KEY missing')
if not settings.google_cloud_project:
    errors.append('GOOGLE_CLOUD_PROJECT missing')
if errors:
    print('  ERROR Missing:', ', '.join(errors))
    exit(1)
else:
    print('  API keys OK')
" || exit 1

echo ""
echo "Starting Gradio demo..."
echo "Access at: http://$(hostname):7860"
echo "Press Ctrl+C to stop"
echo "------------------------------------------------------------------------"
echo ""

python main.py --mode app
