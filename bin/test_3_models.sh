#!/bin/bash
#
# Quick Test: 1 NEJM Case × 3 MedGemma Models
#
# Tests:
#   1. MedGemma-27B-IT (Hugging Face)
#   2. MedGemma-4B-IT (Hugging Face) - NEW
#   3. MedGemma-1.5-4B-IT (Vertex AI)
#
# Usage:
#   bash bin/test_3_models.sh
#

set -e

PROJECT_DIR="/users/ha00014/Halimas_projects/MedGemma"
cd "$PROJECT_DIR"

echo "========================================================================"
echo "  Quick Test: 1 NEJM Case × 3 MedGemma Models"
echo "========================================================================"
echo ""
echo "Started: $(date)"
echo ""

# Activate conda
if [ -z "$CONDA_PREFIX" ] || [[ "$CONDA_PREFIX" != *"pytorch"* ]]; then
    eval "$(conda shell.bash hook)"
    conda activate pytorch
    echo "✓ Activated conda: pytorch"
else
    echo "✓ Conda already active: $CONDA_PREFIX"
fi
echo ""

# Offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SENTENCE_TRANSFORMERS_HOME=$HOME/.cache/huggingface/hub
echo "✓ Offline mode enabled (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)"
echo ""

# Check GPU
if nvidia-smi &>/dev/null; then
    echo "✓ GPU available:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "⚠ WARNING: No GPU detected. Models will run slowly."
fi
echo ""

# Check RAG backend
echo "RAG Backend Configuration:"
grep "RAG_BACKEND" .env || echo "  (not set, using default)"
echo ""

echo "========================================================================"
echo "  TEST 1/3: MedGemma-27B-IT (Hugging Face, 27B params)"
echo "========================================================================"
echo "Started: $(date)"
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1 && \
python scripts/evaluate_nejim_cases.py \
    --input NEJIM/image_challenge_input \
    --agent-model medgemma \
    --max-cases 1 \
    --output logs/test_medgemma-27b-it
echo "✓ Finished: $(date)"
echo ""

echo "========================================================================"
echo "  TEST 2/3: MedGemma-4B-IT (Hugging Face, 4B params)"
echo "========================================================================"
echo "Started: $(date)"
echo "⏳ First run will download model weights (~8GB), subsequent runs use cache"
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1 && \
python scripts/evaluate_nejim_cases.py \
    --input NEJIM/image_challenge_input \
    --agent-model medgemma-4b \
    --max-cases 1 \
    --output logs/test_medgemma-4b-it
echo "✓ Finished: $(date)"
echo ""

echo "========================================================================"
echo "  TEST 3/3: MedGemma-1.5-4B-IT (Vertex AI, cloud endpoint)"
echo "========================================================================"
echo "Started: $(date)"
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1 && \
python scripts/evaluate_nejim_cases.py \
    --input NEJIM/image_challenge_input \
    --agent-model medgemma-vertex \
    --max-cases 1 \
    --output logs/test_medgemma-1.5-4b-it
echo "✓ Finished: $(date)"
echo ""

echo "========================================================================"
echo "  ALL TESTS COMPLETE ✓"
echo "========================================================================"
echo "Finished: $(date)"
echo ""
echo "Results saved to:"
echo "  logs/test_medgemma-27b-it/"
echo "  logs/test_medgemma-4b-it/"
echo "  logs/test_medgemma-1.5-4b-it/"
echo ""
echo "Compare the outputs to verify all 3 models work correctly."
echo ""
echo "If all tests pass, run the full evaluation:"
echo "  bash bin/run_all_evaluation.sh"
echo "========================================================================"
