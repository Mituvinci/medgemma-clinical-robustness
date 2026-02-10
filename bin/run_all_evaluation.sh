#!/bin/bash
#
# Run All NEJM Evaluations
#
# 750 total: 25 cases x 5 variants x 2 data paths x 3 models
#
# Models:
#   1. MedGemma-27B-IT (Hugging Face)
#   2. MedGemma-4B-IT (Hugging Face)
#   3. MedGemma-1.5-4B-IT (Vertex AI)
#
# Usage (interactive GPU session):
#   bash bin/run_all_evaluation.sh
#
# Usage (batch job - runs in background):
#   sbatch bin/run_all_evaluation.sh
#
#SBATCH --partition=gpu_2day
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/evaluation_job_%j.log
#SBATCH --job-name=medgemma_eval

set -e

PROJECT_DIR="/users/ha00014/Halimas_projects/MedGemma"
cd "$PROJECT_DIR"

echo "========================================================================"
echo "  MedGemma Full Evaluation Pipeline"
echo "  750 evaluations: 25 cases x 5 variants x 2 data paths x 3 models"
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

# Offline mode: compute nodes have no internet access.
# HuggingFace/sentence-transformers models are cached locally but the library
# still tries to verify online. These flags force local-only cache usage.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SENTENCE_TRANSFORMERS_HOME=$HOME/.cache/huggingface/hub
echo "Offline mode enabled (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)"

# Check GPU
if nvidia-smi &>/dev/null; then
    echo "GPU available:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "WARNING: No GPU detected. MedGemma-27B-IT runs will be slow."
fi
echo ""

# ChromaDB check
python -c "
from src.rag.vector_store import VectorStore
vs = VectorStore()
stats = vs.get_collection_stats()
print(f'ChromaDB: {stats[\"count\"]} chunks ready')
" 2>/dev/null || echo "WARNING: ChromaDB check failed"
echo ""

echo "========================================================================"
echo "  RUN 1/6: MedGemma-27B-IT + without options"
echo "========================================================================"
echo "Started: $(date)"
python scripts/evaluate_nejim_cases.py \
    --input NEJIM/image_challenge_input \
    --agent-model medgemma \
    --output logs/evaluation_medgemma-27b-it_without_options
echo "Finished: $(date)"
echo ""

echo "========================================================================"
echo "  RUN 2/6: MedGemma-27B-IT + with options (MCQ)"
echo "========================================================================"
echo "Started: $(date)"
python scripts/evaluate_nejim_cases.py \
    --input NEJIM/image_challenge_input_with_options \
    --agent-model medgemma \
    --output logs/evaluation_medgemma-27b-it_with_options
echo "Finished: $(date)"
echo ""

echo "========================================================================"
echo "  RUN 3/6: MedGemma-4B-IT + without options"
echo "========================================================================"
echo "Started: $(date)"
python scripts/evaluate_nejim_cases.py \
    --input NEJIM/image_challenge_input \
    --agent-model medgemma-4b \
    --output logs/evaluation_medgemma-4b-it_without_options
echo "Finished: $(date)"
echo ""

echo "========================================================================"
echo "  RUN 4/6: MedGemma-4B-IT + with options (MCQ)"
echo "========================================================================"
echo "Started: $(date)"
python scripts/evaluate_nejim_cases.py \
    --input NEJIM/image_challenge_input_with_options \
    --agent-model medgemma-4b \
    --output logs/evaluation_medgemma-4b-it_with_options
echo "Finished: $(date)"
echo ""

echo "========================================================================"
echo "  RUN 5/6: MedGemma-1.5-4B-IT (Vertex AI) + without options"
echo "========================================================================"
echo "Started: $(date)"
python scripts/evaluate_nejim_cases.py \
    --input NEJIM/image_challenge_input \
    --agent-model medgemma-vertex \
    --output logs/evaluation_medgemma-1.5-4b-it_without_options
echo "Finished: $(date)"
echo ""

echo "========================================================================"
echo "  RUN 6/6: MedGemma-1.5-4B-IT (Vertex AI) + with options (MCQ)"
echo "========================================================================"
echo "Started: $(date)"
python scripts/evaluate_nejim_cases.py \
    --input NEJIM/image_challenge_input_with_options \
    --agent-model medgemma-vertex \
    --output logs/evaluation_medgemma-1.5-4b-it_with_options
echo "Finished: $(date)"
echo ""

echo "========================================================================"
echo "  ALL EVALUATIONS COMPLETE"
echo "========================================================================"
echo "Finished: $(date)"
echo ""
echo "Results saved to:"
echo "  logs/evaluation_medgemma-27b-it_without_options/"
echo "  logs/evaluation_medgemma-27b-it_with_options/"
echo "  logs/evaluation_medgemma-4b-it_without_options/"
echo "  logs/evaluation_medgemma-4b-it_with_options/"
echo "  logs/evaluation_medgemma-1.5-4b-it_without_options/"
echo "  logs/evaluation_medgemma-1.5-4b-it_with_options/"
