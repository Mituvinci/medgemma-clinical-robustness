#!/bin/bash
#
# Run NEJM Evaluations - HuggingFace Models (Local GPU)
#
# 500 total: 25 cases x 5 variants x 2 data paths x 2 models
#
# Models:
#   1. MedGemma-27B-IT (Hugging Face, local GPU)
#   2. MedGemma-4B-IT (Hugging Face, local GPU)
#
# IMPORTANT: Do NOT run this at the same time as run_evaluation_vertex.sh!
#   Both scripts use Gemini Pro (Google ADK) for orchestration.
#   Running them in parallel will exhaust the Gemini API daily quota (429 errors).
#   Run this script AFTER run_evaluation_vertex.sh has fully completed, or vice versa.
#
# Usage (interactive GPU session):
#   bash bin/run_evaluation_hf.sh
#
# Usage (batch job - runs in background):
#   sbatch bin/run_evaluation_hf.sh
#
#SBATCH --partition=gpu_2day
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/evaluation_hf_job_%j.log
#SBATCH --job-name=medgemma_eval_hf

set -e

PROJECT_DIR="/users/ha00014/Halimas_projects/MedGemma"
cd "$PROJECT_DIR"

echo "========================================================================"
echo "  MedGemma Evaluation Pipeline - HuggingFace Models (Local GPU)"
echo "  500 evaluations: 25 cases x 5 variants x 2 data paths x 2 models"
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

# Force ChromaDB (local) since GPU compute nodes have no internet.
# Vertex RAG requires API calls which are blocked on compute nodes.
export RAG_BACKEND=chroma
echo "RAG Backend: ChromaDB (local) - forced for offline compute nodes"

# Check GPU
if nvidia-smi &>/dev/null; then
    echo "GPU available:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "ERROR: No GPU detected. HuggingFace models require local GPU."
    echo "Request a GPU session: srun --partition=gpu_2day --gres=gpu:1 --mem=64G --time=72:00:00 --pty bash"
    exit 1
fi
echo ""

# RAG Backend check
RAG_BACKEND=$(grep '^RAG_BACKEND=' .env | cut -d= -f2)
echo "RAG Backend: ${RAG_BACKEND:-chroma}"
if [ "$RAG_BACKEND" = "vertex" ]; then
    echo "Using Vertex AI RAG (cloud) - requires internet access from compute node"
    echo "WARNING: If compute nodes have no internet, Vertex RAG calls will fail."
    echo "         The evaluation will still run but RAG retrieval may return 0 results."
    echo "         To use local RAG instead, set RAG_BACKEND=chroma in .env"
else
    python -c "
from src.rag.vector_store import VectorStore
vs = VectorStore()
stats = vs.get_collection_stats()
print(f'ChromaDB: {stats[\"count\"]} chunks ready')
" 2>/dev/null || echo "WARNING: ChromaDB check failed"
fi
echo ""

echo "========================================================================"
echo "  RUN 1/4: MedGemma-27B-IT + without options"
echo "========================================================================"
echo "Started: $(date)"
python scripts/evaluate_nejim_cases.py \
    --input NEJIM/image_challenge_input \
    --agent-model medgemma \
    --resume \
    --output logs/evaluation_medgemma-27b-it_without_options
echo "Finished: $(date)"
echo ""
echo "Waiting 60 seconds between runs to avoid Gemini API rate limits..."
sleep 60

echo "========================================================================"
echo "  RUN 2/4: MedGemma-27B-IT + with options (MCQ)"
echo "========================================================================"
echo "Started: $(date)"
python scripts/evaluate_nejim_cases.py \
    --input NEJIM/image_challenge_input_with_options \
    --agent-model medgemma \
    --resume \
    --output logs/evaluation_medgemma-27b-it_with_options
echo "Finished: $(date)"
echo ""
echo "Waiting 60 seconds between runs to avoid Gemini API rate limits..."
sleep 60

echo "========================================================================"
echo "  RUN 3/4: MedGemma-4B-IT + without options"
echo "========================================================================"
echo "Started: $(date)"
python scripts/evaluate_nejim_cases.py \
    --input NEJIM/image_challenge_input \
    --agent-model medgemma-4b \
    --resume \
    --output logs/evaluation_medgemma-4b-it_without_options
echo "Finished: $(date)"
echo ""
echo "Waiting 60 seconds between runs to avoid Gemini API rate limits..."
sleep 60

echo "========================================================================"
echo "  RUN 4/4: MedGemma-4B-IT + with options (MCQ)"
echo "========================================================================"
echo "Started: $(date)"
python scripts/evaluate_nejim_cases.py \
    --input NEJIM/image_challenge_input_with_options \
    --agent-model medgemma-4b \
    --resume \
    --output logs/evaluation_medgemma-4b-it_with_options
echo "Finished: $(date)"
echo ""

echo "========================================================================"
echo "  HUGGINGFACE EVALUATIONS COMPLETE"
echo "========================================================================"
echo "Finished: $(date)"
echo ""
echo "Results saved to:"
echo "  logs/evaluation_medgemma-27b-it_without_options/"
echo "  logs/evaluation_medgemma-27b-it_with_options/"
echo "  logs/evaluation_medgemma-4b-it_without_options/"
echo "  logs/evaluation_medgemma-4b-it_with_options/"
echo ""
echo "Next: Run Vertex AI evaluations (no GPU needed):"
echo "  bash bin/run_evaluation_vertex.sh"
