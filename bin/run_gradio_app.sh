#!/bin/bash
#
# Run MedGemma Gradio UI with GPU
#
# This script:
# 1. Checks if you're in a GPU session
# 2. Requests GPU if needed
# 3. Activates conda environment
# 4. Launches Gradio app
#

set -e  # Exit on error

PROJECT_DIR="/users/ha00014/Halimas_projects/MedGemma"
cd "$PROJECT_DIR"

echo "========================================================================"
echo "  MedGemma Gradio UI Launcher"
echo "========================================================================"
echo ""

# Function to check if we're in a GPU session
check_gpu_session() {
    if nvidia-smi &>/dev/null; then
        echo "✓ Already in GPU session"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
        echo ""
        return 0
    else
        echo "✗ Not in GPU session"
        return 1
    fi
}

# Function to request GPU session
request_gpu() {
    echo "========================================================================"
    echo "  Requesting Interactive GPU Session"
    echo "========================================================================"
    echo ""
    echo "Requesting: 1 GPU, 64GB RAM, 8 hours (for interactive use)"
    echo "Partition: gpu_2day (26 GPUs available)"
    echo ""
    echo "⏳ Waiting for GPU allocation..."
    echo ""

    srun --partition=gpu_2day \
         --gres=gpu:1 \
         --mem=64G \
         --time=8:00:00 \
         --pty bash "$0" --in-gpu-session

    exit 0
}

# Main execution
main() {
    # Check if we're already in GPU session
    if ! check_gpu_session; then
        # Not in GPU session - request one
        request_gpu
    fi

    echo "========================================================================"
    echo "  Activating Conda Environment"
    echo "========================================================================"
    echo ""

    # Activate conda
    if [ -z "$CONDA_PREFIX" ] || [[ "$CONDA_PREFIX" != *"pytorch"* ]]; then
        eval "$(conda shell.bash hook)"
        conda activate pytorch
        echo "✓ Activated: $(conda info --envs | grep '*' | awk '{print $1}')"
    else
        echo "✓ Already in pytorch environment"
    fi

    # Offline mode: compute nodes have no internet access.
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export SENTENCE_TRANSFORMERS_HOME=$HOME/.cache/huggingface/hub
    echo "✓ Offline mode enabled (HF_HUB_OFFLINE=1)"
    echo ""

    echo "========================================================================"
    echo "  Pre-flight Checks"
    echo "========================================================================"
    echo ""

    # Check ChromaDB
    echo "Checking ChromaDB..."
    python -c "
from src.rag.vector_store import VectorStore
vs = VectorStore()
stats = vs.get_collection_stats()
print(f'  ✓ ChromaDB: {stats[\"count\"]} chunks ready')
" 2>/dev/null || echo "  ⚠ ChromaDB not initialized (run: python main.py --mode ingest)"

    # Check API keys
    echo "Checking API keys..."
    python -c "
from config.config import settings
errors = []
if not settings.huggingface_api_key:
    errors.append('HUGGINGFACE_API_KEY missing')
if not settings.gemini_api_key:
    errors.append('GEMINI_API_KEY missing')
if errors:
    print('  ❌ Missing:', ', '.join(errors))
    exit(1)
else:
    print('  ✓ All API keys configured')
" || exit 1

    echo ""

    echo "========================================================================"
    echo "  Launching Gradio App"
    echo "========================================================================"
    echo ""
    echo "Starting MedGemma Clinical Robustness Assistant..."
    echo ""
    echo "Access the UI at: http://dsis001:7860"
    echo "(Or find the URL in the output below)"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    echo "------------------------------------------------------------------------"
    echo ""

    # Launch Gradio
    python main.py --mode app

    echo ""
    echo "========================================================================"
    echo "  Gradio App Stopped"
    echo "========================================================================"
    echo ""
}

# Handle being called from within GPU session
if [ "$1" = "--in-gpu-session" ]; then
    main
else
    main
fi
