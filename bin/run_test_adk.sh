#!/bin/bash
#
# Test ADK Multi-Agent Workflow with GPU
#
# This script:
# 1. Checks if you're in a GPU session
# 2. Requests GPU if needed
# 3. Activates conda environment
# 4. Runs the ADK workflow test
#

set -e  # Exit on error

PROJECT_DIR="/users/ha00014/Halimas_projects/MedGemma/"
cd "$PROJECT_DIR"

echo "========================================================================"
echo "  MedGemma ADK Workflow Test"
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
    echo "Requesting: 1 GPU, 64GB RAM, 4 hours"
    echo "Partition: gpu_2day (26 GPUs available)"
    echo ""
    echo "⏳ Waiting for GPU allocation..."
    echo ""

    srun --partition=gpu_2day \
         --gres=gpu:1 \
         --mem=64G \
         --time=4:00:00 \
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
    echo ""

    echo "========================================================================"
    echo "  Checking ChromaDB Status"
    echo "========================================================================"
    echo ""

    python -c "
from src.rag.vector_store import VectorStore
vs = VectorStore()
stats = vs.get_collection_stats()
print(f'ChromaDB chunks: {stats[\"count\"]}')
print(f'Collection: {stats[\"name\"]}')
" 2>/dev/null || echo "⚠ ChromaDB check failed (may need ingestion)"

    echo ""

    echo "========================================================================"
    echo "  Running ADK Workflow Test"
    echo "========================================================================"
    echo ""
    echo "Testing: Triage → Research → Diagnostic workflow"
    echo "Using: MedGemma-27B-IT (local GPU) + Gemini Pro (orchestration)"
    echo ""

    # Run the test (from project root)
    python tests/test_adk_workflow.py

    TEST_EXIT_CODE=$?

    echo ""
    echo "========================================================================"
    echo "  Test Results"
    echo "========================================================================"
    echo ""

    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo "✅ ADK workflow test PASSED"
        echo ""
        echo "Checking generated logs..."

        # Show latest session file
        LATEST_SESSION=$(ls -t logs/sessions/case_*.json 2>/dev/null | head -1)
        if [ -n "$LATEST_SESSION" ]; then
            echo "Latest session: $LATEST_SESSION"
            echo ""
            python -c "
import json
with open('$LATEST_SESSION') as f:
    data = json.load(f)
print(f'Session ID: {data[\"session_id\"]}')
print(f'Case ID: {data[\"case_id\"]}')
print(f'Model: {data[\"model\"]}')
print(f'Steps: {data[\"metadata\"][\"total_steps\"]}')
print(f'Completed: {data[\"completed\"]}')
" 2>/dev/null || echo "Could not parse session file"
        else
            echo "⚠ No session files found in logs/sessions/"
        fi
    else
        echo "❌ ADK workflow test FAILED (exit code: $TEST_EXIT_CODE)"
        echo ""
        echo "Check errors above for details"
    fi

    echo ""
    echo "========================================================================"
    echo ""
}

# Handle being called from within GPU session
if [ "$1" = "--in-gpu-session" ]; then
    main
else
    main
fi
