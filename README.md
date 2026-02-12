# MedGemma Clinical Robustness Assistant

> A multi-agent clinical decision support system for dermatology, demonstrating diagnostic robustness under varying levels of clinical information.

**Competition**: [Med-Gemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
**Submission Deadline**: February 23, 2026

---

## Overview

This "Partner-Style" Clinical Assistant uses a hierarchical multi-agent workflow to provide safe, explainable dermatology consultations. The system demonstrates a key principle: **never guess when data is missing** — instead, it proactively requests clarification before making diagnostic decisions.

### Key Features

- **Multi-Agent Workflow**: 3 specialized agents (Triage, Research, Diagnostic) orchestrated via Google ADK
- **Safety-First**: Pauses and asks questions when clinical data is incomplete
- **Evidence-Based**: Grounds all reasoning in AAD/StatPearls/JAADCR dermatology guidelines via RAG
- **Transparent**: Generates structured SOAP notes with full reasoning and citations
- **Robust**: Evaluated across 750 scenarios with varying data completeness
- **Multi-Model**: Supports 3 MedGemma variants (27B, 4B, 1.5-4B-IT)
- **Multi-GPU**: MedGemma-27B-IT automatically distributed across multiple GPUs for full bfloat16 precision
- **Resilient Orchestration**: 9 Gemini model fallbacks for automatic API quota rotation

---

## Architecture

### Hybrid Two-Tier System

**TIER 1: Orchestration (Google ADK + Gemini)**
- Workflow management and task delegation
- Agent routing and coordination
- **Role**: Orchestrator only — performs NO clinical reasoning
- **Quota Resilience**: 9 Gemini models (Pro and Flash variants) configured as fallbacks. When one model's daily API quota is exhausted (~100 requests/day), the system automatically switches to the next model. This enables ~900 orchestrator requests/day without manual intervention. Models used: `gemini-2.5-pro`, `gemini-pro-latest`, `gemini-3-pro-preview`, `gemini-2.5-flash`, `gemini-3-flash-preview`, and others.

**TIER 2: Clinical Reasoning (MedGemma Specialists)**
- **MedGemma-27B-IT**: Primary medical reasoning (Hugging Face, 27B params, multi-GPU)
- **MedGemma-4B-IT**: Lightweight medical reasoning (Hugging Face, 4B params, single GPU)
- **MedGemma-1.5-4B-IT**: Cloud medical reasoning (Vertex AI endpoint, 4B params)
- **Role**: ALL medical diagnosis and clinical analysis

### Multi-Agent Workflow

```
User Input (multimodal)
    |
+--------------------------------------+
|   Root Coordinator (Gemini/ADK)      |  <-- Orchestration only
|   Auto-rotates across 9 Gemini       |
|   models on quota exhaustion         |
+--------------------------------------+
    |
+--------------------------------------+
|   Triage Agent                       |
|   - Checks case completeness         |
|   - MedGemma identifies missing data |  <-- Medical reasoning
|   - Asks clarifying questions        |
+--------------------------------------+
    |
+--------------------------------------+
|   Research Agent                     |
|   - Retrieves clinical guidelines    |  <-- RAG retrieval
|   - MedGemma synthesizes evidence    |  <-- Medical reasoning
+--------------------------------------+
    |
+--------------------------------------+
|   Diagnostic Agent                   |
|   - MedGemma generates SOAP note     |  <-- Medical reasoning
|   - Differential diagnosis           |
|   - Guideline citations              |
+--------------------------------------+
    |
Structured SOAP Output
```

### Multi-GPU Model Distribution

MedGemma-27B-IT (27 billion parameters) requires significant GPU memory. The system automatically handles GPU distribution:

| GPU Setup | Memory | Strategy | Precision |
|-----------|--------|----------|-----------|
| 1x GPU (< 40GB) | ~24GB | 4-bit quantization (NF4) | ~95-98% quality |
| 1x GPU (40-80GB) | ~46GB | 4-bit quantization | ~95-98% quality |
| 2x GPU (80GB+) | ~92GB | Full bfloat16, auto-distributed | Full quality |
| 3x+ GPU | 120GB+ | Full bfloat16, auto-distributed | Full quality |

The adapter uses `device_map="auto"` from Hugging Face Transformers to automatically split model layers across all available GPUs. With 2x NVIDIA A40 (92GB total), the 27B model runs in full bfloat16 precision without quantization loss.

MedGemma-4B-IT (4 billion parameters) fits comfortably on a single GPU (~8GB in bfloat16).

---

## Technology Stack

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `google-adk` | 1.23.0 | Agent Development Kit — multi-agent orchestration |
| `google-genai` | 1.61.0 | Gemini API — orchestrator model |
| `google-cloud-aiplatform` | 1.135.0 | Vertex AI — RAG corpus and MedGemma endpoint |
| `transformers` | 5.0.0 | Hugging Face — local MedGemma-27B/4B inference |
| `torch` | 2.10.0 | PyTorch — GPU computation and model parallelism |
| `bitsandbytes` | 0.49.1 | 4-bit quantization (NF4) for memory-constrained GPUs |
| `chromadb` | 1.4.1 | Local vector database — fallback RAG backend |
| `sentence-transformers` | 5.2.2 | Embeddings — all-MiniLM-L6-v2 for ChromaDB |
| `gradio` | 6.3.0 | Web UI framework — multimodal clinical interface |
| `huggingface-hub` | 1.3.5 | Model downloads — gated MedGemma access |
| `pypdf` | 6.6.2 | PDF parsing — clinical guideline ingestion |
| `pydantic` | 2.11.10 | Data models — ClinicalCase, SOAPNote, etc. |

### Knowledge Base

| Source | Documents | Description |
|--------|-----------|-------------|
| AAD Guidelines | 10+ | American Academy of Dermatology clinical practice guidelines |
| StatPearls | 5 | Evidence-based medical education reference articles |
| JAADCR | 22 | Journal of the American Academy of Dermatology Case Reports |

**RAG Backends**:
- **Vertex AI RAG** (cloud): 37 documents, text-embedding-005, managed by Google Cloud
- **ChromaDB** (local fallback): 2,668 chunks, all-MiniLM-L6-v2 embeddings, SQLite-backed

---

## Features

### Multimodal Clinical Input
- Dermatology lesion images (JPEG/PNG)
- Clinical case vignettes (text/PDF/JSON)
- Free-text patient history
- Structured exam findings

### Intelligent Triage
- Detects missing history, exam, or image data
- Asks specific follow-up questions
- Never guesses with incomplete information
- Safety-first medical reasoning

### Dual RAG Pipeline
- **Vertex AI RAG** (primary): Cloud-hosted, server-side embeddings, no local model needed
- **ChromaDB** (fallback): Local vector DB for offline/GPU compute nodes without internet
- Top-5 relevant passages with similarity scores per query
- Citations included in every SOAP note

### Structured SOAP Output
```
S (Subjective): Patient-reported symptoms and history
O (Objective): Physical exam findings and diagnostic data
A (Assessment): Differential diagnosis with evidence
P (Plan): Recommended workup and treatment
```

### Orchestrator Quota Resilience
The Gemini API has a daily quota of ~100 requests per model. For large-scale evaluations (750 cases), the system configures 9 Gemini models as fallbacks:
1. When the current model returns a 429 (quota exhausted) error
2. The system automatically switches to the next Gemini model
3. Each model has its own independent quota
4. This enables ~900 orchestrator requests/day without manual intervention

### Audit-Grade Logging
- Complete session logs (JSON/TXT/Markdown)
- Explicit model attribution for every step
- PII-safe (redacts patient identifiers)
- Reproducible for research/auditing

---

## Evaluation

### Test Dataset
- **25 dermatology cases** (NEJM Image Challenge style)
- **5 context variants** per case:
  - Original (complete)
  - History only
  - Image only
  - Exam only
  - Exam restricted (minimal findings)
- **2 data formats**: With/without MCQ options
- **3 models** x 2 formats x 25 cases x 5 variants = **750 evaluations**

### Metrics
1. **Pause Rate**: % of cases where agent asks for more data
2. **Diagnostic Accuracy**: Matches correct diagnosis
3. **Robustness**: Performance delta across context states
4. **Guideline Retrieval**: Relevance of cited sources
5. **Response Quality**: SOAP note completeness
6. **False Positive Rate**: Incorrect pauses on complete cases
7. **Execution Time**: Average time per case
8. **Error Rate**: System failures

### Evaluation Infrastructure
- **HuggingFace models** (500 evals): Runs on HPC GPU nodes via SLURM, 2x NVIDIA A40, ChromaDB for RAG (offline compute nodes)
- **Vertex AI model** (250 evals): Runs on login node or CPU partition, Vertex AI RAG (cloud)
- **Resume support**: Partial results saved every 5 evaluations, `--resume` flag skips completed cases

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (for local MedGemma inference)
  - Minimum: 1x GPU with 24GB VRAM (4-bit quantization for 27B model)
  - Recommended: 2x GPU with 46GB+ VRAM each (full bfloat16 for 27B model)
  - MedGemma-4B-IT: Any single GPU with 16GB+ VRAM
- 64GB+ system RAM recommended
- Google Cloud account (for Vertex AI features)
- Hugging Face account with access to [MedGemma gated models](https://huggingface.co/google/medgemma-27b-it)
- Conda or virtualenv for environment management

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/MedGemma.git
cd MedGemma

# Create environment
conda create -n medgemma python=3.10
conda activate medgemma

# Install dependencies
pip install -r requirements.txt

# For NVIDIA GPUs with CUDA 12.8 (recommended for better performance):
pip install --force-reinstall torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

### Configure API Keys

Create `.env` file:
```bash
# Required
HUGGINGFACE_API_KEY=your_hf_key_here       # For MedGemma-27B/4B-IT download
GOOGLE_API_KEY=your_google_key_here         # For Gemini orchestrator
GOOGLE_CLOUD_PROJECT=your_project_id_here   # For Vertex AI services

# Optional (for Vertex AI)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# RAG Backend (choose one)
RAG_BACKEND=vertex   # Cloud-based (requires internet)
# RAG_BACKEND=chroma # Local fallback (no internet needed)

# Vertex AI RAG (if using vertex backend)
VERTEX_RAG_LOCATION=us-west1
VERTEX_RAG_CORPUS=projects/.../locations/.../ragCorpora/...

# ChromaDB (if using chroma backend)
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_NAME=dermatology_guidelines
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

### Set Up Vertex AI RAG (One-Time)

```bash
# Create RAG corpus and import documents
python scripts/setup_vertex_rag.py

# This creates a Vertex AI RAG corpus and returns the corpus ID
# (e.g., projects/123456/locations/us-west1/ragCorpora/789...)
# Update VERTEX_RAG_CORPUS in .env with this ID

# Add additional documents later
python scripts/add_to_vertex_rag.py --path gs://your-bucket/folder/
```

---

## Usage

### Gradio UI

```bash
# Launch interactive web interface
bash bin/run_gradio_app.sh

# Or directly
python main.py --mode app
```

Open http://localhost:7860 in your browser.

### Evaluation

```bash
# Quick test (1 case, single model)
python scripts/evaluate_nejim_cases.py \
  --input NEJIM/image_challenge_input \
  --agent-model medgemma \
  --max-cases 1

# HuggingFace models (500 evals, requires GPU)
# Interactive:
bash bin/run_evaluation_hf.sh
# Batch job (recommended — survives disconnection):
sbatch bin/run_evaluation_hf.sh

# Vertex AI model (250 evals, no GPU needed)
bash bin/run_evaluation_vertex.sh
```

### Add Clinical Guidelines

```bash
# Add to Vertex RAG (cloud)
python scripts/add_to_vertex_rag.py --path gs://your-bucket/folder/

# Add to ChromaDB (local)
python scripts/add_new_knowledge.py --path /path/to/pdfs/
```

### Ingest Documents to ChromaDB

```bash
# Ingest all documents from a directory
python main.py --mode ingest
```

---

## Project Structure

```
MedGemma/
├── src/
│   ├── agents/
│   │   ├── adk_agents.py              # Multi-agent ADK workflow (1200+ lines)
│   │   ├── conversation_manager.py    # Session logging and audit trail
│   │   ├── workflow_logger.py         # Structured agent decision logging
│   │   ├── registry.py               # Model registry (3 active + 2 stub models)
│   │   ├── factory.py                # Model adapter factory
│   │   └── models/
│   │       ├── base_model.py          # BaseLLM abstract class
│   │       ├── medgemma_adapter.py    # MedGemma-27B/4B-IT (HF, local GPU, multi-GPU)
│   │       ├── vertex_medgemma_adapter.py  # MedGemma-1.5-4B-IT (Vertex AI)
│   │       ├── gemini_adapter.py      # Gemini Pro (orchestration only)
│   │       ├── openai_adapter.py      # GPT-4 (stub)
│   │       └── claude_adapter.py      # Claude (stub)
│   ├── rag/
│   │   ├── vertex_rag_retriever.py    # Vertex AI RAG interface (cloud)
│   │   ├── retriever.py              # ChromaDB retriever (local)
│   │   ├── vector_store.py           # ChromaDB collection management
│   │   ├── embeddings.py            # Sentence-transformers embeddings
│   │   ├── ingestion.py             # Document ingestion pipeline
│   │   ├── chunking.py              # Text chunking (512 chars, 50 overlap)
│   │   └── document_processor.py    # Multi-format loader (PDF, TXT, JSON, DOCX)
│   ├── ui/
│   │   └── app.py                    # Gradio 6.x multimodal interface
│   ├── utils/
│   │   ├── schemas.py               # Pydantic data models (ClinicalCase, SOAPNote, etc.)
│   │   ├── logger.py                # PII-safe structured logging
│   │   └── explainability.py        # Citation and reasoning extraction
│   └── evaluation/
│       └── evaluator.py             # Robustness metrics and assessment
├── scripts/
│   ├── evaluate_nejim_cases.py       # Evaluation engine (750 evals, resume support)
│   ├── setup_vertex_rag.py          # One-time Vertex RAG corpus creation
│   ├── add_to_vertex_rag.py         # Add documents to Vertex RAG
│   ├── add_new_knowledge.py         # Add documents to ChromaDB
│   └── load_nejim_cases.py          # Case loading utility
├── bin/
│   ├── run_evaluation_hf.sh         # HF eval (500, 2x A40 GPU, sbatch)
│   ├── run_evaluation_vertex.sh     # Vertex eval (250, no GPU)
│   ├── run_gradio_app.sh            # Launch Gradio UI
│   └── test_3_models.sh             # Quick 3-model parallel test
├── tests/
│   ├── test_adk_workflow.py         # Multi-agent workflow tests
│   ├── test_config.py               # Configuration validation
│   ├── test_gradio_app.py           # UI component tests
│   ├── test_retrieval.py            # RAG retrieval tests
│   ├── test_schemas.py              # Data model validation
│   └── test_torch_gpu.py            # GPU availability checks
├── config/
│   └── config.py                    # Centralized settings (pydantic-settings)
├── data/
│   ├── chroma/                      # ChromaDB persistent storage (2,668 chunks)
│   └── cases/                       # Test case data
├── NEJIM/                           # 25 NEJM dermatology test cases (5 variants each)
├── SourceKnowledgeBase/             # AAD + StatPearls + JAADCR PDFs (62 files)
├── logs/                            # Session logs and evaluation results
├── main.py                          # CLI entry point (app/ingest/evaluate/test)
├── requirements.txt                 # Python dependencies (55 packages)
├── setup.py                         # Package installation
├── .env                             # API keys and configuration
└── .env.template                    # Configuration template
```

---

## Competition Criteria

This project addresses all Med-Gemma Impact Challenge judging criteria:

### Agentic Workflow (35%)
- 3-agent hierarchical system with explicit tool-call reasoning
- MedGemma performs ALL medical reasoning (Gemini is orchestrator only)
- Transparent model attribution at every step
- Automatic orchestrator quota rotation across 9 Gemini models

### Execution & Communication (30%)
- Professional Gradio UI with multimodal input
- Clear SOAP note output with differential diagnosis
- Follow-up question flow for incomplete cases
- Complete documentation and setup instructions

### Explainability (25%)
- Structured SOAP format with reasoning chain
- Citations from AAD, StatPearls, and JAADCR guidelines
- Agent thought process visible to user
- Full audit logs with PII filtering

### Robustness (10%)
- 750 evaluations across 5 context variants
- Agentic pause when data missing (safety-first)
- Cross-model comparison (MedGemma 27B vs 4B vs 1.5-4B-IT)
- With/without MCQ options analysis

---

## Model Performance

### Models Evaluated

| Model | Parameters | Provider | GPU Requirement | Precision |
|-------|------------|----------|-----------------|-----------|
| MedGemma-27B-IT | 27B | Hugging Face (local) | 2x A40 (92GB) | bfloat16 |
| MedGemma-4B-IT | 4B | Hugging Face (local) | 1x any GPU (16GB+) | bfloat16 |
| MedGemma-1.5-4B-IT | 4B | Vertex AI (cloud) | None (cloud endpoint) | Server-managed |

### Preliminary Results
- **Agentic Pause Rate**: 4.8% false positives on complete cases
- **Evaluation Speed**: ~13 min/case average (27B model)
- **Knowledge Base**: 2,668 chunks (ChromaDB) / 37 documents (Vertex RAG)
- **Retrieval**: Top-5 guideline citations per case

---

## Development

### Run Tests

```bash
pytest tests/

# Individual test files
python tests/test_adk_workflow.py
python tests/test_gradio_app.py
python tests/test_retrieval.py
```

### Code Quality

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

---

## Data Privacy

- All session logs are PII-safe (patient identifiers redacted)
- Clinical guidelines stored locally (not redistributed)
- Mock cases provided for reproducibility
- HIPAA-compliant logging practices

---

## License

This project is for educational and competition purposes. Clinical guidelines are sourced from publicly available AAD, StatPearls, and JAADCR resources.

---

## Acknowledgments

- **Google MedGemma Team** for the foundation clinical models (27B, 4B, 1.5-4B-IT)
- **Google ADK Team** for the Agent Development Kit framework
- **Google Vertex AI** for managed RAG corpus and model endpoint hosting
- **American Academy of Dermatology (AAD)** for clinical practice guidelines
- **StatPearls Publishing** for medical education reference articles
- **Journal of the American Academy of Dermatology Case Reports (JAADCR)** for dermatology case studies
- **NEJM** for inspiring the Image Challenge case format

---

**Status**: Vertex AI evaluation complete (250), HuggingFace evaluation in progress (500) — as of February 12, 2026
