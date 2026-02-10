# MedGemma Clinical Robustness Assistant

> A multi-agent clinical decision support system for dermatology, demonstrating diagnostic robustness under varying levels of clinical information.

**Competition**: [Med-Gemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
**Submission Deadline**: February 23, 2026

---

## Overview

This "Partner-Style" Clinical Assistant uses a hierarchical multi-agent workflow to provide safe, explainable dermatology consultations. The system demonstrates a key principle: **never guess when data is missing** - instead, it proactively requests clarification before making diagnostic decisions.

### Key Features

- **Multi-Agent Workflow**: 3 specialized agents (Triage, Research, Diagnostic) orchestrated via Google ADK
- **Safety-First**: Pauses and asks questions when clinical data is incomplete
- **Evidence-Based**: Grounds all reasoning in AAD/StatPearls dermatology guidelines via RAG
- **Transparent**: Generates structured SOAP notes with full reasoning and citations
- **Robust**: Evaluated across 750 scenarios with varying data completeness
- **Multi-Model**: Supports 3 MedGemma variants (27B, 4B, 1.5-4B-IT)

---

## Architecture

### Hybrid Two-Tier System

**TIER 1: Orchestration (Google ADK + Gemini Pro Latest)**
- Workflow management and task delegation
- Agent routing and coordination
- **Role**: Orchestrator only (no clinical reasoning)

**TIER 2: Clinical Reasoning (MedGemma Specialists)**
- **MedGemma-27B-IT**: Primary medical reasoning (Hugging Face, 27B params)
- **MedGemma-4B-IT**: Lightweight medical reasoning (Hugging Face, 4B params)
- **MedGemma-1.5-4B-IT**: Cloud medical reasoning (Vertex AI, 4B params)
- **Role**: ALL medical diagnosis and clinical analysis

### Multi-Agent Workflow

```
User Input (multimodal)
    ↓
┌──────────────────────────────────────┐
│   Root Coordinator (Gemini/ADK)     │  ← Orchestration only
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│   Triage Agent                       │
│   • Checks case completeness         │
│   • MedGemma identifies missing data │  ← Medical reasoning
│   • Asks clarifying questions        │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│   Research Agent                     │
│   • Retrieves clinical guidelines    │  ← RAG retrieval
│   • MedGemma synthesizes evidence    │  ← Medical reasoning
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│   Diagnostic Agent                   │
│   • MedGemma generates SOAP note     │  ← Medical reasoning
│   • Differential diagnosis           │
│   • Guideline citations              │
└──────────────────────────────────────┘
    ↓
Structured SOAP Output
```

---

## Technology Stack

### Core Technologies
- **Clinical AI**: MedGemma-27B-IT, 4B-IT, 1.5-4B-IT (Google)
- **Orchestration**: Google Agent Development Kit (ADK) v1.23+
- **Workflow Model**: Gemini Pro Latest (orchestration only)
- **RAG Backend**: Vertex AI RAG (cloud, 37 docs) or ChromaDB (local, 2,668 chunks)
- **UI Framework**: Gradio 6.x
- **Embeddings**: text-embedding-005 (Vertex) or all-MiniLM-L6-v2 (ChromaDB)

### Knowledge Base
- **Sources**: American Academy of Dermatology (AAD) Guidelines, StatPearls, JAAD Case Reports
- **Size**: 37 documents (Vertex RAG) / 2,668 chunks (ChromaDB)
- **Domains**: Dermatology, dermato-pathology, clinical guidelines

---

## Features

### Multimodal Clinical Input
- Dermatology lesion images (JPEG/PNG)
- Clinical case vignettes (text/PDF)
- Free-text patient history
- Structured exam findings

### Intelligent Triage
- Detects missing history, exam, or image data
- Asks specific follow-up questions
- Never guesses with incomplete information
- Safety-first medical reasoning

### Evidence-Based Retrieval (RAG)
- Queries 2,668+ chunks of clinical guidelines
- Returns top-5 relevant passages with similarity scores
- Dual backend: Vertex AI RAG (cloud) or ChromaDB (local)
- Citations included in every SOAP note

### Structured SOAP Output
```
S (Subjective): Patient-reported symptoms and history
O (Objective): Physical exam findings and diagnostic data
A (Assessment): Differential diagnosis with evidence
P (Plan): Recommended workup and treatment
```

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
- **3 models** × 2 formats × 25 cases × 5 variants = **750 evaluations**

### Metrics
1. **Pause Rate**: % of cases where agent asks for more data
2. **Diagnostic Accuracy**: Matches correct diagnosis
3. **Robustness**: Performance delta across context states
4. **Guideline Retrieval**: Relevance of cited sources
5. **Response Quality**: SOAP note completeness
6. **False Positive Rate**: Incorrect pauses on complete cases
7. **Execution Time**: Average time per case
8. **Error Rate**: System failures

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (for local MedGemma inference)
- 64GB+ RAM recommended
- Google Cloud account (for Vertex AI features)

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
```

### Configure API Keys

Create `.env` file:
```bash
# Required
HUGGINGFACE_API_KEY=your_hf_key_here
GOOGLE_API_KEY=your_google_key_here
GOOGLE_CLOUD_PROJECT=your_project_id_here

# Optional (for Vertex AI)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# RAG Backend
RAG_BACKEND=vertex  # or "chroma" for local
VERTEX_RAG_LOCATION=us-east1
VERTEX_RAG_CORPUS=projects/.../ragCorpora/...
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

# Full evaluation (750 cases, 48-72 hours)
bash bin/run_all_evaluation.sh
```

### Add Clinical Guidelines

```bash
# Add to Vertex RAG
python scripts/add_to_vertex_rag.py --path gs://your-bucket/folder/

# Add to ChromaDB
python scripts/add_new_knowledge.py --path /path/to/pdfs/
```

---

## Project Structure

```
MedGemma/
├── src/
│   ├── agents/               # Multi-agent implementation
│   │   ├── adk_agents.py    # ADK workflow (1200 lines)
│   │   ├── models/          # Model adapters
│   │   └── registry.py      # Model registry
│   ├── rag/                 # RAG pipeline
│   │   ├── vertex_rag_retriever.py  # Vertex AI RAG
│   │   ├── retriever.py             # ChromaDB
│   │   └── ingestion.py             # Document processing
│   ├── ui/                  # Gradio interface
│   │   └── app.py          # UI (778 lines)
│   └── utils/               # Utilities
│       └── schemas.py       # Data models
├── scripts/
│   ├── evaluate_nejim_cases.py      # Evaluation engine
│   ├── add_to_vertex_rag.py         # Vertex RAG ingestion
│   └── add_new_knowledge.py         # ChromaDB ingestion
├── bin/                     # Shell scripts
│   ├── run_all_evaluation.sh        # Full 750-eval pipeline
│   ├── run_gradio_app.sh            # Launch UI
│   └── test_3_models.sh             # Quick 3-model test
├── data/
│   ├── chroma/              # ChromaDB persistence
│   └── cases/               # Evaluation cases
├── tests/                   # Test suite
├── config/                  # Configuration
├── logs/                    # Application logs
└── requirements.txt         # Python dependencies
```

---

## Competition Criteria

This project addresses all Med-Gemma Impact Challenge judging criteria:

### Agentic Workflow (35%)
- 3-agent hierarchical system
- Explicit reasoning steps with tool calls
- MedGemma performs all medical reasoning
- Transparent model attribution

### Execution & Communication (30%)
- Professional Gradio UI with "partner" personality
- Clear SOAP note output
- 3-minute demo video (in progress)
- Complete documentation

### Explainability (25%)
- Structured SOAP format
- Citations from clinical guidelines
- Agent thought process visible
- Full audit logs

### Robustness (10%)
- 750 evaluations across 5 context variants
- Agentic pause when data missing
- Cross-model comparison (3 MedGemma variants)

---

## Model Performance

### Models Evaluated

| Model | Parameters | Provider | Strengths |
|-------|------------|----------|-----------|
| MedGemma-27B-IT | 27B | Hugging Face | High accuracy, comprehensive reasoning |
| MedGemma-4B-IT | 4B | Hugging Face | Fast inference, lower memory |
| MedGemma-1.5-4B-IT | 4B | Vertex AI | Cloud-hosted, no local GPU needed |

### Preliminary Results
- **Agentic Pause Rate**: 4.8% false positives on complete cases
- **Evaluation Speed**: ~13 min/case average (27B model)
- **Knowledge Base**: 2,668 chunks, 37 documents
- **Retrieval Precision**: Top-5 guideline citations per case

---

## Development

### Run Tests

```bash
# Test ADK workflow
python tests/test_adk_workflow.py

# Test Gradio UI
python tests/test_gradio_app.py

# Test RAG retrieval
python tests/test_retrieval.py
```

### Code Quality

```bash
# Format
black src/ tests/

# Lint
flake8 src/ tests/

# Type check
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

This project is for educational and competition purposes. Clinical guidelines are sourced from publicly available AAD and StatPearls resources.

---

## Acknowledgments

- **Google Med-Gemma Team** for the foundation models
- **Google ADK Team** for the agent framework
- **American Academy of Dermatology (AAD)** for clinical guidelines
- **StatPearls Publishing** for medical education resources
- **NEJM** for inspiring the case format

---

## Citation

If you use this work, please cite:

```bibtex
@misc{medgemma2026,
  title={MedGemma Clinical Robustness Assistant: A Multi-Agent System for Safe Dermatology Consultations},
  author={[Your Name]},
  year={2026},
  publisher={Kaggle Med-Gemma Impact Challenge},
  url={https://github.com/yourusername/MedGemma}
}
```

---

## Contact

For questions or collaboration:
- **Competition**: [Kaggle Discussion](https://www.kaggle.com/competitions/med-gemma-impact-challenge/discussion)
- **Issues**: [GitHub Issues](https://github.com/yourusername/MedGemma/issues)

---

**Status**: Implementation complete, testing in progress (as of Feb 9, 2026)
