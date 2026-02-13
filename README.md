# MedGemma Clinical Robustness Assistant

> A production-grade multi-agent clinical decision support system for dermatology, demonstrating diagnostic robustness under varying levels of clinical information.

**Competition**: [Med-Gemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
**Submission Deadline**: February 23, 2026
**Status**: ✅ Production deployment on Google Cloud Vertex AI

---

## Overview

This "Partner-Style" Clinical Assistant uses a hierarchical multi-agent workflow to provide safe, explainable dermatology consultations. The system demonstrates a key principle: **never guess when data is missing** — instead, it proactively requests clarification before making diagnostic decisions.

### Key Features

- **Cloud-Native Production Deployment**: All MedGemma models deployed on Vertex AI endpoints
- **Multi-Agent Workflow**: 3 specialized agents (Triage, Research, Diagnostic) orchestrated via Google ADK
- **Safety-First**: Pauses and asks questions when clinical data is incomplete
- **Evidence-Based**: Grounds all reasoning in AAD/StatPearls/JAADCR dermatology guidelines via Vertex AI RAG
- **Transparent**: Generates structured SOAP notes with full reasoning and citations
- **Robust**: Evaluated across **1,250 scenarios** with varying data completeness
- **Multi-Model**: 2 MedGemma variants deployed (27B-IT, 4B-IT) via Vertex AI
- **Resilient Orchestration**: 9 Gemini model fallbacks for automatic API quota rotation
- **Scalable**: No local GPU dependency, cloud-hosted inference

---

## Architecture

### Production Cloud-Native Deployment

**TIER 1: Orchestration (Google ADK + Gemini Pro)**
- Workflow management and task delegation
- Agent routing and coordination
- **Role**: Orchestrator only — performs NO clinical reasoning
- **Quota Resilience**: 9 Gemini models (Pro and Flash variants) configured as fallbacks. When one model's daily API quota is exhausted (~100 requests/day), the system automatically switches to the next model. This enables ~900 orchestrator requests/day without manual intervention.

**TIER 2: Clinical Reasoning (MedGemma Vertex AI Endpoints)**
- **MedGemma-27B-IT**: Primary medical reasoning (Vertex AI, 27B params, multimodal: image+text)
- **MedGemma-4B-IT**: Lightweight medical reasoning (Vertex AI, 4B params, multimodal: image+text)
- **Role**: ALL medical diagnosis and clinical analysis
- **Deployment**: Cloud-hosted endpoints, auto-scaling, no local GPU needed

### Multi-Agent Workflow

```
User Input (multimodal: text + images)
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
|   - MedGemma-27B-IT identifies       |  <-- Medical reasoning (Vertex)
|     missing data                     |
|   - Asks clarifying questions        |
+--------------------------------------+
    |
+--------------------------------------+
|   Research Agent                     |
|   - Retrieves clinical guidelines    |  <-- Vertex AI RAG
|   - MedGemma-27B-IT synthesizes      |  <-- Medical reasoning (Vertex)
|     evidence                         |
+--------------------------------------+
    |
+--------------------------------------+
|   Diagnostic Agent                   |
|   - MedGemma-27B-IT generates        |  <-- Medical reasoning (Vertex)
|     SOAP note                        |
|   - Differential diagnosis           |
|   - Guideline citations              |
+--------------------------------------+
    |
Structured SOAP Output
```

---

## Technology Stack

### Google Cloud Platform Stack

| Service | Purpose |
|---------|---------|
| **Vertex AI Endpoints** | MedGemma-27B-IT and 4B-IT model hosting |
| **Vertex AI RAG** | Clinical guideline retrieval (37 documents) |
| **Gemini API** | Multi-agent orchestration |
| **Google ADK** | Agent Development Kit for workflow coordination |

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `google-adk` | 1.23.0 | Agent Development Kit — multi-agent orchestration |
| `google-genai` | 1.61.0 | Gemini API — orchestrator model |
| `google-cloud-aiplatform` | 1.135.0 | Vertex AI — RAG corpus and MedGemma endpoints |
| `gradio` | 6.3.0 | Web UI framework — multimodal clinical interface |
| `pydantic` | 2.11.10 | Data models — ClinicalCase, SOAPNote, etc. |
| `pypdf` | 6.6.2 | PDF parsing — clinical guideline ingestion |

### Knowledge Base (Vertex AI RAG)

| Source | Documents | Description |
|--------|-----------|-------------|
| AAD Guidelines | 6 | American Academy of Dermatology clinical practice guidelines |
| StatPearls | 5 | Evidence-based medical education reference articles |
| JAADCR | 21 | Journal of the American Academy of Dermatology Case Reports |
| **Total** | **37** | **Hosted on Vertex AI RAG (us-west1)** |

**RAG Backend**: Vertex AI RAG (cloud-native, server-side embeddings, text-embedding-005)

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

### Vertex AI RAG Pipeline
- Cloud-hosted RAG corpus (37 dermatology documents)
- Server-side embeddings (text-embedding-005)
- Top-5 relevant passages with similarity scores per query
- Citations included in every SOAP note
- Lowered similarity threshold (0.25) for better recall

### Structured SOAP Output
```
S (Subjective): Patient-reported symptoms and history
O (Objective): Physical exam findings and diagnostic data
A (Assessment): Differential diagnosis with evidence
P (Plan): Recommended workup and treatment
```

### Orchestrator Quota Resilience
The Gemini API has a daily quota of ~100 requests per model. For large-scale evaluations (1,250+ cases), the system configures 9 Gemini models as fallbacks:
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

### Test Datasets

#### **Dataset 1: NEJM Image Challenge Cases (Gold Standard)**
- **25 dermatology cases** (challenging, diverse, prestigious source)
- **5 context variants** per case:
  - Original (complete)
  - History only
  - Image only
  - Exam only
  - Exam restricted (minimal findings)
- **2 data formats**: With/without MCQ options
- **2 models** x 2 formats x 25 cases x 5 variants = **500 evaluations**
- **Purpose**: Tests robustness on challenging out-of-domain cases

#### **Dataset 2: JAADCR Case Reports (Domain-Matched)**
- **25 dermatology cases** (from JAAD Case Reports, NOT in RAG corpus)
- **5 context variants** per case (same as NEJM)
- **2 data formats**: With/without MCQ options
- **2 models** x 2 formats x 25 cases x 5 variants = **500 evaluations**
- **Purpose**: Tests optimal performance with domain-matched data

**Total Evaluations**: 500 (NEJM) + 500 (JAADCR) + 250 (baseline MedGemma-1.5-4B) = **1,250 evaluations**

### Evaluation Matrix

```
NEJM Image Challenge Cases (500 evaluations):
  - MedGemma-27B-IT (Vertex): 250 evals (125 without_options + 125 with_options)
  - MedGemma-4B-IT (Vertex): 250 evals (125 without_options + 125 with_options)

JAADCR Case Reports (500 evaluations):
  - MedGemma-27B-IT (Vertex): 250 evals (125 without_options + 125 with_options)
  - MedGemma-4B-IT (Vertex): 250 evals (125 without_options + 125 with_options)

Baseline (250 evaluations - completed Feb 11):
  - MedGemma-1.5-4B-IT (Vertex): 250 evals (125 without_options + 125 with_options)

TOTAL: 1,250 evaluations across 3 models and 2 datasets
```

### Metrics
1. **Pause Rate**: % of cases where agent asks for more data
2. **Diagnostic Accuracy**: Matches correct diagnosis
3. **Robustness**: Performance delta across context states
4. **Guideline Retrieval**: Relevance of cited sources (RAG similarity scores)
5. **Response Quality**: SOAP note completeness
6. **False Positive Rate**: Incorrect pauses on complete cases
7. **Execution Time**: Average time per case
8. **Error Rate**: System failures

### Key Findings

**NEJM Cases (Challenging)**:
- Lower RAG similarity scores (0.3-0.4) due to domain mismatch
- System still robust, works without optimal guideline matches
- Demonstrates safety (pauses when data incomplete)

**JAADCR Cases (Domain-Matched)**:
- Higher RAG similarity scores (0.5-0.8) with domain-matched corpus
- Better guideline citations in SOAP notes
- Demonstrates optimal performance with relevant knowledge base

---

## Installation

### Prerequisites
- Python 3.10+
- Google Cloud account with Vertex AI enabled
- Google Cloud service account with permissions:
  - Vertex AI User
  - Vertex AI RAG Administrator
  - Gemini API access
- **No local GPU required** (all models deployed on Vertex AI)

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
GOOGLE_API_KEY=your_google_key_here         # For Gemini orchestrator
GOOGLE_CLOUD_PROJECT=your_project_id_here   # For Vertex AI services
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# RAG Backend
RAG_BACKEND=vertex   # Cloud-based Vertex AI RAG (required)

# Vertex AI RAG Configuration
VERTEX_RAG_LOCATION=us-west1
VERTEX_RAG_CORPUS=projects/.../locations/.../ragCorpora/...
```

### Deploy MedGemma Models on Vertex AI

```bash
# Deploy MedGemma-27B-IT to Vertex AI
# 1. Go to Vertex AI Model Garden in Google Cloud Console
# 2. Search for "MedGemma-27B-IT"
# 3. Click "Deploy" (one-click deployment)
# 4. Copy endpoint ID to .env or registry.py

# Deploy MedGemma-4B-IT to Vertex AI
# (Same process as above for 4B model)

# Update src/agents/registry.py with endpoint IDs
```

### Set Up Vertex AI RAG

```bash
# Create RAG corpus and import documents
python scripts/setup_vertex_rag.py

# Add additional documents
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
# Quick test (1 case, MedGemma-27B-IT)
python scripts/evaluate_nejim_cases.py \
  --input NEJIM/image_challenge_input \
  --agent-model medgemma-27b-it-vertex \
  --max-cases 1

# Full NEJM evaluation (500 evals, Vertex models)
bash bin/run_evaluation_nejm.sh

# Full JAADCR evaluation (500 evals, Vertex models)
bash bin/run_evaluation_jaadcr.sh

# All evaluations combined
bash bin/run_all_evaluations.sh  # 1,250 total
```

### Results Location

```
logs/
├── evaluation_medgemma-27b-it-vertex_nejm_without_options/
├── evaluation_medgemma-27b-it-vertex_nejm_with_options/
├── evaluation_medgemma-4b-it-vertex_nejm_without_options/
├── evaluation_medgemma-4b-it-vertex_nejm_with_options/
├── evaluation_medgemma-27b-it-vertex_jaadcr_without_options/
├── evaluation_medgemma-4b-it-vertex_jaadcr_without_options/
└── sessions/  # Detailed session logs for each case
```

---

## Project Structure

```
MedGemma/
├── src/
│   ├── agents/
│   │   ├── adk_agents.py              # Multi-agent orchestration (1200+ lines)
│   │   ├── conversation_manager.py    # Session logging and audit trails
│   │   ├── models/
│   │   │   ├── vertex_medgemma_adapter.py  # Vertex AI MedGemma adapter
│   │   │   ├── gemini_adapter.py           # Gemini orchestrator adapter
│   │   │   └── base_model.py               # Base LLM interface
│   │   └── registry.py                # Model registry (Vertex endpoints)
│   ├── rag/
│   │   └── vertex_rag_retriever.py    # Vertex AI RAG retrieval
│   ├── ui/
│   │   └── app.py                     # Gradio interface (778 lines)
│   └── utils/
│       └── schemas.py                 # Pydantic data models
├── scripts/
│   ├── evaluate_nejim_cases.py        # Main evaluation engine (621 lines)
│   ├── add_to_vertex_rag.py           # Add documents to Vertex RAG
│   └── setup_vertex_rag.py            # Initial Vertex RAG setup
├── bin/
│   ├── run_evaluation_nejm.sh         # NEJM 500 evaluations
│   ├── run_evaluation_jaadcr.sh       # JAADCR 500 evaluations
│   ├── run_all_evaluations.sh         # All 1,250 evaluations
│   └── run_gradio_app.sh              # Launch UI
├── config/
│   └── config.py                      # Pydantic settings
├── NEJM/                              # NEJM test cases (25 cases × 5 variants)
├── JAADCR/                            # JAADCR test cases (25 cases × 5 variants)
├── logs/                              # Evaluation results and session logs
├── .env                               # API keys and configuration
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── TODO.md                            # Project status and roadmap
```

---

## Competition Alignment

### Explainability (25%)
✅ SOAP format shows complete reasoning chain
✅ Citations from clinical guidelines via Vertex AI RAG
✅ Agent thought process visible in session logs
✅ Full model attribution (Gemini orchestrator + MedGemma specialist)

### Robustness (25%)
✅ 5 context variants demonstrate handling of incomplete data
✅ Tested on 2 datasets (NEJM challenging + JAADCR domain-matched)
✅ 1,250 evaluation runs across multiple conditions
✅ Agentic pause when data missing (safety-first approach)

### Safety (25%)
✅ Never guesses with incomplete data
✅ Proactively asks clarifying questions
✅ Grounded in evidence-based AAD/StatPearls/JAADCR guidelines

### Technical Implementation (15%)
✅ Full Google Cloud stack: Vertex AI + Gemini + ADK + Vertex RAG
✅ Production-grade deployment (cloud-hosted, scalable)
✅ Clean, well-documented code
✅ Comprehensive evaluation infrastructure

### Execution & Communication (10%)
✅ Professional Gradio UI
✅ Complete documentation (README, TODO, session logs)
✅ 3-minute demo video (in progress)

---

## Why Cloud Deployment (Vertex AI)?

### Advantages over Local GPU Inference

| Aspect | Local GPU (HuggingFace) | Cloud Deployment (Vertex AI) |
|--------|------------------------|------------------------------|
| **Setup Time** | 5-10 min model download | Instant (pre-deployed) ✅ |
| **GPU Requirements** | 24-92GB VRAM needed | None (cloud-hosted) ✅ |
| **Scalability** | Limited by local GPUs | Unlimited (auto-scaling) ✅ |
| **Cost** | GPU time + electricity | Pay-per-request ✅ |
| **Production Ready** | Requires DevOps | Production-grade ✅ |
| **Multi-User** | Single user | Concurrent users ✅ |
| **Latency** | <1s | 1-3s (acceptable) |
| **Maintenance** | Manual updates | Managed by Google ✅ |

**Verdict**: Vertex AI deployment is superior for production systems and demonstrates enterprise-grade architecture to competition judges.

---

## License

Apache 2.0

---

## Citation

If you use this system in your research, please cite:

```bibtex
@software{medgemma_clinical_assistant_2026,
  title={MedGemma Clinical Robustness Assistant},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/MedGemma},
  note={Med-Gemma Impact Challenge Submission}
}
```

---

**Last Updated**: February 12, 2026
**Competition**: Med-Gemma Impact Challenge
**Submission Deadline**: February 23, 2026
