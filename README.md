# MedGemma Clinical Robustness Assistant

A multi-agent clinical decision support system for dermatology that evaluates diagnostic robustness under varying levels of clinical information. Built for the [Med-Gemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge).

## Overview

This "Partner-Style" Clinical Assistant uses a hierarchical multi-agent workflow to:
1. Identify missing clinical context (History, Physical Exam, Images)
2. Proactively request missing data before making diagnostic decisions
3. Ground reasoning in evidence-based dermatology guidelines via RAG (Retrieval-Augmented Generation)
4. Generate structured SOAP notes with transparent reasoning and full audit trails

## Architecture

### Hybrid Two-Tier System


**TIER 1: Clinical Reasoning (MedGemma-27B-IT Specialist)**
- **Model**: MedGemma-27B-IT via Hugging Face Transformers
- **Role**: Medical diagnosis, clinical analysis, SOAP generation
- **Responsibility**: ALL high-stakes medical reasoning
- **Invoked via**: FunctionTools (`medgemma_triage_analysis`, `medgemma_guideline_synthesis`, `medgemma_clinical_diagnosis`)


**TIER 2: Orchestration (Google ADK + Gemini Pro Latest)**
- **Framework**: Google Agent Development Kit (ADK) v1.23.0
- **Model**: Gemini Pro Latest
- **Role**: Workflow management, agent routing, tool coordination
- **Responsibility**: Delegates tasks but does NOT perform clinical reasoning


### Why This Architecture?

1. **Competition Requirement**: MedGemma-27B-IT performs all clinical reasoning (mandatory)
2. **Best of Both Worlds**: Gemini's advanced orchestration + MedGemma's medical expertise
3. **Production-Grade**: Separates workflow management from domain-specific expertise
4. **Full Transparency**: Every step logged with explicit model attribution

### Multi-Agent Workflow

**Root Coordinator (Gemini Pro Latest)**
- Manages overall workflow
- Delegates to three specialized agents

**Triage Agent**
- Step 1: Gemini orchestrates case completeness check
- Step 2: MedGemma analyzes clinical data and identifies missing information
- Step 3: Gemini summarizes and delegates to Research Agent

**Research Agent**
- Step 1: Gemini retrieves clinical guidelines from ChromaDB (RAG)
- Step 2: Gemini may refine search query (autonomous retrieval)
- Step 3: MedGemma synthesizes guidelines with case data
- Step 4: Gemini delegates to Diagnostic Agent

**Diagnostic Agent**
- Step 1: MedGemma generates complete SOAP note with differential diagnosis
- Step 2: Gemini assembles final output

### Technology Stack
- **Clinical Reasoning**: MedGemma-27B-IT (google/medgemma-27b-it via Hugging Face)
- **Workflow Orchestration**: Gemini Pro Latest (gemini-pro-latest via Google GenAI)
- **Agent Framework**: Google Agent Development Kit (ADK) v1.23.0
- **Vector Database**: ChromaDB v1.4.1 with sentence-transformers embeddings
- **UI**: Gradio v6.3.0 (multimodal interface)
- **Embeddings**: all-MiniLM-L6-v2 (384-dim)

## Audit-Grade Logging

### Research-Grade Session Logs

Every workflow execution generates comprehensive JSON logs in `logs/sessions/` with:

**Dual-Model Tracking**
```json
"models": {
  "orchestrator": {
    "name": "gemini-pro-latest",
    "provider": "google_genai",
    "role": "workflow_coordination"
  },
  "specialist": {
    "name": "google/medgemma-27b-it",
    "provider": "huggingface_transformers",
    "role": "clinical_reasoning"
  }
}
```

**Input Causality**
```json
"input": {
  "source_type": "step_5_ResearchAgent_output",
  "reference": {
    "step_id": 5,
    "agent": "ResearchAgent",
    "data_flow": "sequential"
  }
}
```

**Execution Details**
```json
"execution": {
  "orchestrator_action": "specialist_invocation",
  "operation_type": "specialist_guideline_synthesis",
  "tools_called": ["medgemma_guideline_synthesis"]
}
```

**Output Attribution**
```json
"output": {
  "type": "guideline_synthesis",
  "content": "...",
  "produced_by": "specialist"
}
```

**Step Metadata**
```json
"step_metadata": {
  "step_role": "guideline_synthesis",
  "step_phase": "specialist_reasoning",
  "is_final_resolution": false
}
```

**Trust Verification**
```json
"trust_metadata": {
  "clinical_reasoning_by_specialist": true,
  "specialist_model": "google/medgemma-27b-it",
  "orchestrator_clinical_role": "none"
}
```

### Step Phases

Every step is classified into one of three phases:

- **`specialist_reasoning`**: MedGemma-27B-IT performing clinical analysis
- **`rag_retrieval`**: ChromaDB guideline retrieval
- **`orchestration`**: Gemini coordinating workflow

### Output Types

Clear semantic types for every output:

- `triage_analysis` - MedGemma case completeness assessment
- `guideline_synthesis` - MedGemma guideline interpretation
- `diagnostic_reasoning` - MedGemma SOAP note generation
- `rag_results` - Retrieved clinical guidelines
- `completeness_report` - Case data completeness check
- `delegation_notice` - Agent transfer
- `coordination_message` - Workflow coordination

### Verifying MedGemma Usage

To verify MedGemma performed clinical reasoning, check logs for steps with:
```json
{
  "step_phase": "specialist_reasoning",
  "specialist_model": "google/medgemma-27b-it",
  "clinical_reasoning_by_specialist": true,
  "output": {
    "type": "triage_analysis" | "guideline_synthesis" | "diagnostic_reasoning",
    "produced_by": "specialist"
  }
}
```

### Example Workflow Log

A typical case generates 8-10 steps:

1. **Step 1**: RootCoordinator delegates (`orchestration`)
2. **Step 2**: TriageAgent checks completeness (`orchestration`)
3. **Step 3**: TriageAgent invokes MedGemma (`specialist_reasoning`) ✓
4. **Step 4**: TriageAgent transfers to Research (`orchestration`)
5. **Step 5**: ResearchAgent retrieves guidelines (`rag_retrieval`)
6. **Step 6**: ResearchAgent refines search (`rag_retrieval`)
7. **Step 7**: ResearchAgent invokes MedGemma (`specialist_reasoning`) ✓
8. **Step 8**: ResearchAgent transfers to Diagnostic (`orchestration`)
9. **Step 9**: DiagnosticAgent invokes MedGemma (`specialist_reasoning`) ✓
10. **Step 10**: DiagnosticAgent outputs final SOAP (`orchestration`)

**Result**: 3 specialist reasoning steps (MedGemma) + 2 RAG steps + 5 orchestration steps

## Project Structure

```
MedGemma/
├── config/                        # Configuration management
│   ├── config.py                 # Settings loader with environment validation
│   └── __init__.py
├── src/
│   ├── agents/                   # Multi-agent implementation (Google ADK)
│   │   ├── adk_agents.py        # Hybrid Gemini + MedGemma agents
│   │   ├── conversation_manager.py  # Audit-grade session logging
│   │   └── models/              # Model adapters
│   │       ├── medgemma_adapter.py  # MedGemma-27B-IT interface
│   │       └── gemini_adapter.py    # Gemini Pro Latest interface
│   ├── rag/                      # RAG pipeline
│   │   ├── ingestion.py         # Document chunking & embedding
│   │   ├── vector_store.py      # ChromaDB interface
│   │   └── retriever.py         # Semantic search with similarity scoring
│   ├── ui/                       # Gradio interface
│   │   └── app.py               # Multimodal UI with agent thought trace
│   └── utils/                    # Shared utilities
│       ├── schemas.py           # Pydantic data models (ClinicalCase, SOAPNote)
│       └── logger.py            # Structured logging
├── data/
│   ├── chroma/                   # ChromaDB persistence (2,184 chunks)
│   └── cases/                    # Evaluation cases (NEJM-style vignettes)
├── tests/                        # Test suite
│   ├── test_adk_workflow.py     # Multi-agent workflow test
│   └── test_gradio_app.py       # UI integration test
├── bin/                          # Shell scripts
│   └── run_test_adk.sh          # GPU session manager for HPC
├── logs/                         # Application logs
│   └── sessions/                # JSON/TXT/MD session logs
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies (pinned versions)
├── .env.template                 # Environment variables template
└── README.md                     # This file
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for local MedGemma inference)
- 64GB+ RAM for optimal performance

### 1. Clone and Install

```bash
cd "My path\MedGemma"
pip install -r requirements.txt
```

**Note**: This project uses the latest stable versions:
- Google ADK 1.23.0 (improved telemetry and async support)
- ChromaDB 1.x (better performance and vector search capabilities)
- All dependencies are pinned for reproducibility

### 2. Configure API Keys

Copy the environment template and add My API keys:

```bash
copy .env.template .env
```

Edit `.env` and add:
- `HUGGINGFACE_API_KEY`: My Hugging Face API key (for MedGemma-27B-IT clinical reasoning)
- `GEMINI_API_KEY`: My Google Gemini API key (for Gemini Pro Latest orchestration)
- `GOOGLE_CLOUD_PROJECT`: My Google Cloud project ID (for ADK v1.23.0)

### 3. Ingest Guidelines

Populate ChromaDB with dermatology guidelines:

```bash
python main.py --mode ingest
```

### 4. Launch Application

Start the Gradio UI:

```bash
python main.py --mode app
```

### 5. Run Evaluation

Evaluate on 25 test cases:

```bash
python main.py --mode evaluate
```

## Usage

### Gradio Interface

The UI supports:
- **Image Upload**: Dermatology lesion photos
- **Case File Upload**: Clinical vignettes (text/PDF)
- **Interactive Chat**: Agent asks follow-up questions for missing data
- **Thought Process View**: Transparent agent reasoning
- **SOAP Note Output**: Structured diagnostic assessment with guideline citations

### Example Workflow

1. User uploads an image of a skin lesion (no history provided)
2. Triage Agent detects missing history and asks: "How long has this lesion been present?"
3. User provides: "2 weeks, started as a small red bump"
4. Research Agent retrieves relevant AAD guidelines on acute dermatoses
5. Diagnostic Agent generates SOAP note with differential diagnosis

## Data Privacy

- All logs are PII-safe (patient identifiers redacted)
- Scraped guidelines stored in `local_cache/` (not committed to git)
- Mock cases provided for reproducibility

## Development

### Run Tests

```bash
python main.py --mode test
```

### Code Quality

```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

## Competition Criteria

This project addresses the Med-Gemma Impact Challenge judging criteria:

1. **Agentic Workflow (35%)**: 3-agent system with explicit reasoning steps
2. **Execution & Communication (30%)**: Polished Gradio UI with "partner" personality
3. **Explainability (25%)**: SOAP notes with guideline citations
4. **Robustness (10%)**: Evaluation across 4 context states (Original, History-Only, Image-Only, Exam-Restricted)

## License

This project is for educational and competition purposes. Medical guidelines are sourced from AAD and StatPearls.

## Acknowledgments

- Google Med-Gemma team
- Gemini
- Google ADK
- Google Gen AI
- American Academy of Dermatology (AAD)
- StatPearls Publishing
