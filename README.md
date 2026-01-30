# MedGemma Clinical Robustness Assistant

A multi-agent clinical decision support system for dermatology that evaluates diagnostic robustness under varying levels of clinical information. Built for the [Med-Gemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge).

## Overview

This "Partner-Style" Clinical Assistant uses a hierarchical multi-agent workflow to:
1. Identify missing clinical context (History, Physical Exam, Images)
2. Proactively request missing data before making diagnostic decisions
3. Ground reasoning in evidence-based dermatology guidelines via RAG (Retrieval-Augmented Generation)
4. Generate structured SOAP notes with transparent reasoning

## Architecture

### Multi-Agent System
- **Triage Agent**: Analyzes input for missing clinical data and triggers clarification requests
- **Research Agent**: Queries ChromaDB vector store for relevant AAD/StatPearls guidelines
- **Diagnostic Agent**: Synthesizes information into structured SOAP notes with differential diagnoses

### Technology Stack
- **Foundation Models**: MedGemma-27B (Hugging Face) and Gemini Pro (Google)
- **Agent Framework**: Google Agent Development Kit (ADK)
- **Vector Database**: ChromaDB with sentence-transformers embeddings
- **UI**: Gradio (multimodal interface for images, text, and chat)

## Project Structure

```
MedGemma/
├── config/                  # Configuration management
│   ├── config.py           # Settings loader
│   └── __init__.py
├── src/
│   ├── agents/             # Multi-agent implementation
│   │   ├── triage_agent.py
│   │   ├── research_agent.py
│   │   └── diagnostic_agent.py
│   ├── rag/                # RAG pipeline
│   │   ├── ingestion.py    # Document chunking & embedding
│   │   ├── vector_store.py # ChromaDB interface
│   │   └── retriever.py    # Semantic search
│   ├── ui/                 # Gradio interface
│   │   └── app.py
│   └── utils/              # Shared utilities
│       ├── schemas.py      # Pydantic data models
│       └── logger.py       # Structured logging
├── data/
│   ├── chroma/             # ChromaDB persistence (gitignored)
│   ├── cases/              # Evaluation cases
│   └── guidelines/         # AAD/StatPearls documents
├── tests/                  # Unit & integration tests
├── local_cache/            # Scraped data (gitignored)
├── logs/                   # Application logs (gitignored)
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── .env.template           # Environment variables template
└── README.md
```

## Setup Instructions

### 1. Clone and Install

```bash
cd "D:\Halima's Data\more\LLM_Udemy\MedGemma"
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the environment template and add your API keys:

```bash
copy .env.template .env
```

Edit `.env` and add:
- `HUGGINGFACE_API_KEY`: Your Hugging Face API key (for MedGemma-27B)
- `GEMINI_API_KEY`: Your Google Gemini API key
- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud project ID (for ADK)

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
- American Academy of Dermatology (AAD)
- StatPearls Publishing
