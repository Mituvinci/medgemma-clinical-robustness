# MedGemma Clinical Robustness Assistant

A multi-agent clinical decision support system for dermatology that demonstrates diagnostic robustness under varying levels of clinical information. Built for the [Med-Gemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge).

The core principle: never guess when data is missing. The system pauses and asks clarifying questions instead.

---

## Architecture

Two-tier hybrid system:

**Tier 1 -- Orchestration (Google ADK + Gemini Pro)**: Manages workflow, delegates tasks, coordinates agents. Does not perform any clinical reasoning. 9 Gemini model fallbacks for API quota resilience (~900 requests/day).

**Tier 2 -- Clinical Reasoning (MedGemma on Vertex AI)**: All medical diagnosis performed by MedGemma models deployed as Vertex AI endpoints.

| Model | Parameters | Deployment |
|-------|-----------|------------|
| MedGemma-27B-IT | 27B (multimodal) | Vertex AI endpoint |
| MedGemma-4B-IT | 4B (multimodal) | Vertex AI endpoint |
| MedGemma-1.5-4B-IT | 4B (multimodal) | Vertex AI endpoint |

**Multi-Agent Workflow**:
1. Triage Agent -- checks case completeness, identifies missing data, asks clarifying questions
2. Research Agent -- retrieves clinical guidelines from Vertex AI RAG (55 documents: AAD, StatPearls, JAADCR)
3. Diagnostic Agent -- generates SOAP note with differential diagnoses and guideline citations

### Example: Multi-Agent Workflow in Action

**Complete case (diagnosis provided):**

> *Input:* "A 45-year-old woman presents with a 3-week history of an itchy, scaly, well-demarcated erythematous plaque on her right elbow. No new medications. Family history of autoimmune disease. On exam: silvery-white scale on an erythematous base, Auspitz sign positive."

| Agent | Action |
|-------|--------|
| **Triage (MedGemma)** | Missing Items: None. Case data is complete -- history, exam findings, demographics present. **Proceed.** |
| **Research (MedGemma + RAG)** | Query: "well-demarcated erythematous scaly plaque elbow Auspitz sign" -- 4 guidelines retrieved (AAD Psoriasis Guidelines, StatPearls Plaque Psoriasis). Synthesis: presentation consistent with plaque-type psoriasis given classic morphology and Auspitz sign. |
| **Diagnostic (MedGemma)** | **SOAP Note** -- Primary Dx: Plaque Psoriasis (Confidence: 0.90). Differentials: Nummular Dermatitis (0.40), Tinea Corporis (0.25). Plan: KOH prep to rule out fungal, initiate topical corticosteroid, dermatology referral. |

**Incomplete case (agentic pause):**

> *Input:* "A patient has a skin rash."

| Agent | Action |
|-------|--------|
| **Triage (MedGemma)** | Missing Items: Patient age, rash location, duration, morphology, associated symptoms. **Request Clarification.** Questions: "What is the patient's age?", "Where on the body is the rash?", "How long has it been present?" |

The workflow **pauses** -- no diagnosis attempted. This is the core safety behavior.

---

## Evaluation

1,500 evaluations across 3 models, 2 datasets, 5 context variants, and 2 data formats.

**Datasets**:
- NEJM Image Challenge: 25 challenging dermatology cases (out-of-domain)
- JAADCR Case Reports: 25 open access (CC BY-NC-ND 4.0) domain-matched cases with ground truth extracted via Gemini API. See [JAADCR_EVALUATION_CASES.md](JAADCR_EVALUATION_CASES.md) for full case list with source URLs.

**Context Variants** (per case): Original (complete), History only, Image only, Exam only, Exam restricted (vague)

**Key Results** (1,500 evaluations complete -- original variant, complete cases):

| Model | Dataset | Format | Top-1 | Top-3 | Top-4 | Orig Pause | Incomplete Pause |
|-------|---------|--------|-------|-------|-------|------------|------------------|
| MedGemma-4B-IT | NEJM | no options | **36%** | **44%** | **44%** | 20% | 92% |
| MedGemma-27B-IT | NEJM | no options | 32% | 32% | 36% | 16% | 90% |
| MedGemma-27B-IT | NEJM | with options | 28% | 28% | 28% | 16% | 83% |
| MedGemma-4B-IT | NEJM | with options | 16% | 20% | 24% | 12% | 94% |
| MedGemma-1.5-4B-IT | NEJM | no options | 12% | 16% | 16% | 24% | 92% |
| MedGemma-1.5-4B-IT | NEJM | with options | 8% | 8% | 8% | 12% | 99% |
| MedGemma-1.5-4B-IT | JAADCR | with options | **52%** | **52%** | **52%** | 20% | 92% |
| MedGemma-27B-IT | JAADCR | no options | 24% | 28% | 32% | 32% | 90% |
| MedGemma-27B-IT | JAADCR | with options | 32% | 32% | 32% | 16% | 88% |
| MedGemma-4B-IT | JAADCR | with options | 24% | 24% | 24% | 20% | 87% |
| MedGemma-1.5-4B-IT | JAADCR | no options | 20% | 20% | 20% | 16% | 90% |
| MedGemma-4B-IT | JAADCR | no options | 16% | 16% | 16% | 28% | 94% |

**Key patterns:**
- Provides diagnoses on complete cases (0-32% pause), pauses on incomplete ones (83-99% pause). This is the robustness behavior.
- Top-3/Top-4 accuracy improves over Top-1, showing correct diagnoses appear in differentials.
- Best NEJM (out-of-domain): MedGemma-4B-IT at 36% Top-1, 44% Top-3.
- Best JAADCR (domain-matched): MedGemma-1.5-4B-IT with options at 52% Top-1.

---

## Setup

**Prerequisites**: Python 3.10+, Google Cloud account with Vertex AI, no local GPU required.

```bash
git clone <repo-url>
cd MedGemma
conda create -n medgemma python=3.10 && conda activate medgemma
pip install -r requirements.txt
```

Create `.env`:
```bash
GOOGLE_API_KEY=your_key              # Gemini orchestrator
GOOGLE_CLOUD_PROJECT=your_project    # Vertex AI
GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
RAG_BACKEND=vertex
VERTEX_RAG_LOCATION=us-west1
VERTEX_RAG_CORPUS=projects/.../locations/.../ragCorpora/...
```

Deploy MedGemma models via Vertex AI Model Garden (one-click deploy), then update endpoint IDs in `src/agents/registry.py`.

---

## Usage

```bash
# Launch Gradio UI
bash bin/run_gradio_app.sh

# Run evaluations
python scripts/evaluate_nejim_cases.py \
  --input NEJIM/image_challenge_input \
  --agent-model medgemma-27b-it-vertex \
  --max-cases 1

# JAADCR evaluation
python scripts/evaluate_jdcr_cases.py \
  --input JAADCR/jaadcr_input \
  --agent-model medgemma-4b-it-vertex \
  --max-cases 1
```

---

## JAADCR Data Pipeline

Reproducible pipeline for downloading and preprocessing JDCR Case Challenge PDFs:

```bash
# Stage 1: Extract and split into evaluation format using Gemini API
python scripts/jdcr_data_downlaod_preprocess/stage_1_extract_and_split.py \
  --extracted-dir ./backup_extracted --output-dir ./output

# Stage 2: Build ground truth CSV
python scripts/jdcr_data_downlaod_preprocess/stage_2_build_ground_truth.py \
  --metadata-dir ./output/case_metadata --output-csv ./output/JAADCR_Groundtruth.csv

# Stage 3: Evaluate with MedGemma
python scripts/evaluate_jdcr_cases.py \
  --input ./output/jaadcr_input --agent-model medgemma-27b-it-vertex
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | Google ADK |
| Clinical Reasoning | MedGemma-27B-IT, 4B-IT, 1.5-4B-IT (Vertex AI) |
| Orchestration | Gemini Pro (9-model fallback) |
| RAG | Vertex AI RAG (55 documents, text-embedding-005) |
| UI | Gradio 6.x |
| Knowledge Base | AAD Guidelines, StatPearls, JAADCR Case Reports |

---

Apache 2.0 License

Last Updated: February 15, 2026
