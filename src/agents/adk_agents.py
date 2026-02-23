"""
Google ADK Multi-Agent System for MedGemma Clinical Robustness Assistant

HYBRID ARCHITECTURE: Manager-Specialist Pattern
==============================================

This implements a sophisticated two-tier reasoning system:

TIER 1: Orchestration (Gemini via Google ADK)
- Framework: Google Agent Development Kit (ADK)
- Model: Gemini Pro Latest (gemini-pro-latest)
- Role: Workflow management, delegation, coordination
- Handles: Agent routing, state management, tool orchestration

TIER 2: Clinical Reasoning (MedGemma-27B-IT Specialist)
- Model: MedGemma-27B-IT (google/medgemma-27b-it via HuggingFace API)
- Role: Medical diagnosis, clinical analysis, SOAP generation
- Handles: All high-stakes medical reasoning tasks
- Invoked via: FunctionTools (medgemma_triage_analysis, medgemma_guideline_synthesis, medgemma_clinical_diagnosis)

Why This Architecture?
----------------------
1. Competition Requirement: Uses MedGemma-27B-IT for medical reasoning (mandatory)
2. Google ADK Integration: Demonstrates advanced agent orchestration
3. Best of Both Worlds: Gemini Pro Latest orchestration + MedGemma-27B-IT medical expertise
4. Production-Grade: Separates workflow management from domain expertise

Agent Workflow:
--------------
1. Triage Agent (ADK/Gemini Pro Latest orchestrates → MedGemma-27B-IT analyzes completeness)
2. Research Agent (ADK/Gemini Pro Latest orchestrates RAG → MedGemma-27B-IT synthesizes guidelines)
3. Diagnostic Agent (ADK/Gemini Pro Latest orchestrates → MedGemma-27B-IT generates SOAP note)
4. Root Coordinator (ADK/Gemini Pro Latest manages the entire workflow)

Toggle Comparison Feature:
--------------------------
The system supports comparing MedGemma-27B-IT (default) vs Gemini Pro Latest for the
three specialist agents. The orchestrator ALWAYS uses Gemini Pro Latest via Google ADK.
- Default: All 3 agents use MedGemma-27B-IT for clinical reasoning
- Toggle: Switch agents to use Gemini Pro Latest for baseline comparison

This is the "Clinical Specialist Pattern": Gemini Pro Latest is the clinic manager,
MedGemma-27B-IT is the specialist physician.
"""

import logging
import os
import re
from typing import Dict, Any, List, Optional

# Disable ADK telemetry to avoid JSON serialization issues with bytes
os.environ['ADK_TELEMETRY_DISABLED'] = '1'
os.environ['GOOGLE_ADK_TELEMETRY'] = '0'

from google.adk import Agent, Runner
from google.adk.tools import FunctionTool
from google.genai import types

# Monkey-patch ADK telemetry to avoid JSON serialization errors with bytes
try:
    from google.adk import telemetry
    # Replace trace functions with no-ops
    telemetry.trace_call_llm = lambda *args, **kwargs: None
    telemetry.trace_llm_response = lambda *args, **kwargs: None
    print("✓ ADK telemetry disabled (monkey-patched)")
except Exception:
    pass  # Silently continue if telemetry module not found

from src.utils.schemas import (
    ClinicalCase,
    TriageResult,
    ResearchResult,
    DiagnosticResult,
    RetrievedDocument,
    SOAPNote,
    MissingDataType
)
from src.agents.conversation_manager import get_conversation_manager
from config.config import settings

logger = logging.getLogger(__name__)

# Module-level dict to pass MedGemma tool outputs back to run_async().
# ADK runs tools in a thread pool (different thread from run_async),
# so threading.local() doesn't work — use a plain shared dict instead.
# Safe because the UI processes one request at a time.
_run_outputs: Dict[str, Any] = {
    "last_triage_output": None,
    "last_research_context": None,
    "last_diagnostic_output": None,
    "last_image_path": None,         # image path threaded to MedGemma micro-calls
    "last_retrieved_guidelines": [],  # guidelines from RAG, passed to UI for citations
    "last_rag_query": "",             # search query used, for citations display
}


def _sanitize_response(text: str) -> str:
    """
    Sanitize MedGemma response to prevent JSON serialization errors.

    MedGemma occasionally outputs backslashes in medical/scientific notation
    (e.g., \\alpha, \\beta, \\mu) that are not valid JSON escape sequences.
    When ADK packages the tool response as JSON to pass back to Gemini,
    these invalid escapes cause JSONDecodeError. This function replaces them
    with safe double-backslash equivalents before the response is returned.

    Valid JSON escape characters: \" \\ \/ \\b \\f \\n \\r \\t \\uXXXX
    Everything else (e.g., \\a, \\p, \\m) is invalid and replaced.
    """
    # Strip MedGemma thinking tokens that sometimes leak into output
    # e.g., "thoughtThe user wants me to..." → "The user wants me to..."
    if text.startswith('thought'):
        text = text[len('thought'):]
    # Fix invalid JSON escape sequences
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)


# ============================================================================
# TOOLS - Functions that agents can call
# ============================================================================

# Initialize retriever (shared across agents)
# Choose between ChromaDB (local) or Vertex AI RAG (cloud) based on config
if settings.rag_backend == "vertex" and settings.vertex_rag_corpus:
    from src.rag.vertex_rag_retriever import VertexRAGRetriever
    _retriever = VertexRAGRetriever(
        project_id=settings.google_cloud_project,
        location=settings.vertex_rag_location,
        corpus_name=settings.vertex_rag_corpus,
    )
    logger.info(f"✓ Vertex AI RAG retriever initialized (corpus: {settings.vertex_rag_corpus})")
else:
    from src.rag.retriever import Retriever
    _retriever = Retriever()
    logger.info(f"✓ ChromaDB retriever initialized (collection: {settings.chroma_collection_name})")

# MedGemma adapter (initialized lazily to avoid import-time errors)
from src.agents.models.medgemma_adapter import MedGemmaAdapter
_medgemma_specialist = None
_agent_model_choice = "medgemma"  # Global: "medgemma" or "gemini"

def _specialist_display_name() -> str:
    """Return the actual model display name based on current _agent_model_choice."""
    names = {
        "medgemma-vertex":        "google/medgemma-1.5-4b-it (Vertex AI)",
        "medgemma-1.5-4b":        "google/medgemma-1.5-4b-it",
        "medgemma-hf":            "google/medgemma-1.5-4b-it (HF API)",
        "medgemma":               "google/medgemma-27b-it",
        "medgemma-27b-it-vertex": "google/medgemma-27b-it (Vertex AI)",
        "medgemma-4b-it-vertex":  "google/medgemma-4b-it (Vertex AI)",
        "gemini":                 "gemini (baseline)",
    }
    return names.get(_agent_model_choice, _agent_model_choice)

def _get_medgemma_specialist():
    """
    Get or create specialist for clinical reasoning (lazy initialization).

    Returns MedGemma (specialized) or Gemini (baseline) based on agent_model_choice.
    """
    global _medgemma_specialist, _agent_model_choice

    # Check if we should use Gemini instead of MedGemma
    if _agent_model_choice == "gemini":
        from src.agents.models.gemini_adapter import GeminiAdapter
        logger.info("Using Gemini Pro for agent reasoning (baseline comparison)")
        if _medgemma_specialist is None or not isinstance(_medgemma_specialist, GeminiAdapter):
            _medgemma_specialist = GeminiAdapter(
                model_id=settings.gemini_model_id,
                api_key=settings.google_api_key
            )
        return _medgemma_specialist

    # Check if we should use Vertex AI MedGemma (1.5-4B-IT, 4B-IT, or 27B-IT)
    if _agent_model_choice in ["medgemma-vertex", "medgemma-4b-it-vertex", "medgemma-27b-it-vertex"]:
        from src.agents.models.vertex_medgemma_adapter import VertexMedGemmaAdapter
        from src.agents.registry import MODEL_REGISTRY

        # Determine model size for logging
        if _agent_model_choice == "medgemma-27b-it-vertex":
            model_size = "27B-IT"
        elif _agent_model_choice == "medgemma-4b-it-vertex":
            model_size = "4B-IT"
        else:
            model_size = "1.5-4B-IT"

        logger.info(f"Using MedGemma-{model_size} via Vertex AI for agent reasoning (multimodal: image+text)")
        if _medgemma_specialist is None or not isinstance(_medgemma_specialist, VertexMedGemmaAdapter):
            vertex_config = MODEL_REGISTRY[_agent_model_choice]
            _medgemma_specialist = VertexMedGemmaAdapter(
                model_id=vertex_config["model_id"],
                project_id=vertex_config["project_id"],
                region=vertex_config["region"],
                endpoint_id=vertex_config["endpoint_id"]
            )
        return _medgemma_specialist

    # Use MedGemma-1.5-4B-IT locally on GPU (weights cached from login node)
    if _agent_model_choice == "medgemma-1.5-4b":
        logger.info("Using MedGemma-1.5-4B-IT for agent reasoning (local GPU, cached weights)")
        if _medgemma_specialist is None or not isinstance(_medgemma_specialist, MedGemmaAdapter):
            _medgemma_specialist = MedGemmaAdapter(
                model_id="google/medgemma-1.5-4b-it",
                api_key=settings.huggingface_api_key
            )
        return _medgemma_specialist

    # Use MedGemma via HuggingFace Inference API (serverless, no local GPU)
    if _agent_model_choice == "medgemma-hf":
        from src.agents.models.hf_inference_adapter import HFInferenceAdapter
        logger.info("Using MedGemma-1.5-4B-IT via HuggingFace Inference API (serverless)")
        if _medgemma_specialist is None or not isinstance(_medgemma_specialist, HFInferenceAdapter):
            _medgemma_specialist = HFInferenceAdapter(
                model_id="google/medgemma-1.5-4b-it",
                api_key=settings.huggingface_api_key
            )
        return _medgemma_specialist

    # Use MedGemma local GPU (default/specialized)
    if _medgemma_specialist is None or not isinstance(_medgemma_specialist, MedGemmaAdapter):
        logger.info("Using MedGemma-27B-IT for agent reasoning (specialized)")
        _medgemma_specialist = MedGemmaAdapter(
            model_id=settings.medgemma_model_id,
            api_key=settings.huggingface_api_key
        )
    return _medgemma_specialist

def retrieve_clinical_guidelines(
    query: str,
    n_results: int = 5
) -> Dict[str, Any]:
    """
    Retrieve relevant clinical guidelines from ChromaDB.

    Args:
        query: Natural language query describing the clinical presentation
        n_results: Number of guidelines to retrieve (default: 5)

    Returns:
        Dict with retrieved guidelines and metadata
    """
    logger.info(f"Tool called: retrieve_clinical_guidelines(query='{query[:50]}...', n_results={n_results})")

    # Query RAG system (Vertex RAG or ChromaDB)
    # Lower threshold for Vertex RAG which returns distance-based scores (0.25-0.4 typical)
    retrieved_docs = _retriever.retrieve(
        query=query,
        n_results=n_results,
        min_similarity=0.25  # Lowered from 0.5 to work with Vertex RAG scores
    )

    # Format for agent consumption
    guidelines = []
    for doc in retrieved_docs:
        guidelines.append({
            "content": doc.content,
            "source": doc.source,
            "title": doc.title or "Unknown",
            "similarity_score": doc.similarity_score,
            "metadata": doc.metadata
        })

    result = {
        "query": query,
        "guideline_count": len(guidelines),
        "guidelines": guidelines,
        "top_similarity": guidelines[0]["similarity_score"] if guidelines else 0.0
    }

    # If no guidelines found, add a clear message so the orchestrator doesn't loop
    if len(guidelines) == 0:
        result["message"] = (
            "No matching guidelines found in the knowledge base for this query. "
            "This is acceptable — proceed with clinical reasoning without guideline citations. "
            "Do NOT retry with different queries. Move on to the next step."
        )

    logger.info(f"Retrieved {len(guidelines)} guidelines (top similarity: {result['top_similarity']:.2f})")

    # Store for UI citation display (bypasses function-call arg parsing which is fragile)
    _run_outputs["last_retrieved_guidelines"] = guidelines
    _run_outputs["last_rag_query"] = query

    return result


def analyze_case_completeness(
    history: Optional[str] = None,
    physical_exam: Optional[str] = None,
    image_available: bool = False,
    patient_age: Optional[int] = None,
    patient_gender: Optional[str] = None,
    duration: Optional[str] = None
) -> Dict[str, Any]:
    """
    Collect all available clinical data and pass it to MedGemma for evaluation.

    NOTE: This function does NOT judge completeness — it simply assembles the
    full case text so MedGemma can decide for itself whether there is enough
    information to proceed with diagnosis.

    Returns:
        Dict with the combined case text for MedGemma triage analysis
    """
    logger.info("Tool called: analyze_case_completeness")

    # Combine all available text into one block for MedGemma to evaluate
    parts = []
    if history:
        parts.append(history.strip())
    if physical_exam:
        parts.append(physical_exam.strip())
    full_case_text = "\n\n".join(parts)

    if not full_case_text:
        logger.info("Triage: no case text provided at all")
        return {
            "full_case_text": "",
            "has_sufficient_context": False,
            "reasoning": "No case text provided. Please describe the patient case."
        }

    logger.info(f"Triage: assembled case text ({len(full_case_text)} chars) for MedGemma evaluation")
    return {
        "full_case_text": full_case_text,
        "has_sufficient_context": True,  # MedGemma will make the final call
        "reasoning": "Case text assembled. MedGemma will evaluate clinical completeness."
    }



# ============================================================================
# MEDGEMMA SPECIALIST TOOLS - Core Clinical Reasoning
# ============================================================================

def medgemma_triage_analysis(
    case_summary: str,
    missing_items: List[str]
) -> Dict[str, Any]:
    """
    Call MedGemma specialist to analyze case completeness and generate clarification questions.

    This delegates high-stakes triage reasoning to MedGemma-27B.

    Args:
        case_summary: Summary of the clinical case
        missing_items: List of missing data items

    Returns:
        Dict with MedGemma's triage analysis and questions
    """
    logger.info("Calling MedGemma Specialist for triage analysis")

    image_path = _run_outputs.get("last_image_path")

    # ── Python pre-filter ─────────────────────────────────────────────────────
    # MedGemma ignores triage instructions when "SOAP" appears in the input and
    # just generates a SOAP note directly.  Detect "command-only" inputs in Python
    # before calling MedGemma — this is 100% reliable and instant.
    #
    # Strategy: require at least ONE real demographic or time marker.
    # Simple keywords like "diagnosis" or "treatment" are too weak —
    # "Can you diagnose this?" would match.  Real clinical narratives almost
    # always contain age (digits + year), gender, or a duration (N months/weeks).
    import re as _re
    CLINICAL_PATTERNS = [
        r'\d+[\s\-]*year',            # "45-year", "45 year" → age
        r'year[\s\-]*old',            # "year-old", "year old"
        r'\b(male|female|man|woman|boy|girl|baby|infant|toddler|child|newborn|adolescent|teen|elderly)\b',   # gender/age group
        r'\d+[\s\-]*(month|week|day)s?\b',         # "3 months", "2-week" → duration
        r'presents?\s+with',          # "presents with" → clinical phrasing
        r'chief\s+complaint',         # structured clinical intake
        r'history\s+of',              # clinical history marker
    ]
    summary_lower = (case_summary or "").lower().strip()
    has_clinical_content = any(_re.search(p, summary_lower) for p in CLINICAL_PATTERNS)

    if not has_clinical_content and not image_path:
        # No clinical text AND no image — skip MedGemma, return LACK INFORMATION directly
        lack_msg = (
            "LACK INFORMATION: I need more clinical details to generate a SOAP note.\n\n"
            "Please provide:\n"
            "1. **Patient age and gender** (e.g., 45-year-old female)\n"
            "2. **Chief complaint** (main reason for visit)\n"
            "3. **Duration of symptoms** (how long have they had this?)\n"
            "4. **Relevant medical history** (prior conditions, medications, treatments)"
        )
        logger.info("Pre-filter: no clinical keywords and no image — returning LACK INFORMATION without calling MedGemma")
        _run_outputs["last_triage_output"] = lack_msg
        return {
            "medgemma_analysis": lack_msg,
            "missing_items": missing_items,
            "specialist_used": "Pre-filter (no clinical content)"
        }
    # ── End pre-filter ────────────────────────────────────────────────────────

    # ── Image-only detection ──────────────────────────────────────────────────
    # When user uploads only image(s) with no clinical text history, we still
    # call MedGemma but with a prompt that forces it to ask for missing info.
    # An image alone (no age, gender, history, duration) is NEVER sufficient.
    _is_image_only = image_path and not has_clinical_content
    # ── End image-only detection ──────────────────────────────────────────────

    image_note = "\n\nA clinical image has been provided and is attached to this request." if image_path else ""

    if _is_image_only:
        prompt = f"""You are an expert dermatology triage specialist.

A clinical image has been provided but NO clinical history, patient demographics, or symptom description was given.

YOUR TASK:
Look at the attached clinical image. Describe what you observe in the image (lesion morphology, color, distribution, pattern). Then ask the user for the missing clinical information needed before any diagnosis can be made.

You MUST respond with "LACK INFORMATION:" followed by:
- A brief description of what you see in the image
- Specific questions asking for the missing information

You MUST ask about ALL of the following:
1. Patient age and gender
2. Chief complaint / reason for visit
3. Duration of symptoms
4. Location on the body (if not clear from image)
5. Relevant medical history, medications, allergies

STRICT RULES:
- Do NOT generate a SOAP note
- Do NOT provide a diagnosis or differential
- Do NOT guess patient demographics from the image
- You MUST start your response with "LACK INFORMATION:"
"""
    else:
        prompt = f"""You are an expert dermatology triage specialist.

FULL CASE TEXT:
{case_summary}{image_note}

YOUR TASK:
Read the case text above carefully. If a clinical image is attached, consider it as part of the available data (you can see lesion morphology, distribution, color). YOU decide whether there is sufficient clinical information to make a diagnosis.

Decision rules:
- If the case contains patient demographics AND symptoms/lesion description (from text OR visible in the image) → respond ONLY with: "DETAILS: Proceed with diagnosis."
- If critical clinical context is missing → respond ONLY with: "LACK INFORMATION:" followed by specific clarifying questions.

STRICT RULES — DO NOT VIOLATE:
- Do NOT hallucinate or guess patient age, gender, or demographics from the image alone. Only use demographics explicitly stated in the text.
- Do NOT invent symptoms, history, or findings that are not in the provided text or image.
- If text says "skin changes" but does not describe them AND no image is provided, that IS missing information.
- If text says "skin changes" and an image IS provided showing visible findings, the image counts as the description.

Do NOT generate a SOAP note. Do NOT diagnose. Just assess completeness."""

    try:
        specialist = _get_medgemma_specialist()
        response = specialist.generate(
            prompt=prompt,
            max_tokens=300,
            temperature=0.0,
            **({"image_path": image_path} if image_path else {})
        )
        response = _sanitize_response(response)

        # If MedGemma ignored instructions and generated a SOAP note anyway:
        # - Image-only: force LACK INFORMATION (image alone is never enough)
        # - Normal case with clinical text: treat as DETAILS (case had enough info)
        response_lower = response.lower()
        if "lack information" not in response_lower and "details" not in response_lower:
            if _is_image_only:
                logger.warning("MedGemma ignored image-only triage instructions — forcing LACK INFORMATION")
                response = "LACK INFORMATION: " + response
            else:
                logger.warning("MedGemma ignored triage instructions — treating as DETAILS (case had clinical content)")
                response = "DETAILS: Proceed with diagnosis."
        elif _is_image_only and "details" in response_lower and "lack information" not in response_lower:
            # MedGemma said DETAILS for image-only — override to LACK INFORMATION
            logger.warning("MedGemma said DETAILS for image-only case — forcing LACK INFORMATION")
            response = response.replace("DETAILS", "LACK INFORMATION", 1)
            response = response.replace("Proceed with diagnosis.", "Additional clinical information is needed.", 1)

        # Store in thread-local so run_async() can retrieve MedGemma's actual output
        _run_outputs["last_triage_output"] = response

        return {
            "medgemma_analysis": response,
            "missing_items": missing_items,
            "specialist_used": _specialist_display_name()
        }
    except Exception as e:
        logger.error(f"MedGemma triage analysis failed: {e}", exc_info=True)
        import traceback
        error_details = traceback.format_exc()
        error_msg = f"Error calling MedGemma triage: {e}"
        # Store error so run_async() priority chain doesn't fall back to ghost ADK events
        _run_outputs["last_triage_output"] = error_msg
        return {
            "medgemma_analysis": f"Error calling MedGemma: {e}\n\nDetails:\n{error_details}",
            "missing_items": missing_items,
            "specialist_used": _specialist_display_name() + " (failed)"
        }


def medgemma_guideline_synthesis(
    case_data: str,
    retrieved_guidelines: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Call MedGemma specialist to synthesize retrieved guidelines with case data.

    This delegates clinical guideline interpretation to MedGemma-27B.

    Args:
        case_data: Clinical case information
        retrieved_guidelines: Guidelines retrieved from ChromaDB

    Returns:
        Dict with MedGemma's synthesis
    """
    logger.info("Calling MedGemma Specialist for guideline synthesis")

    # Format guidelines for MedGemma
    guidelines_text = "\n\n".join([
        f"**{g.get('title', 'Guideline')}** (Source: {g.get('source', 'Unknown')})\n{g.get('content', '')}"
        for g in retrieved_guidelines[:5]  # Top 5
    ])

    prompt = f"""You are a medical research specialist synthesizing clinical guidelines.

Clinical Case:
{case_data}

Retrieved Clinical Guidelines:
{guidelines_text}

Your Task:
1. Identify which guidelines are most relevant to this case
2. Extract key diagnostic criteria and clinical features
3. Summarize evidence-based recommendations
4. Provide specific citations (Source: Title)

Provide a structured synthesis focusing on differential diagnosis support."""

    try:
        specialist = _get_medgemma_specialist()
        response = specialist.generate(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.0
        )
        response = _sanitize_response(response)

        # Store in thread-local so medgemma_clinical_diagnosis can use it directly
        # (Gemini passes garbage fragments as research_context — bypass that entirely)
        _run_outputs["last_research_context"] = response

        return {
            "medgemma_synthesis": response,
            "guidelines_count": len(retrieved_guidelines),
            "specialist_used": _specialist_display_name()
        }
    except Exception as e:
        logger.error(f"MedGemma guideline synthesis failed: {e}", exc_info=True)
        import traceback
        error_details = traceback.format_exc()
        return {
            "medgemma_synthesis": f"Error calling MedGemma: {e}\n\nDetails:\n{error_details}",
            "guidelines_count": len(retrieved_guidelines),
            "specialist_used": _specialist_display_name() + " (failed)"
        }


def medgemma_clinical_diagnosis(
    case_data: str
) -> Dict[str, Any]:
    """
    Call MedGemma specialist for final clinical diagnosis and SOAP note generation.

    This is the PRIMARY clinical reasoning task — delegates all medical reasoning to
    MedGemma (27B, 4B, or 1.5-4B-IT via Vertex AI depending on agent_model_choice).

    Parameter Bypass Design:
    ────────────────────────
    The `triage_notes` and `research_context` parameters from Gemini are intentionally
    bypassed. When Gemini calls this tool, it passes its own summarized/truncated
    fragments of the context — not the actual MedGemma outputs from the previous agents.
    Instead, we read directly from `_run_outputs`, a module-level dict that stores the
    raw MedGemma outputs from `medgemma_triage_analysis` and `medgemma_guideline_synthesis`
    as they ran. This guarantees MedGemma sees the real clinical context, not Gemini's
    paraphrase.

    Conditional Prompt Strategy (Feb 2026 fix):
    ────────────────────────────────────────────
    Previously, the prompt led with triage notes and told MedGemma to "READ TRIAGE
    NOTES FIRST". This caused MedGemma to narrate/echo the triage notes rather than
    generating a SOAP note (it treated the instruction as a reading comprehension task).

    The fix: detect whether triage said DETAILS or LACK INFORMATION, then branch:
    - DETAILS → clean, direct "Generate a complete SOAP note" prompt with case data
      and guidelines. No triage notes in the prompt at all — avoids the echo problem.
    - LACK INFORMATION → short prompt asking MedGemma to output the clarifying questions
      identified during triage, with case data for reference.

    Recovery Path Compatibility:
    ─────────────────────────────
    This function is also called directly by the run_async() recovery path when Gemini
    stops the pipeline before DiagnosticAgent runs (e.g., after empty RAG results).
    In that case, `_run_outputs` are restored to the saved local copies before this
    call, ensuring the same context is available as in the normal pipeline.

    Args:
        case_data: Complete clinical case information (the original user message or
                   formatted case text — used directly in the prompt).
                   research_context and triage_notes are intentionally removed from
                   the signature: they are read from _run_outputs internally, and
                   removing them prevents Gemini from generating long JSON function
                   call arguments that occasionally cause ADK JSON parse errors.

    Returns:
        Dict with keys:
          - soap_note: MedGemma's SOAP note text (or clarifying questions if incomplete)
          - specialist_used: Display name of the model that performed clinical reasoning
          - reasoning_engine: Human-readable engine description
    """
    logger.info("Calling MedGemma Specialist for clinical diagnosis (SOAP note)")

    # Read real MedGemma outputs stored by previous tool calls in the pipeline.
    # _run_outputs["last_triage_output"] is set by medgemma_triage_analysis().
    # _run_outputs["last_research_context"] is set by medgemma_guideline_synthesis().
    # These are authoritative — Gemini's parameter values are bypassed entirely
    # because Gemini summarizes/truncates context when passing args to this tool.
    real_triage_notes = _run_outputs.get("last_triage_output") or 'No triage notes.'
    real_research_context = _run_outputs.get("last_research_context") or ''
    image_path = _run_outputs.get("last_image_path")  # pass to every MedGemma generate() call

    # Safety net: if ResearchAgent skipped RAG (Gemini non-determinism), force-call it now.
    # This ensures every diagnosis has RAG citations when possible.
    if not real_research_context and not _run_outputs.get("last_retrieved_guidelines"):
        logger.warning("ResearchAgent skipped RAG — forcing retrieve_clinical_guidelines + medgemma_guideline_synthesis")
        try:
            # Extract medical keywords for a focused RAG query (3-8 words)
            _stop = {
                'a','an','the','is','was','were','been','being','have','has','had','do','does',
                'did','will','would','shall','should','may','might','can','could','am','are',
                'to','of','in','for','on','with','at','by','from','as','into','through',
                'during','before','after','between','out','off','over','under','then','once',
                'when','where','why','how','all','each','every','both','few','more','most',
                'other','some','such','no','nor','not','only','own','same','so','than','too',
                'very','just','because','but','and','or','if','who','which','that','this',
                'these','those','it','its','he','she','they','them','their','his','her','him',
                'presented','present','patient','man','woman','boy','girl','admitted','hospital',
                'days','weeks','months','years','year','day','week','month','old','new','left',
                'right','due','received','receiving','first','second','reported','noted','known',
                'history','medical','cm','mm','diagnosis','also','upon','about','there','here',
            }
            _words = re.findall(r'[a-zA-Z-]+', case_data[:500])
            _keywords = [w for w in _words if w.lower() not in _stop and len(w) > 2][:8]
            rag_query = ' '.join(_keywords) if _keywords else case_data.split('.')[0][:100]
            logger.info(f"Safety-net RAG query: '{rag_query}'")
            rag_result = retrieve_clinical_guidelines(query=rag_query.strip(), n_results=5)
            guidelines = rag_result.get("guidelines", [])
            # Also force guideline synthesis
            synth_result = medgemma_guideline_synthesis(
                case_data=case_data,
                retrieved_guidelines=guidelines
            )
            real_research_context = _run_outputs.get("last_research_context") or 'No guidelines retrieved.'
            logger.info(f"Forced RAG retrieval: {len(guidelines)} guidelines, synthesis={len(real_research_context)} chars")
        except Exception as e:
            logger.error(f"Forced RAG retrieval failed: {e}")
            real_research_context = 'No guidelines retrieved.'

    if not real_research_context:
        real_research_context = 'No guidelines retrieved.'

    logger.info(f"Using stored triage ({len(real_triage_notes)} chars) and research ({len(real_research_context)} chars), image={'yes' if image_path else 'no'}")

    # Route to the appropriate prompt based on triage verdict.
    # "details" is NOT a substring of "lack information" so no collision risk.
    _triage_lower = real_triage_notes.strip().lower()
    triage_says_sufficient = (
        not real_triage_notes
        or _triage_lower == "no triage notes."
        or (
            "details" in _triage_lower
            and "lack information" not in _triage_lower
        )
    )

    if triage_says_sufficient:
        # ── OPTION A: 8 micro-calls (ACTIVE) ─────────────────────────────────────────
        # The Vertex one-click-deploy endpoint hard-caps output to ~60-85 chars per call
        # regardless of max_new_tokens — confirmed via RAW response logging.
        # Single-call SOAP (Option B) always truncates mid-sentence.
        # Fix: decompose into 8 focused micro-questions each answerable in ~15-50 chars,
        # then stitch answers into a formatted SOAP note.
        ctx = (
            f"Dermatology case: {case_data}\n"
            f"STRICT: Only use information explicitly provided in the text or visible in the image. "
            f"Do NOT hallucinate or guess patient age, gender, or demographics from the image. "
            f"If a detail was not provided, say 'not provided' — do NOT invent it."
        )
        specialist = _get_medgemma_specialist()

        _PREAMBLES = (
            "here is a brief summary", "here are the", "based on the",
            "here is the", "the following", "i would recommend", "okay,", "sure,",
        )

        _call_counter = [0]  # mutable counter for closure
        _CALL_LABELS = [
            "Primary Diagnosis",
            "Confidence Score",
            "Differentials",
            "Supporting Evidence",
            "Subjective (S)",
            "Objective (O)",
            "Diagnostic Tests",
            "Treatment Plan",
        ]

        def _ask(q):
            _call_counter[0] += 1
            n = _call_counter[0]
            label = _CALL_LABELS[n - 1] if n <= len(_CALL_LABELS) else f"Call {n}"
            print(f"\n  [{n}/8] MedGemma-27B :: {label}", flush=True)
            raw = specialist.generate(
                prompt=f"{ctx}\nAnswer in one short sentence only, no preamble: {q}",
                max_tokens=500,
                temperature=0.0,
                **({"image_path": image_path} if image_path else {})
            ).strip()
            lower = raw.lower()
            for p in _PREAMBLES:
                if lower.startswith(p):
                    cut = raw.find(":")
                    if cut != -1 and cut < 80:
                        raw = raw[cut+1:].strip()
                    break
            print(f"         >>> {raw[:150]}{'...' if len(raw) > 150 else ''}", flush=True)
            return raw

        primary_dx    = _ask("What is the single most likely dermatological diagnosis? Reply with the diagnosis name only, in English, nothing else.")
        confidence    = _ask("On a scale of 0.0 to 1.0, how confident are you in this diagnosis? Reply with a single decimal number only, e.g. 0.85")
        differentials = _ask("List exactly 3 alternative diagnoses in English, separated by commas. Names only, no explanations.")
        evidence      = _ask("In one English sentence only, which clinical features from the case best support the primary diagnosis?")
        subjective    = _ask("In one English sentence only, summarize the patient age, chief complaint, and symptom duration.")
        objective     = _ask("In one English sentence only, describe the lesion morphology, distribution, and color.")
        tests         = _ask("List the recommended diagnostic tests in English, separated by commas. Names only.")
        treatment     = _ask("In one English sentence only, what is the first-line treatment and follow-up plan?")

        logger.info(f"SOAP micro-calls complete: dx={primary_dx}, conf={confidence}")

        response = (
            f"**Subjective (S):**\n{subjective}\n\n"
            f"**Objective (O):**\n{objective}\n\n"
            f"**Assessment (A):**\n"
            f"- Primary Diagnosis: {primary_dx} (Confidence: {confidence})\n"
            f"- Differentials: {differentials}\n"
            f"- Supporting Evidence: {evidence}\n\n"
            f"**Plan (P):**\n"
            f"- Diagnostics: {tests}\n"
            f"- Treatment: {treatment}"
        )
        # ── END OPTION A ──────────────────────────────────────────────────────────────

        # ── OPTION B: single-call compact format (ChatGPT suggestion — DOES NOT WORK) ──
        # Tested 2026-02-19: endpoint still truncates at ~70 chars output even with
        # short prompt. Total context cap is ~60-85 chars output regardless of input size.
        # Example truncation: "S: A 73-year-old man with a history of receiving" → cut off.
        #
        # specialist = _get_medgemma_specialist()
        # single_prompt = (
        #     f"Case:\n{case_data}\n\n"
        #     f"Output exactly in this format:\n"
        #     f"S: ...\n"
        #     f"O: ...\n"
        #     f"A:\n"
        #     f"- Primary DX: ...\n"
        #     f"- Confidence (0-1): ...\n"
        #     f"- Differentials: ...\n"
        #     f"P:\n"
        #     f"- Tests: ...\n"
        #     f"- Treatment: ..."
        # )
        # raw_soap = specialist.generate(prompt=single_prompt, max_tokens=800, temperature=0.0).strip()
        # logger.info(f"SOAP single-call response ({len(raw_soap)} chars)")
        # response = _sanitize_response(raw_soap)
        # _run_outputs["last_diagnostic_output"] = response
        # return {"soap_note": response, "specialist_used": _specialist_display_name(), "reasoning_engine": "MedGemma (single-call compact SOAP)"}
        # ── END OPTION B ──────────────────────────────────────────────────────────────

        response = _sanitize_response(response)
        _run_outputs["last_diagnostic_output"] = response
        return {
            "soap_note": response,
            "specialist_used": _specialist_display_name(),
            "reasoning_engine": "MedGemma (8 micro-calls, stitched SOAP)"
        }
    else:
        prompt = f"""You are an expert dermatology triage specialist.

CLINICAL CASE:
{case_data}

TRIAGE ASSESSMENT:
{real_triage_notes}

Critical clinical information is missing. Output the specific clarifying questions identified during triage.
Be precise — ask only for what is genuinely missing from the case description above."""

    try:
        specialist = _get_medgemma_specialist()
        response = specialist.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.0,
            **({"image_path": image_path} if image_path else {})
        )
        response = _sanitize_response(response)

        # Store in thread-local so run_async() always gets the real MedGemma output
        # (Gemini's text event after the tool call has attribution issues — store here directly)
        _run_outputs["last_diagnostic_output"] = response

        return {
            "soap_note": response,
            "specialist_used": _specialist_display_name(),
            "reasoning_engine": "MedGemma-27B (Health-Specialized)"
        }
    except Exception as e:
        logger.error(f"MedGemma clinical diagnosis failed: {e}", exc_info=True)
        import traceback
        error_details = traceback.format_exc()
        error_msg = f"MedGemma clinical diagnosis failed: {e}\n\nUnable to generate diagnosis. Please check the model endpoint and try again."
        # Store error so run_async() priority chain doesn't fall back to ghost ADK events
        _run_outputs["last_diagnostic_output"] = error_msg
        return {
            "soap_note": error_msg,
            "specialist_used": _specialist_display_name() + " (failed)",
            "reasoning_engine": "Error"
        }


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

def create_triage_agent(model_name: str = "gemini-pro-latest") -> Agent:
    """
    Create Triage Agent using Google ADK.

    ARCHITECTURE: Gemini (ADK) orchestrates, MedGemma provides clinical reasoning.

    Role in pipeline: STEP 1 of 3. Receives the raw case from RootCoordinator,
    checks data completeness, and always passes to ResearchAgent — never directly
    to DiagnosticAgent.

    Prompt Design Rationale:
    ─────────────────────────
    1. "ORCHESTRATION, not clinical reasoning" — forces Gemini to delegate the actual
       medical analysis to MedGemma (via medgemma_triage_analysis tool) rather than
       reasoning medically itself. Competition requirement: MedGemma must do all
       clinical work.

    2. Mandatory 3-step sequence — the instruction explicitly names each step with
       CALL directives. Without this, Gemini sometimes skips tool calls and writes
       the action as plain text (e.g., "I will call the triage tool...") without
       actually executing it. The phrase "EXECUTE the tool call" was required after
       observing this failure mode.

    3. Always call medgemma_triage_analysis (Step 2) even if Step 1 finds no missing
       data — this ensures MedGemma's analysis is always in the context chain passed
       to downstream agents, which improves ResearchAgent's search query quality.

    4. Transfer to ResearchAgent ONLY (never DiagnosticAgent) — ensures guideline
       retrieval always runs before diagnosis. Safety-first: RAG evidence required.

    5. Dermatology-specific missing data criteria: history + (exam OR image) +
       demographics. Visual data (exam findings or images) is non-negotiable in
       dermatology — a diagnosis from history alone is unsafe.

    Args:
        model_name: Gemini model for orchestration (default: flash for speed/cost)

    Returns:
        Google ADK Agent instance
    """
    return Agent(
        name="TriageAgent",
        description="Orchestrates case triage - delegates clinical analysis to MedGemma specialist",
        model=model_name,
        instruction="""
You are a clinical workflow coordinator managing the triage process.

Your role is ORCHESTRATION, not clinical reasoning.

MANDATORY STEPS — COMPLETE ALL 3 IN ORDER:

Step 1: CALL analyze_case_completeness tool.
  - Pass whatever fields you received (history, physical_exam, etc.).
  - This tool assembles the full case text for MedGemma.

Step 2: CALL medgemma_triage_analysis tool.
  - Pass case_summary = the FULL ORIGINAL CASE TEXT verbatim (do NOT summarize or shorten it).
  - Pass missing_items = [] (empty — MedGemma will decide what is missing from the text itself).

Step 3 (CONDITIONAL — read MedGemma's response from Step 2):
  - If MedGemma's response starts with "DETAILS" → CALL transfer_to_agent(agent_name='ResearchAgent').
  - If MedGemma's response starts with "LACK INFORMATION" OR contains clarifying questions → DO NOT transfer to any agent. Output MedGemma's clarifying questions directly as your response and STOP.
  - Writing "Transfer to ResearchAgent" as text is NOT enough — you must EXECUTE the tool call when transferring.
""",
        tools=[
            FunctionTool(analyze_case_completeness),
            FunctionTool(medgemma_triage_analysis)
        ],
        output_schema=None
    )


def create_research_agent(model_name: str = "gemini-pro-latest") -> Agent:
    """
    Create Research Agent using Google ADK.

    ARCHITECTURE: Gemini (ADK) orchestrates RAG retrieval, MedGemma synthesizes guidelines.

    Role in pipeline: STEP 2 of 3. Receives triage context from TriageAgent,
    queries the RAG knowledge base (Vertex AI RAG or ChromaDB), then passes
    synthesized evidence to DiagnosticAgent.

    Prompt Design Rationale:
    ─────────────────────────
    1. RAG retrieval is MANDATORY — even for cases where triage says "proceed".
       This ensures all diagnoses are evidence-grounded. Competition scoring rewards
       explainability, and cited guidelines directly support that (Explainability: 25%).

    2. "If 0 results are returned, proceed immediately" — prevents the agent from
       entering a retry loop searching with different queries. MedGemma's extensive
       parametric medical knowledge handles cases not covered by the RAG corpus.
       The instruction "Do NOT retry with different queries" was added after observing
       this failure mode in early testing.

    3. medgemma_guideline_synthesis is called EVEN IF 0 guidelines were retrieved —
       MedGemma then synthesizes from its pre-trained medical knowledge alone, which
       still provides valuable context for the DiagnosticAgent.

    4. Transfer enforced as explicit tool CALL (not text) — same lesson as TriageAgent:
       Gemini may write "I will transfer to DiagnosticAgent" without calling the tool.

    5. Search strategy guidance (symptoms, morphology, demographics) — without this,
       Gemini generates generic queries that return poor RAG results. The instruction
       focuses queries on clinical features specific to dermatology differential diagnosis.

    6. generate_content_config with FunctionCallingConfigMode.ANY (Feb 2026 fix) —
       Gemini probabilistically stops the pipeline after RAG returns empty or low-score
       results, producing no output and never transferring to DiagnosticAgent. This was
       observed in testing with RAG queries for rare diseases (e.g., annular RC form,
       pediatric bullous pemphigoid) that have no matching documents in the corpus.
       Setting mode=ANY forces every Gemini response within this agent to be a function
       call — so after retrieve_clinical_guidelines returns 0 results, Gemini MUST call
       medgemma_guideline_synthesis, and after that returns, it MUST call
       transfer_to_agent(DiagnosticAgent). There is no way for it to stop with free text.

    Args:
        model_name: Gemini model for orchestration (default: flash for speed/cost)

    Returns:
        Google ADK Agent instance
    """
    return Agent(
        name="ResearchAgent",
        description="Orchestrates guideline retrieval - delegates synthesis to MedGemma specialist",
        model=model_name,
        instruction="""
You are a research workflow coordinator managing evidence retrieval.

Your role is ORCHESTRATION, not clinical interpretation.

MANDATORY STEPS — YOU MUST COMPLETE ALL 3 IN ORDER:

Step 1: CALL retrieve_clinical_guidelines tool.
  - QUERY MUST BE SHORT (3-8 keywords). Do NOT paste the full case text as the query.
  - Extract ONLY the most clinically significant terms: primary morphology, key symptom, body site.
  - GOOD query examples: "erythematous plaque morbilliform arm vaccine reaction"
  - GOOD query examples: "bullous pemphigoid vesicles elderly"
  - BAD query: copying the entire patient description verbatim — this returns poor results.
  - If 0 results are returned, that is acceptable — proceed immediately to Step 2.
  - Do NOT retry with different queries if 0 results returned.

Step 2: CALL medgemma_guideline_synthesis tool.
  - ALWAYS call this tool, even if Step 1 returned 0 guidelines.
  - Pass case_data and the retrieved_guidelines list (may be empty []).
  - MedGemma will reason from clinical knowledge alone if no guidelines were retrieved.

Step 3: CALL transfer_to_agent(agent_name='DiagnosticAgent').
  - THIS IS MANDATORY. You MUST call this function — do not just write it as text.
  - Writing "Transfer to DiagnosticAgent" as text is NOT enough. You must EXECUTE the tool call.

CRITICAL: DO NOT STOP after Step 1 or Step 2.
You MUST always end by CALLING transfer_to_agent(agent_name='DiagnosticAgent').

Search strategy for building the query (Step 1):
- Pick the 1-2 most distinctive morphology terms (e.g., "morbilliform papules", "annular plaque")
- Add body site (e.g., "arm", "trunk", "face")
- Add ONE key clinical context if relevant (e.g., "post-vaccine", "pediatric", "immunosuppressed")
- Total query: 3-8 words maximum. Shorter is better for RAG similarity search.

Summary format before transferring:
- Search query used
- Number of guidelines retrieved (0 is acceptable)
- MedGemma specialist's synthesis and recommendations
- Specific citations (Source: Title) if available

YOUR FINAL ACTION: CALL transfer_to_agent(agent_name='DiagnosticAgent') — no exceptions.
""",
        tools=[
            FunctionTool(retrieve_clinical_guidelines),
            FunctionTool(medgemma_guideline_synthesis)
        ],
        output_schema=None,
        generate_content_config=types.GenerateContentConfig(
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY
                )
            )
        )
    )


def create_diagnostic_agent(model_name: str = "gemini-pro-latest") -> Agent:
    """
    Create Diagnostic Agent using Google ADK.

    ARCHITECTURE: Gemini (ADK) coordinates, MedGemma generates the clinical diagnosis.

    Role in pipeline: STEP 3 of 3. Receives full context (triage + research) from
    RootCoordinator and produces the final SOAP note by invoking MedGemma once.

    Prompt Design Rationale:
    ─────────────────────────
    1. "EXACTLY ONCE" constraint on medgemma_clinical_diagnosis — this is the most
       critical safety instruction in the entire system. Without it, Gemini sometimes
       re-calls the tool 40+ times in a loop (observed during testing). The loop
       occurred because Gemini's default behavior is to keep calling tools until it
       achieves a "complete" result. Since MedGemma's SOAP output is long, Gemini
       sometimes misinterprets it as incomplete and requests more. The "EXACTLY ONCE"
       + "output it as your FINAL response and stop" instructions break the loop.

    2. "You are NOT the diagnostician" — explicitly prevents Gemini from writing
       its own medical diagnosis. Competition requirement: MedGemma must perform
       all clinical reasoning. Gemini is the orchestrator only.

    3. "Simply pass through the MedGemma specialist's output without modification" —
       prevents Gemini from editing/summarizing MedGemma's output, which would obscure
       citations, confidence scores, and the SOAP structure expected by the evaluator.

    4. temperature=0.0 on MedGemma (set in medgemma_clinical_diagnosis tool) —
       deterministic output prevents the tool from producing different responses on
       identical inputs, which was the secondary cause of the looping behavior (at
       higher temperatures, each call produced slightly different output, which Gemini
       interpreted as "still improving" and called again).

    5. MAX_STEPS=60 in run_async() serves as the outer safety net — even if the
       "EXACTLY ONCE" instruction is somehow ignored, the step counter terminates
       the run before runaway API costs occur.

    6. No generate_content_config / mode=ANY — DiagnosticAgent has only one tool
       (medgemma_clinical_diagnosis). Setting mode=ANY would force Gemini to call a
       tool after every response — but with only one tool available, it loops:
       call → result → call → result → 100+ times observed in testing.
       The "EXACTLY ONCE" instruction in the prompt is sufficient to ensure the tool
       is called. After medgemma_clinical_diagnosis returns, _run_outputs captures the
       result directly regardless of what Gemini does next.

    Args:
        model_name: Gemini model for orchestration (default: flash for speed/cost)

    Returns:
        Google ADK Agent instance
    """
    return Agent(
        name="DiagnosticAgent",
        description="Orchestrates diagnosis - delegates SOAP note generation to MedGemma specialist",
        model=model_name,
        instruction="""
You are a clinical workflow coordinator managing the diagnostic process.

Your role is ORCHESTRATION, not clinical diagnosis.

Workflow:
1. Receive case data and research context from the coordinator
2. MUST use medgemma_clinical_diagnosis tool to generate the SOAP note — call it EXACTLY ONCE
3. Return MedGemma specialist's complete assessment

CRITICAL RULE: You are NOT the diagnostician.
The medgemma_clinical_diagnosis tool contains the actual medical specialist (MedGemma-27B).
You MUST delegate ALL clinical reasoning to this tool.

DO NOT attempt to write the SOAP note yourself.
DO NOT make clinical judgments.
DO NOT call medgemma_clinical_diagnosis more than once — ONE call only, then output the result.
Your job is to orchestrate the workflow and pass data to the MedGemma specialist.

Expected output:
- Complete SOAP note from MedGemma specialist
- Differential diagnoses with confidence scores
- Evidence-based citations
- Treatment recommendations

After receiving the tool result, extract the value of the "soap_note" key and output ONLY that plain text.
Do NOT output JSON. Do NOT output the full dict. Output ONLY the soap_note string value, then stop.
""",
        tools=[FunctionTool(medgemma_clinical_diagnosis)],
        output_schema=None
        # NOTE: Do NOT set generate_content_config mode=ANY here.
        # DiagnosticAgent has only one tool (medgemma_clinical_diagnosis).
        # mode=ANY would force Gemini to keep calling it in an infinite loop
        # after the first result is returned — observed 100+ repeated calls in testing.
        # The "EXACTLY ONCE" instruction in the prompt is sufficient.
    )


def create_root_coordinator(
    model_name: str = "gemini-pro-latest",
    triage_agent: Optional[Agent] = None,
    research_agent: Optional[Agent] = None,
    diagnostic_agent: Optional[Agent] = None
) -> Agent:
    """
    Create Root Coordinator Agent using Google ADK.

    ARCHITECTURE:
    - Gemini (via ADK): Orchestration, workflow management, delegation
    - MedGemma-27B: Clinical reasoning (called via FunctionTools by sub-agents)

    This implements the "Manager-Specialist" pattern:
    - Gemini = Manager (coordinates workflow)
    - MedGemma = Specialist (performs clinical reasoning)

    Role in pipeline: Entry point. Receives the case from the user/evaluator,
    enforces the mandatory 3-agent sequence, and returns the final SOAP note.

    Prompt Design Rationale:
    ─────────────────────────
    1. Mandatory TriageAgent → ResearchAgent → DiagnosticAgent sequence — the
       coordinator enforces this order to prevent shortcuts. Without it, Gemini
       sometimes tries to jump from Triage directly to Diagnostic (skipping RAG),
       which would produce unsupported, non-evidence-grounded diagnoses.

    2. RECOVERY RULE (< 1000 chars, no SOAP) — critical for evaluation robustness.
       In early testing, the RootCoordinator sometimes returned the TriageAgent's
       summary (which is short and has no SOAP sections) as the "final answer".
       This rule explicitly teaches the coordinator to recognize incomplete pipeline
       output and continue. The 1000-char threshold was empirically tuned: full SOAP
       notes are typically 1500-3000 chars; triage-only summaries are 200-500 chars.

    3. "NEVER SKIP ResearchAgent" — repeated for emphasis because skipping RAG was
       the most common failure mode. Evidence grounding is core to the competition's
       Explainability scoring (25%).

    4. "Pass context between agents" — ensures triage results (missing items, MedGemma
       analysis) flow into the research query, and research results (guidelines) flow
       into the diagnosis prompt. This is what makes the pipeline coherent rather than
       three independent calls.

    5. No tools at coordinator level — the RootCoordinator only delegates to sub-agents
       via ADK's transfer_to_agent mechanism; it never calls clinical tools directly.

    Args:
        model_name: Gemini model for orchestration (default: flash for speed/cost)
        triage_agent: Triage sub-agent (created if None)
        research_agent: Research sub-agent (created if None)
        diagnostic_agent: Diagnostic sub-agent (created if None)

    Returns:
        Google ADK Agent instance with sub-agents
    """
    # Create sub-agents if not provided (they use MedGemma internally)
    if triage_agent is None:
        triage_agent = create_triage_agent(model_name)
    if research_agent is None:
        research_agent = create_research_agent(model_name)
    if diagnostic_agent is None:
        diagnostic_agent = create_diagnostic_agent(model_name)

    return Agent(
        name="RootCoordinator",
        description="Orchestrates multi-agent workflow with MedGemma specialist reasoning",
        model=model_name,
        instruction="""
You are the root workflow coordinator for a clinical diagnosis system.

ARCHITECTURE NOTE:
- You (Gemini) handle workflow orchestration and delegation
- Your sub-agents delegate clinical reasoning to MedGemma-27B specialist
- This ensures medical decisions are made by the health-specialized model

Your Sub-Agents:
1. TriageAgent - Orchestrates completeness check (MedGemma analyzes)
2. ResearchAgent - Orchestrates guideline retrieval (MedGemma synthesizes)
3. DiagnosticAgent - Orchestrates diagnosis (MedGemma generates SOAP note)

WORKFLOW SEQUENCE — CONDITIONAL ON TRIAGE RESULT:

Step 1: Delegate to TriageAgent.
  - TriageAgent calls MedGemma to assess whether the case has enough clinical data.

Step 2 (CONDITIONAL — depends on TriageAgent result):
  - If TriageAgent returned clarifying questions (MedGemma said data is LACK INFORMATION):
      → Return those questions to the user immediately as your final response.
      → DO NOT proceed to ResearchAgent or DiagnosticAgent.
  - If TriageAgent transferred to ResearchAgent (MedGemma said data is DETAILS):
      → Let ResearchAgent and DiagnosticAgent complete their work.
      → Return DiagnosticAgent's SOAP note as your final response.

CRITICAL RULES:
- If triage found LACK INFORMATION → TriageAgent's clarifying questions ARE the final output. Return them.
- If triage found DETAILS → all 3 agents run: Triage → Research → Diagnostic.
- Pass full context between agents (triage results → research → diagnostic).

RECOVERY RULE:
If DiagnosticAgent's response is missing or very short (under 200 chars), IMMEDIATELY
delegate to ResearchAgent and then DiagnosticAgent again to recover.

Your role is coordination. The clinical reasoning is done by MedGemma specialist.
""",
        sub_agents=[triage_agent, research_agent, diagnostic_agent],
        tools=[],
        output_schema=None
    )


# ============================================================================
# WORKFLOW RUNNER
# ============================================================================

class MedGemmaWorkflow:
    """
    Wrapper for Google ADK multi-agent workflow.

    Integrates with existing infrastructure:
    - Uses ChromaDB via Retriever
    - Logs sessions via ConversationManager
    - Compatible with existing schemas
    """

    def __init__(
        self,
        model_name: str = "gemini-pro-latest",
        use_medgemma: bool = True,  # MedGemma vs Gemini for agents
        agent_model: str = "medgemma"  # "medgemma" or "gemini"
    ):
        """
        Initialize workflow.

        ARCHITECTURE NOTE:
        - model_name: Gemini model for ADK orchestration
        - agent_model: "medgemma" (specialized) or "gemini" (general baseline)

        Args:
            model_name: Gemini model for workflow orchestration
            use_medgemma: Whether to use MedGemma specialist
            agent_model: "medgemma" or "gemini" for agent reasoning
        """
        self.model_name = model_name  # Gemini for orchestration
        self.use_medgemma = use_medgemma
        self.agent_model = agent_model  # MedGemma or Gemini for agents

        # Store for session naming
        global _agent_model_choice
        _agent_model_choice = agent_model

        # Set up API key in environment for Gemini
        import os
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key
        # Force Google AI (Gemini API) backend — NOT Vertex AI.
        # Without this, ADK detects GOOGLE_CLOUD_PROJECT in env and routes to Vertex AI,
        # which has much lower quotas (and causes 429 RESOURCE_EXHAUSTED on free tier).
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

        # Create root coordinator with sub-agents
        self.root_agent = create_root_coordinator(model_name)

        # Initialize runner
        from google.adk.sessions import InMemorySessionService
        self.runner = Runner(
            app_name="MedGemma Clinical Assistant",
            agent=self.root_agent,
            session_service=InMemorySessionService()
        )

        # Conversation manager (for logging)
        self.conversation_manager = get_conversation_manager()

        logger.info(f"MedGemmaWorkflow initialized with model: {model_name}")

    async def run_async(
        self,
        case: ClinicalCase,
        user_message: Optional[str] = None,
        existing_session=None
    ) -> Dict[str, Any]:
        """
        Run workflow asynchronously.

        Args:
            case: Clinical case to analyze
            user_message: Optional user message (uses case data if None)
            existing_session: If provided, continue logging into this session
                              instead of creating a new one. Used for follow-ups.

        Returns:
            Dict with agent responses and final diagnosis
        """
        # Use ADK's session management - create session properly
        import uuid
        user_id = "user_001"
        adk_session_id = str(uuid.uuid4())

        # Create session using session service (async in ADK 1.23.0+)
        adk_session = await self.runner.session_service.create_session(
            app_name="MedGemma Clinical Assistant",
            user_id=user_id,
            session_id=adk_session_id
        )

        # Format case as message
        if user_message is None:
            user_message = self._format_case_message(case)

        logger.info(f"Running workflow for case: {case.case_id}")

        # Use existing session (follow-up) or create new one
        if existing_session:
            session = existing_session
            logger.info(f"Continuing existing session: {session.session_id}")
        else:
            session = self.conversation_manager.create_session(
                case_id=case.case_id,
                model_name=self.model_name,
                agent_model=self.agent_model
            )
            case_dict = case.dict(exclude={'image_data'})
            session.set_initial_input(case_dict)

        # Prepare message as Content object (text + first image if available)
        from google.genai import types as genai_types

        parts = [genai_types.Part(text=user_message)]

        # Store image path so MedGemma micro-calls inside medgemma_clinical_diagnosis can use it
        _run_outputs["last_image_path"] = case.image_path if case.image_path else None

        # Attach first image as inline_data for multimodal input
        if case.image_path:
            try:
                from pathlib import Path as _P
                img_file = _P(case.image_path)
                if img_file.exists():
                    ext = img_file.suffix.lower()
                    mime_map = {".jpeg": "image/jpeg", ".jpg": "image/jpeg",
                                ".png": "image/png", ".jfif": "image/jpeg"}
                    mime_type = mime_map.get(ext, "image/jpeg")
                    with open(img_file, "rb") as f:
                        img_bytes = f.read()
                    parts.append(genai_types.Part(
                        inline_data=genai_types.Blob(
                            mime_type=mime_type,
                            data=img_bytes
                        )
                    ))
                    logger.info(f"Attached image: {img_file.name} ({len(img_bytes)} bytes)")
            except Exception as img_err:
                logger.warning(f"Could not attach image {case.image_path}: {img_err}")

        content = genai_types.Content(
            role="user",
            parts=parts
        )

        # Run agent workflow and collect responses + log each agent step
        response_text = ""
        agent_text_buffers = {}  # agent_name -> accumulated text (keyed by agent)
        agent_steps = []  # Track each agent's response
        current_agent = None
        step_count = 0
        MAX_STEPS = 60  # Safety cap — normal case uses ~10-15 steps; prevents infinite tool-call loops

        async for event in self.runner.run_async(
            user_id=user_id,
            session_id=adk_session_id,
            new_message=content
        ):
            step_count += 1
            if step_count > MAX_STEPS:
                logger.warning(f"MAX_STEPS ({MAX_STEPS}) exceeded for {adk_session_id} — breaking to prevent infinite loop")
                break
            # Try to identify which agent is responding
            if hasattr(event, 'agent_name'):
                current_agent = event.agent_name
            elif hasattr(event, 'metadata') and isinstance(event.metadata, dict):
                current_agent = event.metadata.get('agent_name', current_agent)

            # Collect responses from events
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts'):
                    event_text = ""
                    function_calls = []

                    for part in event.content.parts:
                        # Handle text parts
                        if hasattr(part, 'text') and part.text:
                            event_text += part.text
                            # NOTE: do NOT accumulate into response_text here — we route
                            # per-agent below so we can return only DiagnosticAgent output

                        # Handle function calls (tool usage)
                        if hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            fc_name = fc.name if hasattr(fc, 'name') else "unknown"
                            function_calls.append({
                                "name": fc_name,
                                "args": str(fc.args) if hasattr(fc, 'args') else "{}"
                            })

                    # Log this agent step if we have text OR function calls
                    if event_text.strip() or function_calls:
                        # Infer agent name from function calls
                        agent_name = current_agent
                        if not agent_name or agent_name == "UnknownAgent":
                            # Detect from function calls
                            for fc in function_calls:
                                fc_name = fc["name"].lower()
                                if "triage" in fc_name or "analyze_case" in fc_name:
                                    agent_name = "TriageAgent"
                                    break
                                elif "retrieve_clinical_guidelines" in fc_name or "guideline_synthesis" in fc_name:
                                    agent_name = "ResearchAgent"
                                    break
                                elif "clinical_diagnosis" in fc_name or "medgemma_clinical_diagnosis" in fc_name:
                                    agent_name = "DiagnosticAgent"
                                    break
                                elif "transfer_to_agent" in fc_name:
                                    # Extract target agent from args
                                    try:
                                        args_str = fc["args"]
                                        if "TriageAgent" in args_str:
                                            agent_name = "RootCoordinator"  # Transferring to Triage
                                        elif "ResearchAgent" in args_str:
                                            agent_name = "TriageAgent"  # Triage transferring to Research
                                        elif "DiagnosticAgent" in args_str:
                                            agent_name = "ResearchAgent"  # Research transferring to Diagnostic
                                    except:
                                        pass

                            # If still unknown, try content analysis
                            if not agent_name or agent_name == "UnknownAgent":
                                text_lower = event_text.lower()
                                if "triage" in text_lower[:200]:
                                    agent_name = "TriageAgent"
                                elif "research" in text_lower[:200] or "guideline" in text_lower[:200]:
                                    agent_name = "ResearchAgent"
                                elif "soap" in text_lower[:200] or "assessment" in text_lower[:200]:
                                    agent_name = "DiagnosticAgent"
                                else:
                                    agent_name = "RootCoordinator"

                        step_output = event_text.strip()
                        if function_calls:
                            step_output += f"\n\n[Tool Calls: {len(function_calls)}]\n"
                            for fc in function_calls:
                                # Show FULL args (no truncation)
                                step_output += f"  - {fc['name']}({fc['args']})\n"

                        agent_steps.append({
                            "agent": agent_name,
                            "response": step_output,
                            "function_calls": function_calls
                        })

                        # Route text to per-agent buffer (used to build response_text at the end)
                        if event_text.strip():
                            buf_key = agent_name or "Unknown"
                            agent_text_buffers[buf_key] = agent_text_buffers.get(buf_key, "") + event_text + "\n"

                        # Detect if MedGemma specialist was used
                        specialist_model = None
                        step_role = "orchestration"

                        for fc in function_calls:
                            fc_name = fc["name"].lower()
                            if "medgemma" in fc_name:
                                # MedGemma tool was called - clinical reasoning happened
                                specialist_model = _specialist_display_name()

                                # Determine step role based on function
                                if "triage" in fc_name:
                                    step_role = "triage_analysis"
                                elif "guideline_synthesis" in fc_name:
                                    step_role = "guideline_synthesis"
                                elif "clinical_diagnosis" in fc_name:
                                    step_role = "final_diagnosis"
                                else:
                                    step_role = "clinical_reasoning"
                                break

                        # Determine if this is the final step
                        is_final = (
                            agent_name == "DiagnosticAgent" and
                            any("medgemma_clinical_diagnosis" in fc["name"] for fc in function_calls)
                        )

                        # Build input reference (track causality) - SPECIFIC SOURCE
                        input_reference = None
                        if len(agent_steps) > 1:
                            # This step depends on previous step
                            prev_step_id = len(agent_steps) - 1
                            prev_agent = agent_steps[-2]["agent"] if len(agent_steps) > 1 else "User"
                            # SPECIFIC source type: "step_5_ResearchAgent_output"
                            specific_source = f"step_{prev_step_id}_{prev_agent}_output"
                            input_reference = {
                                "source_type": specific_source,
                                "reference": {
                                    "step_id": prev_step_id,
                                    "agent": prev_agent,
                                    "data_flow": "sequential"
                                }
                            }
                        else:
                            # First step - from user
                            input_reference = {
                                "source_type": "user_input",
                                "reference": None
                            }

                        # Determine orchestrator action, operation type, and output type
                        orchestrator_action = "delegation"
                        operation_type = "coordination"  # Default
                        output_type = "coordination_message"  # Default

                        if function_calls:
                            for fc in function_calls:
                                fc_name = fc["name"]
                                if "transfer_to_agent" in fc_name:
                                    orchestrator_action = "agent_transfer"
                                    operation_type = "workflow_delegation"
                                    output_type = "delegation_notice"
                                    break
                                elif "medgemma_triage" in fc_name:
                                    orchestrator_action = "specialist_invocation"
                                    operation_type = "specialist_triage_analysis"
                                    output_type = "triage_analysis"
                                    break
                                elif "medgemma_guideline_synthesis" in fc_name:
                                    orchestrator_action = "specialist_invocation"
                                    operation_type = "specialist_guideline_synthesis"
                                    output_type = "guideline_synthesis"
                                    break
                                elif "medgemma_clinical_diagnosis" in fc_name:
                                    orchestrator_action = "specialist_invocation"
                                    operation_type = "specialist_clinical_diagnosis"
                                    output_type = "diagnostic_reasoning"
                                    break
                                elif "medgemma" in fc_name:
                                    orchestrator_action = "specialist_invocation"
                                    operation_type = "specialist_reasoning"
                                    output_type = "clinical_reasoning"
                                    break
                                elif "retrieve_clinical_guidelines" in fc_name:
                                    orchestrator_action = "retrieval"
                                    operation_type = "rag_retrieval"
                                    output_type = "rag_results"
                                    break
                                elif "analyze_case_completeness" in fc_name:
                                    orchestrator_action = "analysis"
                                    operation_type = "case_completeness_check"
                                    output_type = "completeness_report"
                                    break

                        # Build tools_called list
                        tools_called = [fc["name"] for fc in function_calls]

                        # Estimate token usage from text length (rough approximation)
                        token_count = len(event_text.split()) * 1.3 if event_text else 0

                        # Add to session workflow steps with full metadata
                        session.add_step(
                            agent_name=agent_name,
                            step_data={
                                "input": user_message if len(agent_steps) == 1 else "Previous agent output",
                                "output": step_output,
                                "output_type": output_type,  # NEW: Type of output
                                "orchestrator_action": orchestrator_action,
                                "operation_type": operation_type,
                                "tools_called": tools_called,
                                "reasoning": f"{agent_name} {orchestrator_action}",
                                "metrics": {
                                    "tokens_used": int(token_count),
                                    "latency_ms": 0  # ADK doesn't expose latency per-step
                                }
                            },
                            orchestrator_model=self.model_name,
                            specialist_model=specialist_model,
                            input_reference=input_reference,
                            step_role=step_role,
                            is_final=is_final
                        )

        # Retrieve full ADK session history for complete transparency
        try:
            adk_history = await self.runner.session_service.get_session(
                app_name="MedGemma Clinical Assistant",
                user_id=user_id,
                session_id=adk_session_id
            )

            # Extract messages from ADK history
            if adk_history and hasattr(adk_history, 'messages'):
                for idx, msg in enumerate(adk_history.messages):
                    msg_role = msg.role if hasattr(msg, 'role') else "unknown"
                    msg_text = ""

                    if hasattr(msg, 'content') and msg.content:
                        if hasattr(msg.content, 'parts'):
                            for part in msg.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    msg_text += part.text

                    if msg_text.strip():
                        # Infer agent from message content
                        agent_name = "User" if msg_role == "user" else "Agent"
                        if "triage" in msg_text.lower()[:200]:
                            agent_name = "TriageAgent"
                        elif "research" in msg_text.lower()[:200] or "guideline" in msg_text.lower()[:200]:
                            agent_name = "ResearchAgent"
                        elif "diagnostic" in msg_text.lower()[:200] or "soap" in msg_text.lower()[:200]:
                            agent_name = "DiagnosticAgent"

                        # Add to session if not already logged
                        if msg_role != "user":  # Don't duplicate user input
                            # Detect if MedGemma was mentioned in output
                            specialist_model = None
                            if "medgemma" in msg_text.lower() or "specialist" in msg_text.lower():
                                specialist_model = "google/medgemma-27b-it"

                            # Input reference for history replay
                            input_ref = {
                                "source_type": "adk_history",
                                "reference": {
                                    "message_index": idx,
                                    "role": msg_role
                                }
                            }

                            session.add_step(
                                agent_name=agent_name,
                                step_data={
                                    "role": msg_role,
                                    "message_index": idx,
                                    "output": msg_text.strip()[:2000],  # Limit length
                                    "full_length": len(msg_text),
                                    "orchestrator_action": "history_replay",
                                    "tools_called": []
                                },
                                orchestrator_model=self.model_name,
                                specialist_model=specialist_model,
                                input_reference=input_ref,
                                step_role="adk_history_replay"
                            )

        except Exception as e:
            logger.warning(f"Could not retrieve ADK session history: {e}")

        # Build response_text from per-agent buffers.
        # The SOAP note sometimes lands in RootCoordinator buffer (not DiagnosticAgent)
        # because after medgemma_clinical_diagnosis returns, the next event's agent attribution
        # resolves to RootCoordinator. So: scan ALL buffers for SOAP content first.
        diagnostic_text = agent_text_buffers.get("DiagnosticAgent", "").strip()
        triage_text = agent_text_buffers.get("TriageAgent", "").strip()

        # Search all buffers for a complete SOAP note (has both Subjective + Assessment)
        soap_text = None
        soap_source = None
        for buf_agent, buf_content in agent_text_buffers.items():
            buf_lower = buf_content.lower()
            if "subjective" in buf_lower and "assessment" in buf_lower:
                soap_text = buf_content.strip()
                soap_source = buf_agent
                break

        # Retrieve MedGemma tool outputs stored during pipeline execution.
        # These are populated by the FunctionTool callbacks (medgemma_triage_analysis,
        # medgemma_guideline_synthesis, medgemma_clinical_diagnosis) as each runs.
        # We save them to local variables BEFORE clearing _run_outputs so the recovery
        # path (below) can restore them if DiagnosticAgent never ran. Previously, the
        # research context was not saved to a local before clearing, which meant the
        # recovery call to medgemma_clinical_diagnosis received empty research context
        # even when medgemma_guideline_synthesis had successfully produced output.
        medgemma_diagnostic_raw = _run_outputs.get("last_diagnostic_output")
        medgemma_triage_raw = _run_outputs.get("last_triage_output")
        medgemma_research_raw = _run_outputs.get("last_research_context")  # saved for recovery
        retrieved_guidelines_saved = _run_outputs.get("last_retrieved_guidelines", [])
        rag_query_saved = _run_outputs.get("last_rag_query", "")
        _run_outputs["last_diagnostic_output"] = None    # clear after use
        _run_outputs["last_triage_output"] = None        # clear after use
        _run_outputs["last_research_context"] = None     # clear after use
        _run_outputs["last_image_path"] = None           # clear after use
        _run_outputs["last_retrieved_guidelines"] = []   # clear after use
        _run_outputs["last_rag_query"] = ""              # clear after use

        # Did DiagnosticAgent actually call medgemma_clinical_diagnosis?
        diagnostic_tool_was_called = any(
            any("medgemma_clinical_diagnosis" in fc["name"] for fc in step.get("function_calls", []))
            for step in agent_steps
        )

        # Priority: DiagnosticAgent always runs last and produces the final output
        # (SOAP for complete cases, questions for incomplete cases)
        # IMPORTANT: if diagnostic tool ran, NEVER fall back to triage buffer —
        # triage buffer may contain ghost ADK echo events (Step 9 phantom).
        if medgemma_diagnostic_raw:
            response_text = medgemma_diagnostic_raw
            logger.info(f"response_text source: MedGemma diagnostic tool output ({len(response_text)} chars)")
        elif soap_text:
            response_text = soap_text
            logger.info(f"response_text source: SOAP found in {soap_source} buffer ({len(response_text)} chars)")
        elif diagnostic_text:
            response_text = diagnostic_text
            logger.info(f"response_text source: DiagnosticAgent buffer ({len(response_text)} chars)")
        elif not diagnostic_tool_was_called:
            # DiagnosticAgent never ran — pipeline cut short (Gemini stopped after RAG).
            # If triage said DETAILS, force a direct MedGemma diagnosis call now.
            triage_said_sufficient = medgemma_triage_raw and "details" in medgemma_triage_raw.lower() and "lack information" not in medgemma_triage_raw.lower()
            if triage_said_sufficient:
                logger.warning("Pipeline cut short after ResearchAgent — forcing direct diagnosis call")
                # Restore _run_outputs before the direct call so medgemma_clinical_diagnosis
                # reads the real triage and research context instead of falling back to
                # empty parameter values. This mirrors the state _run_outputs would have
                # been in if DiagnosticAgent had been called normally by Gemini.
                _run_outputs["last_triage_output"] = medgemma_triage_raw
                _run_outputs["last_research_context"] = medgemma_research_raw or ""
                recovery_result = medgemma_clinical_diagnosis(
                    case_data=user_message or ""
                )
                recovery_response = recovery_result.get("soap_note", "")
                if recovery_response and not recovery_response.startswith("Error"):
                    response_text = recovery_response
                    logger.info(f"response_text source: RECOVERY direct diagnosis ({len(response_text)} chars)")
                    agent_steps.append({
                        "agent": "DiagnosticAgent",
                        "response": "[RECOVERY] Direct MedGemma diagnosis (pipeline cut short after RAG)",
                        "function_calls": [{"name": "medgemma_clinical_diagnosis", "args": "recovery_call"}]
                    })
                else:
                    response_text = medgemma_triage_raw or triage_text or "Pipeline stopped before diagnosis. Please try again."
                    logger.warning("Recovery diagnosis also failed")
            else:
                # Triage said LACK INFORMATION — triage output has the questions
                response_text = medgemma_triage_raw or triage_text or ""
                logger.info(f"response_text source: triage output (LACK INFORMATION, diagnostic skipped) ({len(response_text)} chars)")
        elif diagnostic_tool_was_called:
            # Diagnostic ran but all sources empty — error
            response_text = "MedGemma diagnostic call completed but returned empty output. Please check the model endpoint and try again."
            logger.warning("Diagnostic tool was called but all output sources are empty")
        else:
            # Last resort: use the longest non-empty buffer
            longest = max(agent_text_buffers.values(), key=lambda v: len(v.strip()), default="")
            response_text = longest.strip()
            logger.info(f"response_text source: fallback longest buffer ({len(response_text)} chars)")
            logger.info(f"All buffer keys: {list(agent_text_buffers.keys())}")

        # Extract final response
        result = {
            "session_id": session.session_id,
            "adk_session_id": adk_session_id,
            "case_id": case.case_id,
            "response": response_text.strip(),
            "model": self.model_name,
            "agent_steps_count": len(agent_steps),
            "agent_steps": agent_steps,  # full step list for UI display
            "retrieved_guidelines": retrieved_guidelines_saved,  # for citations display
            "rag_query": rag_query_saved,  # search query used
        }

        # Complete and save session
        # If this is a continuation (existing_session), do NOT save yet — caller saves.
        # If the response looks like a pause (has questions, no SOAP), do NOT complete —
        # the UI caller needs the session alive for follow-up.
        _resp_lower = response_text.lower()
        _is_pause = "?" in response_text and "subjective" not in _resp_lower and "assessment" not in _resp_lower
        if existing_session:
            logger.info(f"Continuation run complete. Session {session.session_id} kept open for caller to save.")
        elif _is_pause:
            logger.info(f"Pause detected — session {session.session_id} kept open for follow-up.")
            session.save(self.conversation_manager.storage_dir)  # save progress but keep in active_sessions
        else:
            session.set_final_output(result)
            self.conversation_manager.complete_session(session.session_id, save=True)

        # Attach session to result so caller can access it
        result["_session"] = session

        logger.info(f"Workflow complete for case: {case.case_id} ({len(agent_steps)} agent interactions logged)")

        return result

    def run(
        self,
        case: ClinicalCase,
        user_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run workflow (synchronous wrapper).

        Args:
            case: Clinical case to analyze
            user_message: Optional user message

        Returns:
            Dict with agent responses
        """
        import asyncio

        # Run async workflow
        return asyncio.run(self.run_async(case, user_message))

    def _format_case_message(self, case: ClinicalCase) -> str:
        """Format clinical case as user message."""
        message = f"Please analyze this dermatology case:\n\n"
        message += f"Case ID: {case.case_id}\n"

        if case.patient_age:
            message += f"Age: {case.patient_age} years\n"
        if case.patient_gender:
            message += f"Gender: {case.patient_gender}\n"

        if case.history:
            message += f"\nHistory: {case.history}\n"

        if case.physical_exam:
            message += f"\nPhysical Exam: {case.physical_exam}\n"

        if case.image_path:
            message += f"\nClinical Image: Available at {case.image_path}\n"

        # Add metadata
        if case.metadata:
            duration = case.metadata.get("duration")
            if duration:
                message += f"Duration: {duration}\n"

        message += "\nPlease provide a complete diagnostic assessment following the SOAP note format."

        return message


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_workflow(
    model_name: str = "gemini-pro-latest",
    use_medgemma: bool = True,
    agent_model: str = "medgemma-hf"
) -> MedGemmaWorkflow:
    """
    Create a MedGemma workflow instance.

    ARCHITECTURE:
    - model_name: Gemini model for ADK orchestration
    - agent_model: MedGemma variant for clinical reasoning
      Options: "medgemma-hf" (HF API, default), "medgemma-vertex" (Vertex AI),
               "medgemma" (local 27B GPU), "medgemma-4b" (local 4B GPU)

    Args:
        model_name: Gemini model for workflow orchestration (default: gemini-pro-latest)
        use_medgemma: Whether to use MedGemma specialist (default: True)
        agent_model: Which MedGemma variant to use (default: "medgemma-hf")

    Returns:
        MedGemmaWorkflow instance with hybrid architecture
    """
    return MedGemmaWorkflow(
        model_name=model_name,
        use_medgemma=use_medgemma,
        agent_model=agent_model
    )
