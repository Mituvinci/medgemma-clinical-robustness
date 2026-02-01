"""
Google ADK Multi-Agent System for MedGemma Clinical Robustness Assistant

HYBRID ARCHITECTURE: Manager-Specialist Pattern
==============================================

This implements a sophisticated two-tier reasoning system:

TIER 1: Orchestration (Gemini via Google ADK)
- Framework: Google Agent Development Kit (ADK)
- Model: Gemini 1.5 Flash
- Role: Workflow management, delegation, coordination
- Handles: Agent routing, state management, tool orchestration

TIER 2: Clinical Reasoning (MedGemma-27B Specialist)
- Model: MedGemma-27B (via HuggingFace API)
- Role: Medical diagnosis, clinical analysis, SOAP generation
- Handles: All high-stakes medical reasoning tasks
- Invoked via: FunctionTools (medgemma_triage_analysis, medgemma_guideline_synthesis, medgemma_clinical_diagnosis)

Why This Architecture?
----------------------
1. Competition Requirement: Uses MedGemma for medical reasoning (mandatory)
2. Google ADK Integration: Demonstrates advanced agent orchestration
3. Best of Both Worlds: Gemini's orchestration + MedGemma's medical expertise
4. Production-Grade: Separates workflow management from domain expertise

Agent Workflow:
--------------
1. Triage Agent (ADK/Gemini orchestrates → MedGemma analyzes completeness)
2. Research Agent (ADK/Gemini orchestrates RAG → MedGemma synthesizes guidelines)
3. Diagnostic Agent (ADK/Gemini orchestrates → MedGemma generates SOAP note)
4. Root Coordinator (ADK/Gemini manages the entire workflow)

This is the "Clinical Specialist Pattern": Gemini is the clinic manager,
MedGemma is the specialist physician.
"""

import logging
from typing import Dict, Any, List, Optional
from google.adk import Agent, Runner
from google.adk.tools import FunctionTool
from google.genai import types

from src.rag.retriever import Retriever
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


# ============================================================================
# TOOLS - Functions that agents can call
# ============================================================================

# Initialize retriever (shared across agents)
_retriever = Retriever()

# MedGemma adapter (initialized lazily to avoid import-time errors)
from src.agents.models.medgemma_adapter import MedGemmaAdapter
_medgemma_specialist = None

def _get_medgemma_specialist():
    """Get or create MedGemma specialist (lazy initialization)."""
    global _medgemma_specialist
    if _medgemma_specialist is None:
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

    # Query ChromaDB
    retrieved_docs = _retriever.retrieve(
        query=query,
        n_results=n_results,
        min_similarity=0.5
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

    logger.info(f"Retrieved {len(guidelines)} guidelines (top similarity: {result['top_similarity']:.2f})")

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
    Analyze a clinical case for data completeness.

    Args:
        history: Patient history text
        physical_exam: Physical examination findings
        image_available: Whether clinical images are available
        patient_age: Patient age in years
        patient_gender: Patient gender
        duration: Duration of symptoms

    Returns:
        Dict with missing data analysis and clarification questions
    """
    logger.info("Tool called: analyze_case_completeness")

    missing_data = []
    questions = []

    # Check critical fields
    if not history or len(history.strip()) < 10:
        missing_data.append("history")
        questions.append("Can you provide the patient's chief complaint and symptom history?")

    if not physical_exam and not image_available:
        missing_data.append("physical_exam_or_image")
        questions.append("Can you describe the physical examination findings or provide clinical images?")

    if not patient_age:
        missing_data.append("patient_age")
        questions.append("What is the patient's age?")

    if not duration:
        missing_data.append("duration")
        questions.append("How long have the symptoms been present?")

    # Determine if we can proceed
    has_sufficient_context = (
        bool(history) and
        (bool(physical_exam) or image_available) and
        bool(patient_age)
    )

    result = {
        "missing_data": missing_data,
        "clarification_questions": questions,
        "has_sufficient_context": has_sufficient_context,
        "reasoning": _build_triage_reasoning(missing_data, has_sufficient_context)
    }

    logger.info(f"Triage analysis: {len(missing_data)} missing items, sufficient_context={has_sufficient_context}")

    return result


def _build_triage_reasoning(missing_data: List[str], has_sufficient: bool) -> str:
    """Build reasoning explanation for triage result."""
    if has_sufficient:
        return "Sufficient clinical data available to proceed with diagnosis. All critical fields present."
    else:
        missing_str = ", ".join(missing_data)
        return f"Cannot proceed with confident diagnosis. Missing critical data: {missing_str}. " \
               f"In dermatology, both patient history and visual assessment (exam or image) are essential."


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

    prompt = f"""You are an expert medical triage specialist analyzing a dermatology case.

Case Summary:
{case_summary}

Missing Data Identified:
{', '.join(missing_items) if missing_items else 'None - case appears complete'}

Your Task:
1. If data is missing: Generate specific, clinically relevant clarification questions
2. If data is complete: Confirm we can proceed with diagnosis
3. Explain WHY each missing item is critical for accurate dermatological diagnosis

Provide your analysis as a structured response."""

    try:
        specialist = _get_medgemma_specialist()
        response = specialist.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.3
        )

        return {
            "medgemma_analysis": response,
            "missing_items": missing_items,
            "specialist_used": "MedGemma-27B"
        }
    except Exception as e:
        logger.error(f"MedGemma triage analysis failed: {e}", exc_info=True)
        import traceback
        error_details = traceback.format_exc()
        return {
            "medgemma_analysis": f"Error calling MedGemma: {e}\n\nDetails:\n{error_details}",
            "missing_items": missing_items,
            "specialist_used": "MedGemma-27B (failed)"
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
            max_tokens=800,
            temperature=0.4
        )

        return {
            "medgemma_synthesis": response,
            "guidelines_count": len(retrieved_guidelines),
            "specialist_used": "MedGemma-27B"
        }
    except Exception as e:
        logger.error(f"MedGemma guideline synthesis failed: {e}", exc_info=True)
        import traceback
        error_details = traceback.format_exc()
        return {
            "medgemma_synthesis": f"Error calling MedGemma: {e}\n\nDetails:\n{error_details}",
            "guidelines_count": len(retrieved_guidelines),
            "specialist_used": "MedGemma-27B (failed)"
        }


def medgemma_clinical_diagnosis(
    case_data: str,
    research_context: str,
    triage_notes: str = ""
) -> Dict[str, Any]:
    """
    Call MedGemma specialist for final clinical diagnosis and SOAP note generation.

    This is the PRIMARY clinical reasoning task - delegates to MedGemma-27B.

    Args:
        case_data: Complete clinical case information
        research_context: Synthesized guidelines from research agent
        triage_notes: Notes from triage agent

    Returns:
        Dict with MedGemma's SOAP note and diagnosis
    """
    logger.info("Calling MedGemma Specialist for clinical diagnosis (SOAP note)")

    prompt = f"""You are an expert dermatologist providing a clinical assessment.

Triage Notes:
{triage_notes if triage_notes else 'Case cleared for diagnosis'}

Clinical Case Data:
{case_data}

Evidence-Based Research Context:
{research_context}

Your Task:
Generate a complete SOAP note (Subjective, Objective, Assessment, Plan) with:

**Subjective (S):**
- Patient history, chief complaint, demographics
- Symptom duration and progression

**Objective (O):**
- Physical examination findings
- Lesion characteristics (morphology, distribution, color, texture)

**Assessment (A):**
- Differential diagnoses ranked by likelihood
- Primary diagnosis with confidence score (0.0-1.0)
- Evidence from guidelines supporting each diagnosis
- Specific citations (e.g., "AAD Guidelines: Psoriasis Management")

**Plan (P):**
- Recommended diagnostic tests
- Treatment options per guidelines
- Follow-up recommendations

Confidence Scoring Guidelines:
- 0.9-1.0: Pathognomonic features, complete data, strong guideline match
- 0.7-0.89: Good data quality, solid guideline support
- 0.5-0.69: Some ambiguity or missing data
- Below 0.5: Significant uncertainty

Format your response as a structured SOAP note with clear section headers."""

    try:
        specialist = _get_medgemma_specialist()
        response = specialist.generate(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.3
        )

        return {
            "soap_note": response,
            "specialist_used": "MedGemma-27B",
            "reasoning_engine": "MedGemma-27B (Health-Specialized)"
        }
    except Exception as e:
        logger.error(f"MedGemma clinical diagnosis failed: {e}", exc_info=True)
        import traceback
        error_details = traceback.format_exc()
        return {
            "soap_note": f"Error calling MedGemma specialist: {e}\n\nDetails:\n{error_details}\n\nFallback: Unable to generate diagnosis.",
            "specialist_used": "MedGemma-27B (failed)",
            "reasoning_engine": "Error"
        }


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

def create_triage_agent(model_name: str = "gemini-pro-latest") -> Agent:
    """
    Create Triage Agent using Google ADK.

    ARCHITECTURE: Gemini (ADK) orchestrates, MedGemma provides clinical reasoning.

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

Workflow:
1. Use analyze_case_completeness tool to check for missing data
2. If missing data found: MUST use medgemma_triage_analysis tool to get specialist assessment
3. Return the MedGemma specialist's analysis to the coordinator

CRITICAL: You are NOT the medical expert. Always delegate clinical reasoning
to the MedGemma specialist via the medgemma_triage_analysis tool.

For dermatology cases, critical data includes:
- Patient history (chief complaint, symptoms)
- Physical examination OR clinical images
- Patient demographics (age, gender)
- Symptom duration

Output format:
- List missing items (if any)
- Include MedGemma specialist's clinical assessment
- Recommendation: Proceed or Request Clarification
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

Workflow:
1. Formulate effective search query from the clinical case
2. Use retrieve_clinical_guidelines tool to query ChromaDB (1,492 chunks)
3. MUST use medgemma_guideline_synthesis tool to have MedGemma interpret the guidelines
4. Return MedGemma specialist's synthesis to the coordinator

CRITICAL: You retrieve documents, but MedGemma (the medical specialist) interprets them.
Always delegate guideline synthesis to MedGemma via the medgemma_guideline_synthesis tool.

Search strategy:
- Extract key features: symptoms, location, morphology, patient demographics
- Focus queries on: differential diagnosis, diagnostic criteria, treatment

Output format:
- Search query used
- Number of guidelines retrieved
- MedGemma specialist's synthesis and recommendations
- Specific citations (Source: Title)
""",
        tools=[
            FunctionTool(retrieve_clinical_guidelines),
            FunctionTool(medgemma_guideline_synthesis)
        ],
        output_schema=None
    )


def create_diagnostic_agent(model_name: str = "gemini-pro-latest") -> Agent:
    """
    Create Diagnostic Agent using Google ADK.

    ARCHITECTURE: Gemini (ADK) coordinates, MedGemma generates the clinical diagnosis.

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
2. MUST use medgemma_clinical_diagnosis tool to generate the SOAP note
3. Return MedGemma specialist's complete assessment

CRITICAL RULE: You are NOT the diagnostician.
The medgemma_clinical_diagnosis tool contains the actual medical specialist (MedGemma-27B).
You MUST delegate ALL clinical reasoning to this tool.

DO NOT attempt to write the SOAP note yourself.
DO NOT make clinical judgments.
Your job is to orchestrate the workflow and pass data to the MedGemma specialist.

Expected output:
- Complete SOAP note from MedGemma specialist
- Differential diagnoses with confidence scores
- Evidence-based citations
- Treatment recommendations

Simply pass through the MedGemma specialist's output without modification.
""",
        tools=[FunctionTool(medgemma_clinical_diagnosis)],
        output_schema=None
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

Workflow:
1. Receive clinical case from user
2. Delegate to TriageAgent
   - If MedGemma specialist identifies missing data: ASK USER for clarification
   - If complete: Continue to next step
3. Delegate to ResearchAgent to retrieve and synthesize guidelines
4. Delegate to DiagnosticAgent for final SOAP note (MedGemma specialist)
5. Return the MedGemma specialist's SOAP note to user

CRITICAL RULES:
- ALWAYS run Triage first
- NEVER skip to diagnosis if missing critical data
- When missing data found, ASK THE USER (agentic pause)
- Sequential flow: Triage → Research → Diagnostic
- Pass context between agents

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
        use_medgemma: bool = True  # Always true - MedGemma is the specialist
    ):
        """
        Initialize workflow.

        ARCHITECTURE NOTE:
        - model_name: Gemini model for ADK orchestration (default: flash for cost/speed)
        - MedGemma-27B is ALWAYS used for clinical reasoning (via FunctionTools)

        Args:
            model_name: Gemini model for workflow orchestration
            use_medgemma: Whether to use MedGemma specialist (always True)
        """
        self.model_name = model_name  # Gemini for orchestration
        self.use_medgemma = use_medgemma  # MedGemma for clinical reasoning

        # Set up API key in environment for Gemini
        import os
        os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key

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
        user_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run workflow asynchronously.

        Args:
            case: Clinical case to analyze
            user_message: Optional user message (uses case data if None)

        Returns:
            Dict with agent responses and final diagnosis
        """
        # Use ADK's session management - create session properly
        import uuid
        user_id = "user_001"
        adk_session_id = str(uuid.uuid4())

        # Create session using session service (async)
        adk_session = await self.runner.session_service.create_session(
            app_name="MedGemma Clinical Assistant",
            user_id=user_id,
            session_id=adk_session_id
        )

        # Format case as message
        if user_message is None:
            user_message = self._format_case_message(case)

        logger.info(f"Running workflow for case: {case.case_id}")

        # Prepare message as Content object
        from google.genai import types as genai_types
        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)]
        )

        # Run agent workflow and collect responses
        response_text = ""
        async for event in self.runner.run_async(
            user_id=user_id,
            session_id=adk_session_id,
            new_message=content
        ):
            # Collect responses from events
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:  # Check not None
                            response_text += part.text + "\n"

        # Create our session for logging
        session = self.conversation_manager.create_session(
            case_id=case.case_id,
            model_name=self.model_name
        )
        session.set_initial_input(case.dict())

        # Extract final response
        result = {
            "session_id": session.session_id,
            "adk_session_id": adk_session_id,
            "case_id": case.case_id,
            "response": response_text.strip(),
            "model": self.model_name
        }

        # Complete session
        session.set_final_output(result)
        self.conversation_manager.complete_session(session.session_id, save=True)

        logger.info(f"Workflow complete for case: {case.case_id}")

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
    use_medgemma: bool = True
) -> MedGemmaWorkflow:
    """
    Create a MedGemma workflow instance.

    ARCHITECTURE:
    - model_name: Gemini model for ADK orchestration (flash recommended for cost/speed)
    - use_medgemma: Whether to use MedGemma-27B for clinical reasoning (always True)

    Args:
        model_name: Gemini model for workflow orchestration (default: gemini-1.5-flash)
        use_medgemma: Whether to use MedGemma specialist (default: True)

    Returns:
        MedGemmaWorkflow instance with hybrid architecture
    """
    return MedGemmaWorkflow(
        model_name=model_name,
        use_medgemma=use_medgemma
    )
