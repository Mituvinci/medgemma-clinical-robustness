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
_agent_model_choice = "medgemma"  # Global: "medgemma" or "gemini"

def _get_medgemma_specialist():
    """
    Get or create specialist for clinical reasoning (lazy initialization).

    Returns MedGemma (specialized) or Gemini (baseline) based on agent_model_choice.
    """
    global _medgemma_specialist, _agent_model_choice

    # Check if we should use Gemini instead of MedGemma
    if _agent_model_choice == "gemini":
        # Use Gemini for baseline comparison
        from src.agents.models.gemini_adapter import GeminiAdapter
        logger.info("🔄 Using Gemini Pro for agent reasoning (baseline comparison)")
        if _medgemma_specialist is None or not isinstance(_medgemma_specialist, GeminiAdapter):
            _medgemma_specialist = GeminiAdapter(
                model_id=settings.gemini_model_id,
                api_key=settings.gemini_api_key
            )
        return _medgemma_specialist

    # Use MedGemma (default/specialized)
    if _medgemma_specialist is None or not isinstance(_medgemma_specialist, MedGemmaAdapter):
        logger.info("🧬 Using MedGemma-27B-IT for agent reasoning (specialized)")
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
3. IMPORTANT: After completing triage, you MUST transfer to ResearchAgent
   - Use transfer_to_agent with agent_name='ResearchAgent'
   - DO NOT transfer directly to DiagnosticAgent
   - Research Agent must retrieve clinical guidelines before diagnosis

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
- NEXT STEP: Transfer to ResearchAgent (mandatory)
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
4. IMPORTANT: After completing research, you MUST transfer to DiagnosticAgent
   - Use transfer_to_agent with agent_name='DiagnosticAgent'
   - Pass the guideline synthesis results to diagnostic agent
5. Return MedGemma specialist's synthesis to the coordinator

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
- NEXT STEP: Transfer to DiagnosticAgent with research results
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

MANDATORY WORKFLOW SEQUENCE (DO NOT SKIP ANY STEP):
1. Receive clinical case from user
2. FIRST: Always delegate to TriageAgent for analysis
3. SECOND: Always delegate to ResearchAgent to retrieve clinical guidelines
   - This step is MANDATORY even if case seems simple
   - ResearchAgent must query ChromaDB for evidence-based guidelines
4. THIRD: Delegate to DiagnosticAgent for final SOAP note
   - DiagnosticAgent combines triage findings + research guidelines
5. Return the final SOAP note to user

CRITICAL RULES:
- YOU MUST FOLLOW THIS EXACT SEQUENCE: TriageAgent → ResearchAgent → DiagnosticAgent
- NEVER SKIP ResearchAgent - guideline retrieval is mandatory for evidence-based diagnosis
- NEVER let TriageAgent transfer directly to DiagnosticAgent
- Each agent must complete before moving to next
- Pass context between agents (triage results → research → diagnostic)

ERROR TO AVOID: Do NOT allow TriageAgent to transfer directly to DiagnosticAgent.
ResearchAgent MUST run between them to retrieve clinical guidelines.

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

        # Create our session for logging BEFORE workflow (so we can log steps)
        session = self.conversation_manager.create_session(
            case_id=case.case_id,
            model_name=self.model_name,
            agent_model=self.agent_model  # For meaningful filename (medgemma or gemini)
        )
        # Exclude image_data to avoid JSON serialization issues with bytes
        case_dict = case.dict(exclude={'image_data'})
        session.set_initial_input(case_dict)

        # Prepare message as Content object
        from google.genai import types as genai_types
        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)]
        )

        # Run agent workflow and collect responses + log each agent step
        response_text = ""
        agent_steps = []  # Track each agent's response
        current_agent = None

        async for event in self.runner.run_async(
            user_id=user_id,
            session_id=adk_session_id,
            new_message=content
        ):
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
                            response_text += part.text + "\n"

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

                        # Detect if MedGemma specialist was used
                        specialist_model = None
                        step_role = "orchestration"

                        for fc in function_calls:
                            fc_name = fc["name"].lower()
                            if "medgemma" in fc_name:
                                # MedGemma tool was called - clinical reasoning happened
                                specialist_model = "google/medgemma-27b-it"

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

        # Extract final response
        result = {
            "session_id": session.session_id,
            "adk_session_id": adk_session_id,
            "case_id": case.case_id,
            "response": response_text.strip(),
            "model": self.model_name,
            "agent_steps_count": len(agent_steps)
        }

        # Complete session
        session.set_final_output(result)
        self.conversation_manager.complete_session(session.session_id, save=True)

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
        use_medgemma=use_medgemma,
        agent_model="medgemma"  # Default: use MedGemma for agents
    )
