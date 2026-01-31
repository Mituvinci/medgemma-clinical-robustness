"""
Google ADK Multi-Agent System for MedGemma Clinical Robustness Assistant

Uses Google Agent Development Kit to build:
1. Triage Agent - Identifies missing clinical data
2. Research Agent - Retrieves guidelines from ChromaDB
3. Diagnostic Agent - Generates SOAP notes
4. Root Coordinator - Orchestrates the workflow
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
# AGENT DEFINITIONS
# ============================================================================

def create_triage_agent(model_name: str = "gemini-2.0-flash-exp") -> Agent:
    """
    Create Triage Agent using Google ADK.

    Analyzes clinical cases to identify missing data and determine
    if sufficient context exists for diagnosis.

    Args:
        model_name: Gemini model to use

    Returns:
        Google ADK Agent instance
    """
    return Agent(
        name="TriageAgent",
        description="Analyzes clinical cases to identify missing critical information",
        model=model_name,
        instruction="""
You are a medical triage specialist analyzing dermatology cases.

Your responsibilities:
1. Analyze the provided clinical case for completeness
2. Identify what critical data is missing (history, exam, age, duration, etc.)
3. Generate specific clarification questions to obtain missing information
4. Determine if there is sufficient context to proceed with diagnosis

For dermatology cases, you MUST have:
- Patient history (chief complaint, symptoms, onset)
- Physical examination findings OR clinical images
- Patient demographics (age, gender)
- Symptom duration

Use the analyze_case_completeness tool to perform your analysis.

If critical data is missing, explain WHY it's needed for accurate diagnosis.
If sufficient data is present, confirm that we can proceed.

Format your response as a structured analysis with:
- Missing data items (if any)
- Specific questions to ask (if data missing)
- Recommendation on whether to proceed
- Clinical reasoning for your assessment
""",
        tools=[FunctionTool(analyze_case_completeness)],
        output_schema=None  # Free-form output
    )


def create_research_agent(model_name: str = "gemini-2.0-flash-exp") -> Agent:
    """
    Create Research Agent using Google ADK.

    Queries ChromaDB to retrieve relevant clinical guidelines based
    on the case presentation.

    Args:
        model_name: Gemini model to use

    Returns:
        Google ADK Agent instance
    """
    return Agent(
        name="ResearchAgent",
        description="Retrieves evidence-based clinical guidelines from medical literature",
        model=model_name,
        instruction="""
You are a medical research specialist with access to dermatology clinical guidelines.

Your responsibilities:
1. Formulate effective search queries based on the clinical case
2. Retrieve relevant guidelines from the knowledge base (AAD, StatPearls)
3. Summarize key evidence-based recommendations
4. Provide citations for all retrieved guidelines

When given a clinical case:
1. Extract key clinical features (symptoms, location, morphology, demographics)
2. Use the retrieve_clinical_guidelines tool with an effective query
3. Review retrieved guidelines for relevance
4. Summarize the most pertinent diagnostic criteria and treatment recommendations

Focus on retrieving guidelines that help with:
- Differential diagnosis
- Diagnostic criteria
- Clinical features
- Treatment recommendations

Format your response with:
- Search query used
- Number of guidelines retrieved
- Summary of key findings from top guidelines
- Specific citations (Source: Title)
""",
        tools=[FunctionTool(retrieve_clinical_guidelines)],
        output_schema=None  # Free-form output
    )


def create_diagnostic_agent(model_name: str = "gemini-2.0-flash-exp") -> Agent:
    """
    Create Diagnostic Agent using Google ADK.

    Synthesizes clinical case data and retrieved guidelines to generate
    a structured SOAP note with evidence-based diagnosis.

    Args:
        model_name: Gemini model to use

    Returns:
        Google ADK Agent instance
    """
    return Agent(
        name="DiagnosticAgent",
        description="Generates evidence-based clinical diagnoses with structured SOAP notes",
        model=model_name,
        instruction="""
You are an expert dermatologist creating clinical assessments.

Your responsibilities:
1. Synthesize patient history, physical findings, and clinical guidelines
2. Generate a structured SOAP note (Subjective, Objective, Assessment, Plan)
3. Provide differential diagnoses ranked by likelihood
4. Cite evidence from clinical guidelines
5. Assign confidence score based on data quality

SOAP Note Format:

**Subjective (S):**
- Patient's chief complaint and history
- Relevant demographics (age, gender)
- Symptom duration and progression

**Objective (O):**
- Physical examination findings
- Lesion morphology, distribution, characteristics
- Note if exam/images unavailable

**Assessment (A):**
- Differential diagnoses ranked by likelihood
- Primary diagnosis with supporting evidence
- Alternative diagnoses with brief rationale
- Cite specific guidelines that support each diagnosis
- Confidence level (0.0-1.0) based on:
  - Data completeness (higher if exam + history)
  - Guideline support (higher if strong evidence match)
  - Specificity of findings

**Plan (P):**
- Recommended next steps
- Diagnostic tests if needed
- Treatment options per guidelines
- Follow-up recommendations

Confidence Scoring:
- 0.9-1.0: Complete data, clear guideline match, pathognomonic features
- 0.7-0.89: Good data, strong guideline support
- 0.5-0.69: Some missing data or ambiguous presentation
- 0.3-0.49: Significant data gaps
- 0.0-0.29: Insufficient data for diagnosis

Always cite specific guidelines when making diagnostic or treatment recommendations.

Format your response as a complete SOAP note with all four sections clearly labeled.
Include confidence score and list of cited guidelines at the end.
""",
        tools=[],  # No tools - synthesizes information
        output_schema=None  # Free-form SOAP note
    )


def create_root_coordinator(
    model_name: str = "gemini-2.0-flash-exp",
    triage_agent: Optional[Agent] = None,
    research_agent: Optional[Agent] = None,
    diagnostic_agent: Optional[Agent] = None
) -> Agent:
    """
    Create Root Coordinator Agent using Google ADK.

    Orchestrates the multi-agent workflow:
    1. Triage → Check for missing data
    2. Research → Retrieve guidelines
    3. Diagnostic → Generate SOAP note

    Args:
        model_name: Gemini model to use
        triage_agent: Triage sub-agent (created if None)
        research_agent: Research sub-agent (created if None)
        diagnostic_agent: Diagnostic sub-agent (created if None)

    Returns:
        Google ADK Agent instance with sub-agents
    """
    # Create sub-agents if not provided
    if triage_agent is None:
        triage_agent = create_triage_agent(model_name)
    if research_agent is None:
        research_agent = create_research_agent(model_name)
    if diagnostic_agent is None:
        diagnostic_agent = create_diagnostic_agent(model_name)

    return Agent(
        name="RootCoordinator",
        description="Orchestrates multi-agent clinical diagnosis workflow",
        model=model_name,
        instruction="""
You are the root coordinator managing a team of medical specialists.

Your team:
1. TriageAgent - Checks for missing clinical data
2. ResearchAgent - Retrieves evidence-based guidelines
3. DiagnosticAgent - Generates final SOAP note diagnosis

Workflow:
1. Receive clinical case from user
2. Delegate to TriageAgent to check data completeness
   - If missing data: Ask user for clarification (DO NOT proceed)
   - If complete: Continue to next step
3. Delegate to ResearchAgent to retrieve relevant guidelines
4. Delegate to DiagnosticAgent to synthesize diagnosis
5. Return final SOAP note to user

Important:
- ALWAYS run Triage first to check data completeness
- NEVER skip to diagnosis if Triage finds missing critical data
- When Triage identifies missing data, ASK THE USER for that information
- Only proceed to Research and Diagnostic steps if Triage confirms sufficient context
- Coordinate the agents sequentially (Triage → Research → Diagnostic)

When delegating:
- Provide each agent with relevant case information
- Pass Research findings to Diagnostic agent
- Ensure Diagnostic agent has access to both case data and guidelines

Your final output should be the complete SOAP note from the Diagnostic agent.
""",
        sub_agents=[triage_agent, research_agent, diagnostic_agent],
        tools=[],  # Coordinator delegates, doesn't use tools directly
        output_schema=None  # Free-form orchestration
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
        model_name: str = "gemini-2.0-flash-exp",
        use_medgemma: bool = False
    ):
        """
        Initialize workflow.

        Args:
            model_name: Gemini model to use
            use_medgemma: Whether to use MedGemma (requires setup)
        """
        self.model_name = model_name
        self.use_medgemma = use_medgemma

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
        # Use ADK's session management - create new session each time
        import uuid
        user_id = "user_001"
        adk_session_id = str(uuid.uuid4())

        # Create session in ADK's session service
        from google.adk.sessions import Session
        adk_session = Session(user_id=user_id, session_id=adk_session_id)
        self.runner.session_service.create_session(adk_session)

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
                        if hasattr(part, 'text'):
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
    model_name: str = "gemini-2.0-flash-exp",
    use_medgemma: bool = False
) -> MedGemmaWorkflow:
    """
    Create a MedGemma workflow instance.

    Args:
        model_name: Model to use (default: gemini-2.0-flash-exp)
        use_medgemma: Whether to use MedGemma model

    Returns:
        MedGemmaWorkflow instance
    """
    return MedGemmaWorkflow(
        model_name=model_name,
        use_medgemma=use_medgemma
    )
