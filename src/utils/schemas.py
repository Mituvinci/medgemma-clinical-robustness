"""
Data schemas for MedGemma Clinical Robustness Assistant.
Defines Pydantic models for structured data flow between agents.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ContextState(str, Enum):
    """Enumeration of clinical context completeness states."""
    ORIGINAL = "original"
    HISTORY_ONLY = "history_only"
    IMAGE_ONLY = "image_only"
    EXAM_ONLY = "exam_only"
    EXAM_RESTRICTED = "exam_restricted"


class MissingDataType(str, Enum):
    """Types of missing clinical data."""
    HISTORY = "history"
    EXAM = "exam"
    IMAGE = "image"
    AGE = "age"
    DURATION = "duration"
    SYMPTOMS = "symptoms"
    LOCATION = "location"


class ClinicalCase(BaseModel):
    """Represents a clinical dermatology case."""
    case_id: str
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None
    chief_complaint: Optional[str] = None
    history: Optional[str] = None
    physical_exam: Optional[str] = None
    image_path: Optional[str] = None
    image_data: Optional[Any] = None  # PIL Image or base64
    context_state: ContextState = ContextState.ORIGINAL
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TriageResult(BaseModel):
    """Output from the Triage Agent."""
    missing_data: List[MissingDataType] = Field(default_factory=list)
    has_sufficient_context: bool = False
    clarification_questions: List[str] = Field(default_factory=list)
    extracted_info: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""


class RetrievedDocument(BaseModel):
    """A document retrieved from ChromaDB."""
    content: str
    source: str
    title: Optional[str] = None
    section: Optional[str] = None
    similarity_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResearchResult(BaseModel):
    """Output from the Research Agent."""
    query: str
    retrieved_documents: List[RetrievedDocument] = Field(default_factory=list)
    summary: str = ""
    guideline_citations: List[str] = Field(default_factory=list)


class SOAPNote(BaseModel):
    """Structured SOAP note output."""
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""
    differential_diagnoses: List[str] = Field(default_factory=list)
    confidence_level: Optional[str] = None
    guideline_citations: List[str] = Field(default_factory=list)


class DiagnosticResult(BaseModel):
    """Final output from the Diagnostic Agent."""
    soap_note: SOAPNote
    primary_diagnosis: str
    differential_diagnoses: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    warnings: List[str] = Field(default_factory=list)


class AgentMessage(BaseModel):
    """Message passed between agents."""
    sender: str
    recipient: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[str] = None


class WorkflowState(BaseModel):
    """Overall state of the multi-agent workflow."""
    case: ClinicalCase
    triage_result: Optional[TriageResult] = None
    research_result: Optional[ResearchResult] = None
    diagnostic_result: Optional[DiagnosticResult] = None
    messages: List[AgentMessage] = Field(default_factory=list)
    current_step: str = "triage"
    is_complete: bool = False
