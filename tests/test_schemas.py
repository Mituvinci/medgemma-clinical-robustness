"""
Unit tests for data schemas and models.
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.schemas import (
    ClinicalCase,
    ContextState,
    MissingDataType,
    TriageResult,
    RetrievedDocument,
    ResearchResult,
    SOAPNote,
    DiagnosticResult,
    AgentMessage,
    WorkflowState
)


class TestClinicalCase:
    """Test ClinicalCase schema."""

    def test_create_minimal_case(self):
        """Test creating a case with minimal information."""
        case = ClinicalCase(case_id="TEST_001")
        assert case.case_id == "TEST_001"
        assert case.context_state == ContextState.ORIGINAL
        assert case.metadata == {}

    def test_create_full_case(self):
        """Test creating a case with all fields."""
        case = ClinicalCase(
            case_id="TEST_002",
            patient_age=45,
            patient_gender="Female",
            chief_complaint="Itchy rash",
            history="2-week history of rash",
            physical_exam="Erythematous lesions",
            context_state=ContextState.HISTORY_ONLY
        )
        assert case.patient_age == 45
        assert case.patient_gender == "Female"
        assert case.context_state == ContextState.HISTORY_ONLY


class TestTriageResult:
    """Test TriageResult schema."""

    def test_create_triage_result(self):
        """Test creating a triage result."""
        result = TriageResult(
            missing_data=[MissingDataType.IMAGE, MissingDataType.DURATION],
            has_sufficient_context=False,
            clarification_questions=["How long has the lesion been present?"],
            reasoning="Missing critical temporal information"
        )
        assert len(result.missing_data) == 2
        assert MissingDataType.IMAGE in result.missing_data
        assert not result.has_sufficient_context
        assert len(result.clarification_questions) == 1


class TestRetrievedDocument:
    """Test RetrievedDocument schema."""

    def test_create_retrieved_doc(self):
        """Test creating a retrieved document."""
        doc = RetrievedDocument(
            content="Allergic contact dermatitis is a delayed hypersensitivity reaction...",
            source="AAD Guidelines",
            title="Contact Dermatitis Management",
            section="Diagnosis",
            similarity_score=0.89
        )
        assert doc.similarity_score == 0.89
        assert doc.source == "AAD Guidelines"


class TestSOAPNote:
    """Test SOAPNote schema."""

    def test_create_soap_note(self):
        """Test creating a SOAP note."""
        soap = SOAPNote(
            subjective="Patient reports 2-week history of itchy rash on hands",
            objective="Erythematous vesicular lesions on dorsal hands",
            assessment="Allergic contact dermatitis, likely secondary to new detergent exposure",
            plan="Discontinue suspected allergen, topical corticosteroids, follow-up in 2 weeks",
            differential_diagnoses=["Allergic contact dermatitis", "Irritant contact dermatitis", "Dyshidrotic eczema"],
            confidence_level="High"
        )
        assert "contact dermatitis" in soap.assessment.lower()
        assert len(soap.differential_diagnoses) == 3


class TestDiagnosticResult:
    """Test DiagnosticResult schema."""

    def test_create_diagnostic_result(self):
        """Test creating a diagnostic result."""
        soap = SOAPNote(
            subjective="Test",
            objective="Test",
            assessment="Test",
            plan="Test"
        )
        result = DiagnosticResult(
            soap_note=soap,
            primary_diagnosis="Allergic contact dermatitis",
            confidence_score=0.85,
            reasoning="Classic presentation with allergen exposure"
        )
        assert result.confidence_score == 0.85
        assert 0.0 <= result.confidence_score <= 1.0

    def test_confidence_score_validation(self):
        """Test that confidence score is validated."""
        soap = SOAPNote(subjective="", objective="", assessment="", plan="")

        # This should raise a validation error for out-of-range score
        with pytest.raises(Exception):  # Pydantic ValidationError
            DiagnosticResult(
                soap_note=soap,
                primary_diagnosis="Test",
                confidence_score=1.5  # Invalid: > 1.0
            )


class TestWorkflowState:
    """Test WorkflowState schema."""

    def test_create_workflow_state(self):
        """Test creating a workflow state."""
        case = ClinicalCase(case_id="TEST_001")
        state = WorkflowState(
            case=case,
            current_step="triage",
            is_complete=False
        )
        assert state.current_step == "triage"
        assert not state.is_complete
        assert len(state.messages) == 0

    def test_workflow_progression(self):
        """Test workflow state progression."""
        case = ClinicalCase(case_id="TEST_001")
        triage = TriageResult(has_sufficient_context=True)

        state = WorkflowState(
            case=case,
            triage_result=triage,
            current_step="research"
        )
        assert state.triage_result is not None
        assert state.current_step == "research"
