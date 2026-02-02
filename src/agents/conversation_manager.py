"""
Conversation Manager

Tracks complete workflow sessions with full audit trail for:
- Explainability (competition judging criteria)
- Medical compliance (PII-safe logging)
- Debugging and analysis
- Model comparison

This is critical for the competition's explainability scoring (25% of total).
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from src.utils.logger import setup_logger, pii_filter
from config.config import settings

logger = setup_logger(__name__)


class ConversationSession:
    """
    Represents a single workflow session.

    Tracks all interactions, agent decisions, and retrieved evidence
    for complete transparency and explainability.
    """

    def __init__(
        self,
        case_id: str,
        model_name: str,
        session_id: Optional[str] = None,
        agent_model: Optional[str] = None
    ):
        """
        Initialize conversation session.

        Args:
            case_id: Unique case identifier
            model_name: LLM model being used
            session_id: Optional session ID (auto-generated if None)
            agent_model: Agent model type (medgemma or gemini) for file naming
        """
        self.case_id = case_id
        self.model_name = model_name
        self.agent_model = agent_model or "medgemma"
        self.session_id = session_id or self._generate_session_id()

        self.timestamp_start = datetime.utcnow().isoformat() + "Z"
        self.timestamp_end = None

        # Workflow tracking
        self.workflow_steps: List[Dict[str, Any]] = []
        self.current_step = 0

        # Input/output
        self.initial_input: Optional[Dict] = None
        self.final_output: Optional[Dict] = None

        # Metrics
        self.total_tokens = 0
        self.total_latency_ms = 0

        # Flags
        self.pii_redacted = False
        self.completed = False

        logger.info(
            f"Started conversation session: {self.session_id} "
            f"(case={case_id}, model={model_name})"
        )

    def _generate_session_id(self) -> str:
        """
        Generate meaningful session ID with auto-incrementing run number.

        Format: case_{run_number}_{agent_model}_{timestamp}
        Example: case_001_medgemma_20260201_014419

        Auto-increments by checking existing files in logs/sessions/ directory.
        If case_001, case_002, ..., case_005 exist, generates case_006.
        """
        # Get logs directory
        logs_dir = Path(settings.log_dir) / "sessions"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Find existing case_*.json files
        existing_files = list(logs_dir.glob("case_*.json"))

        # Extract run numbers from existing files
        max_run_number = 0
        for filepath in existing_files:
            filename = filepath.stem  # Get filename without .json extension
            # Parse filename: case_XXX_model_timestamp
            parts = filename.split("_")
            if len(parts) >= 2:
                try:
                    # Try to parse the second part as run number
                    run_number = int(parts[1])
                    max_run_number = max(max_run_number, run_number)
                except ValueError:
                    # Skip files that don't match expected format
                    continue

        # Next run number
        next_run_number = max_run_number + 1

        # Generate session ID
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"case_{next_run_number:03d}_{self.agent_model}_{timestamp}"

    def set_initial_input(self, input_data: Dict[str, Any]):
        """
        Set initial case input.

        Args:
            input_data: Initial case data
        """
        # Apply PII filtering
        filtered = pii_filter(json.dumps(input_data))
        self.initial_input = json.loads(filtered)
        self.pii_redacted = True

    def _get_provider(self, model_name: Optional[str]) -> Optional[str]:
        """
        Determine provider from model name.

        Args:
            model_name: Model identifier

        Returns:
            Provider name or None
        """
        if not model_name:
            return None
        model_lower = model_name.lower()
        if "gemini" in model_lower:
            return "google_genai"
        if "medgemma" in model_lower or "google/" in model_name:
            return "huggingface_transformers"
        if "gpt" in model_lower or "openai" in model_lower:
            return "openai"
        if "claude" in model_lower:
            return "anthropic"
        return "unknown"

    def add_step(
        self,
        agent_name: str,
        step_data: Dict[str, Any],
        orchestrator_model: str = "gemini-pro-latest",
        specialist_model: Optional[str] = None,
        input_reference: Optional[Dict[str, Any]] = None,
        step_role: str = "standard",
        is_final: bool = False
    ):
        """
        Add workflow step with explicit model tracking and causality.

        Implements audit-grade logging with:
        - Dual-model tracking (orchestrator vs specialist)
        - Input causality (references, not duplication)
        - Constrained decision rationale
        - Trust metadata for verification

        Args:
            agent_name: Name of agent
            step_data: Step data (execution, output, metrics, etc.)
            orchestrator_model: Model used for workflow coordination
            specialist_model: Model used for clinical reasoning (if any)
            input_reference: Reference to previous step(s)
            step_role: Role of this step (triage, research, diagnostic, final_resolution)
            is_final: Whether this is the final resolution step
        """
        self.current_step += 1

        step = {
            "step_id": self.current_step,
            "agent": agent_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",

            # Model tracking (CORE TRUST ANCHOR)
            "models": {
                "orchestrator": {
                    "name": orchestrator_model,
                    "provider": self._get_provider(orchestrator_model),
                    "role": "workflow_coordination"
                },
                "specialist": {
                    "name": specialist_model,
                    "provider": self._get_provider(specialist_model),
                    "role": "clinical_reasoning"
                } if specialist_model else None
            },

            # Input causality (NO DUPLICATION)
            "input": {
                "source_type": input_reference.get("source_type", "unknown") if input_reference else "user_input",
                "reference": input_reference.get("reference") if input_reference else None,
                "summary": step_data.get("input_summary", "")
            },

            # Execution details
            "execution": {
                "orchestrator_action": step_data.get("orchestrator_action", "unknown"),
                "tools_called": step_data.get("tools_called", [])
            },

            # Output
            "output": {
                "type": step_data.get("output_type", "unknown"),
                **step_data.get("output", {}),

                # Clear attribution
                "produced_by": {
                    "assembled_by": "orchestrator",
                    "clinical_content_by": specialist_model,
                    "reasoning_source_step": input_reference.get("reference", {}).get("step_id") if input_reference and input_reference.get("reference") else None
                } if is_final else (
                    "specialist" if specialist_model else "orchestrator"
                )
            },

            # Constrained rationale (not free-form reasoning)
            "decision_rationale": {
                "summary": step_data.get("decision_rationale_summary", ""),
                "clinical_basis": step_data.get("clinical_basis", ""),
                "guideline_reference": step_data.get("guideline_reference")
            } if step_data.get("decision_rationale_summary") else None,

            # Step metadata
            "step_metadata": {
                "step_role": step_role,
                "is_final_resolution": is_final,
                "next_step_suggestion": step_data.get("next_step"),
                **step_data.get("metadata", {})
            },

            # Trust metadata (VERIFICATION ANCHOR)
            "trust_metadata": {
                "clinical_reasoning_by_specialist": specialist_model is not None,
                "specialist_model": specialist_model,
                "orchestrator_clinical_role": "none" if specialist_model else "coordinator"
            },

            # Metrics with model breakdown
            "metrics": {
                "orchestrator_tokens": step_data.get("orchestrator_tokens", 0),
                "specialist_tokens": step_data.get("specialist_tokens", 0),
                "total_tokens": step_data.get("tokens_used", 0),
                "orchestrator_latency_ms": step_data.get("orchestrator_latency_ms", 0),
                "specialist_latency_ms": step_data.get("specialist_latency_ms", 0),
                "total_latency_ms": step_data.get("latency_ms", 0)
            }
        }

        # Apply PII filtering to entire step
        filtered = pii_filter(json.dumps(step))
        step = json.loads(filtered)

        self.workflow_steps.append(step)

        # Update totals
        self.total_tokens += step["metrics"]["total_tokens"]
        self.total_latency_ms += step["metrics"]["total_latency_ms"]

        logger.debug(f"Added step {self.current_step}: {agent_name} (role: {step_role})")

    def set_final_output(self, output_data: Dict[str, Any]):
        """
        Set final workflow output.

        Args:
            output_data: Final diagnostic result
        """
        # Apply PII filtering
        filtered = pii_filter(json.dumps(output_data))
        self.final_output = json.loads(filtered)

    def complete(self):
        """Mark session as completed."""
        self.timestamp_end = datetime.utcnow().isoformat() + "Z"
        self.completed = True
        logger.info(f"Completed session: {self.session_id}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary.

        Returns:
            Complete session data as dict
        """
        return {
            "session_id": self.session_id,
            "case_id": self.case_id,
            "model": self.model_name,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "completed": self.completed,

            "initial_input": self.initial_input,
            "workflow_steps": self.workflow_steps,
            "final_output": self.final_output,

            "metadata": {
                "total_steps": self.current_step,
                "total_tokens": self.total_tokens,
                "total_latency_ms": self.total_latency_ms,
                "pii_redacted": self.pii_redacted
            }
        }

    def to_json(self) -> str:
        """
        Convert session to JSON string.

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_txt(self) -> str:
        """
        Convert session to human-readable plain text report.

        Returns:
            Plain text report
        """
        lines = []
        lines.append("=" * 70)
        lines.append("MEDGEMMA CLINICAL WORKFLOW SESSION REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Session ID:    {self.session_id}")
        lines.append(f"Case ID:       {self.case_id}")
        lines.append(f"Model:         {self.model_name}")
        lines.append(f"Started:       {self.timestamp_start}")
        lines.append(f"Ended:         {self.timestamp_end or 'In Progress'}")
        lines.append(f"Status:        {'Completed' if self.completed else 'In Progress'}")
        lines.append("")

        # Initial Input
        lines.append("-" * 70)
        lines.append("INITIAL INPUT")
        lines.append("-" * 70)
        if self.initial_input:
            for key, value in self.initial_input.items():
                if key != 'image_data':  # Skip binary data
                    lines.append(f"{key}: {value}")
        lines.append("")

        # Workflow Steps
        lines.append("-" * 70)
        lines.append("WORKFLOW STEPS")
        lines.append("-" * 70)
        for step in self.workflow_steps:
            step_id = step.get('step_id', step.get('step', '?'))
            agent = step.get('agent', 'UnknownAgent')
            step_role = step.get('step_metadata', {}).get('step_role', 'unknown')
            is_final = step.get('step_metadata', {}).get('is_final_resolution', False)

            lines.append(f"\nStep {step_id}: {agent} [{step_role}]" + (" [FINAL]" if is_final else ""))
            lines.append(f"Timestamp: {step.get('timestamp', 'N/A')}")

            # Model tracking
            if 'models' in step:
                models = step['models']
                orch = models.get('orchestrator', {})
                spec = models.get('specialist')

                lines.append(f"\nModels Used:")
                lines.append(f"  Orchestrator: {orch.get('name', 'N/A')} (role: {orch.get('role', 'N/A')})")
                if spec:
                    lines.append(f"  Specialist:   {spec.get('name', 'N/A')} (role: {spec.get('role', 'N/A')})")
                else:
                    lines.append(f"  Specialist:   None (orchestrator-only step)")

            # Input reference
            if 'input' in step:
                input_info = step['input']
                lines.append(f"\nInput:")
                lines.append(f"  Source: {input_info.get('source_type', 'N/A')}")
                if input_info.get('reference'):
                    ref = input_info['reference']
                    lines.append(f"  Reference: Step {ref.get('step_id')} ({ref.get('agent')})")
                lines.append(f"  Summary: {input_info.get('summary', 'N/A')}")

            # Decision rationale
            if step.get('decision_rationale'):
                rationale = step['decision_rationale']
                lines.append(f"\nDecision Rationale:")
                lines.append(f"  {rationale.get('summary', 'N/A')}")
                if rationale.get('clinical_basis'):
                    lines.append(f"  Clinical Basis: {rationale['clinical_basis']}")

            # Output
            if step.get('output'):
                lines.append(f"\nOutput:")
                output = step['output']
                lines.append(f"  Type: {output.get('type', 'N/A')}")

                # Show who produced it
                produced_by = output.get('produced_by')
                if isinstance(produced_by, dict):
                    lines.append(f"  Assembled by: {produced_by.get('assembled_by', 'N/A')}")
                    lines.append(f"  Clinical content by: {produced_by.get('clinical_content_by', 'N/A')}")
                else:
                    lines.append(f"  Produced by: {produced_by}")

            # Metrics
            if step.get('metrics'):
                metrics = step['metrics']
                lines.append(f"\nMetrics:")
                lines.append(f"  Orchestrator: {metrics.get('orchestrator_tokens', 0)} tokens, {metrics.get('orchestrator_latency_ms', 0)} ms")
                lines.append(f"  Specialist:   {metrics.get('specialist_tokens', 0)} tokens, {metrics.get('specialist_latency_ms', 0)} ms")
                lines.append(f"  Total:        {metrics.get('total_tokens', 0)} tokens, {metrics.get('total_latency_ms', 0)} ms")

            lines.append("")

        # Final Output
        if self.final_output:
            lines.append("-" * 70)
            lines.append("FINAL OUTPUT")
            lines.append("-" * 70)
            for key, value in self.final_output.items():
                if isinstance(value, dict):
                    lines.append(f"{key}:")
                    for k, v in value.items():
                        lines.append(f"  {k}: {v}")
                else:
                    lines.append(f"{key}: {value}")
            lines.append("")

        # Summary Metrics
        lines.append("-" * 70)
        lines.append("SUMMARY METRICS")
        lines.append("-" * 70)

        # Calculate model-specific totals
        orch_tokens = sum(s.get('metrics', {}).get('orchestrator_tokens', 0) for s in self.workflow_steps)
        spec_tokens = sum(s.get('metrics', {}).get('specialist_tokens', 0) for s in self.workflow_steps)
        orch_latency = sum(s.get('metrics', {}).get('orchestrator_latency_ms', 0) for s in self.workflow_steps)
        spec_latency = sum(s.get('metrics', {}).get('specialist_latency_ms', 0) for s in self.workflow_steps)

        lines.append(f"Total Steps:         {self.current_step}")
        lines.append(f"")
        lines.append(f"Orchestrator Tokens: {orch_tokens}")
        lines.append(f"Specialist Tokens:   {spec_tokens}")
        lines.append(f"Total Tokens:        {self.total_tokens}")
        lines.append(f"")
        lines.append(f"Orchestrator Time:   {orch_latency} ms ({orch_latency/1000:.2f} sec)")
        lines.append(f"Specialist Time:     {spec_latency} ms ({spec_latency/1000:.2f} sec)")
        lines.append(f"Total Latency:       {self.total_latency_ms} ms ({self.total_latency_ms/1000:.2f} sec)")
        lines.append(f"PII Redacted:        {self.pii_redacted}")
        lines.append("")

        # Trust Verification Summary
        lines.append("-" * 70)
        lines.append("TRUST VERIFICATION SUMMARY")
        lines.append("-" * 70)

        clinical_steps = [s for s in self.workflow_steps if s.get('models', {}).get('specialist')]
        orchestrator_only = [s for s in self.workflow_steps if not s.get('models', {}).get('specialist')]
        final_steps = [s for s in self.workflow_steps if s.get('step_metadata', {}).get('is_final_resolution')]

        lines.append(f"Clinical Reasoning Steps:     {len(clinical_steps)}")
        lines.append(f"Orchestration-Only Steps:     {len(orchestrator_only)}")
        lines.append(f"Final Resolution Steps:       {len(final_steps)}")
        lines.append(f"")

        if clinical_steps:
            specialist_model = clinical_steps[0].get('models', {}).get('specialist', {}).get('name', 'unknown')
            lines.append(f"Clinical Reasoning Performed By: {specialist_model}")

        if final_steps:
            final_step = final_steps[0]
            produced_by = final_step.get('output', {}).get('produced_by', {})
            if isinstance(produced_by, dict):
                lines.append(f"Final Answer Assembled By:       {produced_by.get('assembled_by', 'N/A')}")
                lines.append(f"Final Clinical Content By:       {produced_by.get('clinical_content_by', 'N/A')}")

        lines.append("")
        lines.append("=" * 70)

        return '\n'.join(lines)

    def to_markdown(self) -> str:
        """
        Convert session to human-readable Markdown report.

        Returns:
            Markdown formatted report
        """
        lines = []
        lines.append("# MedGemma Clinical Workflow Session Report")
        lines.append("")
        lines.append("## Session Information")
        lines.append("")
        lines.append(f"- **Session ID:** `{self.session_id}`")
        lines.append(f"- **Case ID:** `{self.case_id}`")
        lines.append(f"- **Model:** {self.model_name}")
        lines.append(f"- **Started:** {self.timestamp_start}")
        lines.append(f"- **Ended:** {self.timestamp_end or 'In Progress'}")
        lines.append(f"- **Status:** {'✅ Completed' if self.completed else '⏳ In Progress'}")
        lines.append("")

        # Initial Input
        lines.append("## Initial Input")
        lines.append("")
        if self.initial_input:
            lines.append("```json")
            filtered_input = {k: v for k, v in self.initial_input.items() if k != 'image_data'}
            lines.append(json.dumps(filtered_input, indent=2))
            lines.append("```")
        lines.append("")

        # Workflow Steps
        lines.append("## Workflow Steps")
        lines.append("")
        for step in self.workflow_steps:
            step_id = step.get('step_id', step.get('step', '?'))
            agent = step.get('agent', 'UnknownAgent')
            step_role = step.get('step_metadata', {}).get('step_role', 'unknown')
            is_final = step.get('step_metadata', {}).get('is_final_resolution', False)

            final_badge = " 🎯 **FINAL**" if is_final else ""
            lines.append(f"### Step {step_id}: {agent} `[{step_role}]`{final_badge}")
            lines.append(f"**Timestamp:** {step.get('timestamp', 'N/A')}")
            lines.append("")

            # Model tracking
            if 'models' in step:
                models = step['models']
                orch = models.get('orchestrator', {})
                spec = models.get('specialist')

                lines.append("**Models Used:**")
                lines.append(f"- **Orchestrator:** `{orch.get('name', 'N/A')}` ({orch.get('role', 'N/A')})")
                if spec:
                    lines.append(f"- **Specialist:** `{spec.get('name', 'N/A')}` ({spec.get('role', 'N/A')})")
                else:
                    lines.append(f"- **Specialist:** None (orchestrator-only step)")
                lines.append("")

            # Input reference
            if 'input' in step:
                input_info = step['input']
                lines.append("**Input:**")
                lines.append(f"- Source: `{input_info.get('source_type', 'N/A')}`")
                if input_info.get('reference'):
                    ref = input_info['reference']
                    lines.append(f"- Reference: Step {ref.get('step_id')} ({ref.get('agent')})")
                lines.append(f"- Summary: {input_info.get('summary', 'N/A')}")
                lines.append("")

            # Decision rationale
            if step.get('decision_rationale'):
                rationale = step['decision_rationale']
                lines.append("**Decision Rationale:**")
                lines.append(f"> {rationale.get('summary', 'N/A')}")
                if rationale.get('clinical_basis'):
                    lines.append(f">")
                    lines.append(f"> Clinical Basis: {rationale['clinical_basis']}")
                lines.append("")

            # Output
            if step.get('output'):
                output = step['output']
                lines.append("**Output:**")
                lines.append(f"- Type: `{output.get('type', 'N/A')}`")

                # Show who produced it
                produced_by = output.get('produced_by')
                if isinstance(produced_by, dict):
                    lines.append(f"- Assembled by: {produced_by.get('assembled_by', 'N/A')}")
                    lines.append(f"- Clinical content by: `{produced_by.get('clinical_content_by', 'N/A')}`")
                else:
                    lines.append(f"- Produced by: {produced_by}")
                lines.append("")

            # Metrics
            if step.get('metrics'):
                metrics = step['metrics']
                lines.append("**Metrics:**")
                lines.append("")
                lines.append("| Model | Tokens | Latency |")
                lines.append("|-------|--------|---------|")
                lines.append(f"| Orchestrator | {metrics.get('orchestrator_tokens', 0)} | {metrics.get('orchestrator_latency_ms', 0)} ms |")
                lines.append(f"| Specialist | {metrics.get('specialist_tokens', 0)} | {metrics.get('specialist_latency_ms', 0)} ms |")
                lines.append(f"| **Total** | **{metrics.get('total_tokens', 0)}** | **{metrics.get('total_latency_ms', 0)} ms** |")
                lines.append("")

        # Final Output
        if self.final_output:
            lines.append("## Final Output")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(self.final_output, indent=2))
            lines.append("```")
            lines.append("")

        # Summary Metrics
        lines.append("## Summary Metrics")
        lines.append("")

        # Calculate model-specific totals
        orch_tokens = sum(s.get('metrics', {}).get('orchestrator_tokens', 0) for s in self.workflow_steps)
        spec_tokens = sum(s.get('metrics', {}).get('specialist_tokens', 0) for s in self.workflow_steps)
        orch_latency = sum(s.get('metrics', {}).get('orchestrator_latency_ms', 0) for s in self.workflow_steps)
        spec_latency = sum(s.get('metrics', {}).get('specialist_latency_ms', 0) for s in self.workflow_steps)

        lines.append("| Metric | Orchestrator | Specialist | Total |")
        lines.append("|--------|--------------|------------|-------|")
        lines.append(f"| Tokens | {orch_tokens} | {spec_tokens} | {self.total_tokens} |")
        lines.append(f"| Latency (ms) | {orch_latency} | {spec_latency} | {self.total_latency_ms} |")
        lines.append(f"| Latency (sec) | {orch_latency/1000:.2f} | {spec_latency/1000:.2f} | {self.total_latency_ms/1000:.2f} |")
        lines.append("")
        lines.append(f"- **Total Steps:** {self.current_step}")
        lines.append(f"- **PII Redacted:** {'Yes' if self.pii_redacted else 'No'}")
        lines.append("")

        # Trust Verification Summary
        lines.append("## Trust Verification Summary")
        lines.append("")

        clinical_steps = [s for s in self.workflow_steps if s.get('models', {}).get('specialist')]
        orchestrator_only = [s for s in self.workflow_steps if not s.get('models', {}).get('specialist')]
        final_steps = [s for s in self.workflow_steps if s.get('step_metadata', {}).get('is_final_resolution')]

        lines.append(f"- **Clinical Reasoning Steps:** {len(clinical_steps)}")
        lines.append(f"- **Orchestration-Only Steps:** {len(orchestrator_only)}")
        lines.append(f"- **Final Resolution Steps:** {len(final_steps)}")
        lines.append("")

        if clinical_steps:
            specialist_model = clinical_steps[0].get('models', {}).get('specialist', {}).get('name', 'unknown')
            lines.append(f"**Clinical Reasoning Performed By:** `{specialist_model}`")
            lines.append("")

        if final_steps:
            final_step = final_steps[0]
            produced_by = final_step.get('output', {}).get('produced_by', {})
            if isinstance(produced_by, dict):
                lines.append(f"**Final Answer:**")
                lines.append(f"- Assembled by: {produced_by.get('assembled_by', 'N/A')}")
                lines.append(f"- Clinical content by: `{produced_by.get('clinical_content_by', 'N/A')}`")
                lines.append("")

        lines.append("---")
        lines.append("")
        lines.append(f"*Generated: {datetime.utcnow().isoformat()}Z*")

        return '\n'.join(lines)

    def save(self, directory: Optional[Path] = None):
        """
        Save session to JSON, TXT, and Markdown files.

        Creates three files with the same base name:
        - {session_id}.json - Machine-readable ground truth
        - {session_id}.txt  - Human-readable plain text report
        - {session_id}.md   - Human-readable Markdown report

        Args:
            directory: Directory to save to (default: logs/sessions/)

        Returns:
            Path to JSON file
        """
        if directory is None:
            directory = Path(settings.log_dir) / "sessions"

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        base_path = directory / self.session_id

        # Save JSON (ground truth)
        json_path = base_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

        # Save TXT (human-readable)
        txt_path = base_path.with_suffix('.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(self.to_txt())

        # Save Markdown (human-readable)
        md_path = base_path.with_suffix('.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self.to_markdown())

        logger.info(f"Saved session to:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  TXT:  {txt_path}")
        logger.info(f"  MD:   {md_path}")

        return json_path

    @classmethod
    def load(cls, session_id: str, directory: Optional[Path] = None) -> 'ConversationSession':
        """
        Load session from JSON file.

        Args:
            session_id: Session ID to load
            directory: Directory to load from

        Returns:
            ConversationSession instance
        """
        if directory is None:
            directory = Path(settings.log_dir) / "sessions"

        filepath = Path(directory) / f"{session_id}.json"

        if not filepath.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reconstruct session
        session = cls(
            case_id=data["case_id"],
            model_name=data["model"],
            session_id=data["session_id"]
        )

        session.timestamp_start = data["timestamp_start"]
        session.timestamp_end = data["timestamp_end"]
        session.initial_input = data["initial_input"]
        session.workflow_steps = data["workflow_steps"]
        session.final_output = data["final_output"]
        session.current_step = data["metadata"]["total_steps"]
        session.total_tokens = data["metadata"]["total_tokens"]
        session.total_latency_ms = data["metadata"]["total_latency_ms"]
        session.pii_redacted = data["metadata"]["pii_redacted"]
        session.completed = data["completed"]

        return session

    def get_reasoning_chain(self) -> List[str]:
        """
        Extract reasoning chain from all steps.

        Returns:
            List of reasoning statements
        """
        chain = []
        for step in self.workflow_steps:
            if "reasoning" in step:
                chain.append(f"[{step['agent']}] {step['reasoning']}")
        return chain

    def get_cited_guidelines(self) -> List[Dict[str, Any]]:
        """
        Extract all cited guidelines.

        Returns:
            List of guideline citations with metadata
        """
        citations = []

        for step in self.workflow_steps:
            # Check for retrieved documents (research agent)
            if "retrieved_documents" in step:
                for doc in step["retrieved_documents"]:
                    citations.append({
                        "source": doc.get("source"),
                        "title": doc.get("title"),
                        "similarity": doc.get("similarity"),
                        "text_preview": doc.get("text", "")[:100] + "..."
                    })

            # Check for cited guidelines in output
            if "output" in step and "cited_guidelines" in step["output"]:
                for guideline in step["output"]["cited_guidelines"]:
                    if isinstance(guideline, str):
                        citations.append({"guideline": guideline})
                    else:
                        citations.append(guideline)

        return citations

    def __repr__(self):
        return (
            f"ConversationSession(id={self.session_id}, "
            f"case={self.case_id}, model={self.model_name}, "
            f"steps={self.current_step}, completed={self.completed})"
        )


class ConversationManager:
    """
    Manages multiple conversation sessions.

    Provides session lifecycle management and analytics.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize conversation manager.

        Args:
            storage_dir: Directory for session storage
        """
        if storage_dir is None:
            storage_dir = Path(settings.log_dir) / "sessions"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.active_sessions: Dict[str, ConversationSession] = {}

        logger.info(f"ConversationManager initialized (storage: {self.storage_dir})")

    def create_session(
        self,
        case_id: str,
        model_name: str,
        agent_model: Optional[str] = None
    ) -> ConversationSession:
        """
        Create new conversation session.

        Args:
            case_id: Case identifier
            model_name: Model name
            agent_model: Agent model type (medgemma or gemini) for file naming

        Returns:
            New ConversationSession
        """
        session = ConversationSession(
            case_id=case_id,
            model_name=model_name,
            agent_model=agent_model
        )

        self.active_sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Get active session by ID.

        Args:
            session_id: Session ID

        Returns:
            ConversationSession or None
        """
        return self.active_sessions.get(session_id)

    def complete_session(self, session_id: str, save: bool = True):
        """
        Complete and optionally save session.

        Args:
            session_id: Session ID
            save: Whether to save to disk
        """
        session = self.active_sessions.get(session_id)

        if not session:
            logger.warning(f"Session not found: {session_id}")
            return

        session.complete()

        if save:
            session.save(self.storage_dir)

        # Remove from active sessions
        del self.active_sessions[session_id]

    def list_sessions(
        self,
        case_id: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> List[str]:
        """
        List saved session IDs.

        Args:
            case_id: Filter by case ID
            model_name: Filter by model name

        Returns:
            List of session IDs
        """
        # Support both old (session_*) and new (case_*) filename formats
        session_files = list(self.storage_dir.glob("*.json"))
        session_ids = []

        for filepath in session_files:
            # Quick load to check filters
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Apply filters
            if case_id and data.get("case_id") != case_id:
                continue
            if model_name and data.get("model") != model_name:
                continue

            session_ids.append(data["session_id"])

        return sorted(session_ids)

    def load_session(self, session_id: str) -> ConversationSession:
        """
        Load session from disk.

        Args:
            session_id: Session ID

        Returns:
            ConversationSession
        """
        return ConversationSession.load(session_id, self.storage_dir)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get manager statistics.

        Returns:
            Statistics dict
        """
        all_sessions = self.list_sessions()

        return {
            "total_sessions": len(all_sessions),
            "active_sessions": len(self.active_sessions),
            "storage_directory": str(self.storage_dir)
        }


# Global conversation manager instance
_conversation_manager = None


def get_conversation_manager() -> ConversationManager:
    """
    Get global conversation manager instance.

    Returns:
        ConversationManager singleton
    """
    global _conversation_manager

    if _conversation_manager is None:
        _conversation_manager = ConversationManager()

    return _conversation_manager
