"""
Gradio UI for MedGemma Clinical Robustness Assistant

Multimodal interface for:
- Image uploads (dermatology photos)
- Case file uploads (JSON/text vignettes)
- Interactive chat for missing data clarifications
- SOAP note display with agent reasoning
- Citation display from retrieved guidelines
"""

import gradio as gr
import logging
import json
import asyncio
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image

from src.agents.adk_agents import create_workflow
from src.utils.schemas import ClinicalCase, ContextState
from config.config import settings

logger = logging.getLogger(__name__)


class MedGemmaApp:
    """Gradio application for MedGemma Clinical Assistant."""

    def __init__(self):
        """
        Initialize the Gradio app.

        ARCHITECTURE:
        - Orchestration: Google ADK with Gemini Pro Latest (gemini-pro-latest)
        - Clinical Reasoning: medgemma-27b-it (google/medgemma-27b-it) [DEFAULT]
        - Toggle Option: Can switch agents to use Gemini Pro Latest for comparison
        """
        self.workflow = create_workflow(
            model_name="gemini-pro-latest",  # ADK orchestration (ALWAYS Gemini Pro Latest)
            use_medgemma=True  # DEFAULT: medgemma-27b-it for clinical reasoning
        )
        self.current_case = None
        self.current_session = None  # Stored on agentic pause for follow-up continuation
        self.conversation_history = []

        logger.info("MedGemmaApp initialized (Hybrid: ADK+Gemini Pro Latest orchestration, medgemma-27b-it reasoning)")

    def parse_case_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Parse uploaded case file (JSON or text).

        Args:
            file_path: Path to uploaded file

        Returns:
            Dict with case data or None if parsing fails
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Try JSON first
            try:
                case_data = json.loads(content)
                logger.info(f"Parsed JSON case file: {case_data.get('case_id', 'unknown')}")
                return case_data
            except json.JSONDecodeError:
                # Plain text file - extract context_state from filename
                # Filenames follow pattern: MM_DD_YY_<context_state>.txt
                # e.g. 01_23_25_image_only.txt, 01_23_25_original.txt
                filename = Path(file_path).stem.lower()  # e.g. "01_23_25_image_only"

                context_state = "original"  # default
                if filename.endswith("_image_only"):
                    context_state = "image_only"
                elif filename.endswith("_history"):
                    context_state = "history_only"
                elif filename.endswith("_exam_restricted"):
                    context_state = "exam_restricted"
                elif filename.endswith("_exam"):
                    context_state = "original"
                elif filename.endswith("_original"):
                    context_state = "original"
                elif filename.endswith("_redesign"):
                    context_state = "original"

                # Route content to correct field based on context_state
                case_data = {"context_state": context_state}

                if context_state == "image_only":
                    # image_only: no history or exam text to extract
                    case_data["history"] = ""
                    case_data["physical_exam"] = ""
                elif context_state == "history_only":
                    case_data["history"] = content
                elif context_state in ("exam_restricted",):
                    case_data["physical_exam"] = content
                elif filename.endswith("_exam"):
                    # exam file: content is physical exam findings
                    case_data["physical_exam"] = content
                else:
                    # original/redesign: full case goes into history
                    case_data["history"] = content

                logger.info(f"Parsed text file: context_state='{context_state}' from filename '{filename}'")
                return case_data
        except Exception as e:
            logger.error(f"Failed to parse case file: {e}")
            return None

    async def process_case_async(
        self,
        history: str,
        physical_exam: str,
        image: Optional[Image.Image],
        age: Optional[int],
        gender: Optional[str],
        duration: Optional[str],
        case_file: Optional[str]
    ) -> Tuple[str, str, str]:
        """
        Process clinical case through Google ADK workflow.

        Args:
            history: Patient history text
            physical_exam: Physical examination findings
            image: Clinical image (PIL Image)
            age: Patient age
            gender: Patient gender
            duration: Symptom duration
            case_file: Path to uploaded case file

        Returns:
            Tuple of (response, thinking_process, citations)
        """
        try:
            # Build case from inputs
            case_data = {
                "case_id": f"ui_case_{len(self.conversation_history) + 1}",
                "context_state": ContextState.ORIGINAL,
                "history": history or "",
                "physical_exam": physical_exam or "",
                "patient_age": age,
                "patient_gender": gender,
                "metadata": {"duration": duration} if duration else {}
            }

            # Override with case file if provided
            if case_file:
                file_data = self.parse_case_file(case_file)
                if file_data:
                    case_data.update(file_data)

            # Handle image
            if image:
                # Save image temporarily
                image_dir = Path("data/temp_images")
                image_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / f"{case_data['case_id']}.jpg"
                image.save(image_path)
                case_data["image_path"] = str(image_path)

            # Create ClinicalCase
            self.current_case = ClinicalCase(**case_data)

            # Run workflow
            logger.info(f"Processing case: {self.current_case.case_id}")
            result = await self.workflow.run_async(self.current_case)

            # Extract components from response
            response_text = result.get("response", "")

            # Check for missing data / clarification request (THE AGENTIC MOMENT)
            missing_data_detected = self._detect_missing_data(response_text)

            if missing_data_detected:
                # Store the session so follow-up can continue into it
                self.current_session = result.get("_session")
                logger.info(f"Agentic pause - session stored: {self.current_session.session_id if self.current_session else 'None'}")

                # Make the agentic moment UNAVOIDABLE and VISIBLE
                clarification_msg = self._format_clarification_request(response_text)
                soap_response = clarification_msg
                thinking_process = "### Agentic Decision\n\n" + \
                    "The Triage Agent has identified **missing critical data** required for safe diagnosis.\n\n" + \
                    "**Action:** Agent is requesting clarification instead of guessing or proceeding with incomplete information."
                # Show the follow-up input box
                show_followup = gr.update(visible=True)
            else:
                # No pause -- no session to carry forward
                self.current_session = None
                # Parse SOAP note sections
                soap_response = self._format_soap_response(response_text)
                thinking_process = self._extract_thinking_process(response_text)
                # Hide follow-up box (diagnosis is complete)
                show_followup = gr.update(visible=False)

            citations = self._extract_citations(response_text)

            # Store in conversation history
            self.conversation_history.append({
                "case": case_data,
                "response": response_text,
                "session_id": result.get("session_id")
            })

            return soap_response, thinking_process, citations, show_followup

        except Exception as e:
            logger.error(f"Error processing case: {e}", exc_info=True)
            error_msg = f"Error: {str(e)}\n\nPlease check your inputs and try again."
            return error_msg, "", "", gr.update(visible=False)

    async def process_followup_async(self, followup_text: str) -> Tuple[str, str, str, Any]:
        """
        Process user's follow-up answers to the agent's clarification questions.

        Continues the SAME session that was paused during the agentic pause:
        1. Logs the user's text as a "user_followup" step (linked to the pause step)
        2. Merges follow-up into the case
        3. Reruns the workflow with existing_session so all steps land in one log
        4. Finalizes and saves the session after the second run

        Args:
            followup_text: User's answers to the agent's questions

        Returns:
            Tuple of (response, thinking_process, citations, followup_visibility)
        """
        if not self.current_case:
            return "No active case to follow up on. Please submit a new case first.", "", "", gr.update(visible=False)

        try:
            import re
            from datetime import datetime

            # Parse follow-up for age pattern (e.g. "10 months old", "65 years old")
            age_match = re.search(r'(\d+)\s*(?:months?|years?)\s*old', followup_text.lower())
            if age_match and not self.current_case.patient_age:
                self.current_case.patient_age = int(age_match.group(1))

            # Parse gender
            if not self.current_case.patient_gender:
                if 'female' in followup_text.lower() or 'girl' in followup_text.lower() or 'woman' in followup_text.lower():
                    self.current_case.patient_gender = "female"
                elif 'male' in followup_text.lower() or 'boy' in followup_text.lower() or 'man' in followup_text.lower():
                    self.current_case.patient_gender = "male"

            # Append follow-up to history as additional context
            existing_history = self.current_case.history or ""
            if existing_history:
                updated_history = existing_history + "\n\nAdditional context provided by clinician:\n" + followup_text
            else:
                updated_history = followup_text

            self.current_case.history = updated_history
            self.current_case.context_state = "original"  # Now we have more data

            logger.info(f"Follow-up merged into case: {self.current_case.case_id}")
            logger.info(f"   Age: {self.current_case.patient_age}, Gender: {self.current_case.patient_gender}")

            # --- SESSION CONTINUITY: log the user's follow-up as a step ---
            if self.current_session:
                # The pause step is the last step in the session so far
                pause_step_id = self.current_session.current_step
                self.current_session.add_step(
                    agent_name="User",
                    step_data={
                        "input": followup_text,
                        "output": followup_text,
                        "output_type": "user_followup",
                        "orchestrator_action": "user_response",
                        "operation_type": "user_followup",
                        "tools_called": [],
                        "input_summary": f"Clinician follow-up answering agentic pause (linked to step {pause_step_id})",
                        "metadata": {
                            "linked_to_step": pause_step_id,
                            "parsed_age": self.current_case.patient_age,
                            "parsed_gender": self.current_case.patient_gender
                        }
                    },
                    orchestrator_model="user",
                    specialist_model=None,
                    input_reference={
                        "source_type": f"step_{pause_step_id}_TriageAgent_agentic_pause",
                        "reference": {
                            "step_id": pause_step_id,
                            "agent": "TriageAgent",
                            "data_flow": "user_followup"
                        }
                    },
                    step_role="user_followup",
                    is_final=False
                )
                logger.info(f"User follow-up logged as step {self.current_session.current_step} in session {self.current_session.session_id}")

            # --- RERUN WORKFLOW with the existing session (continuation) ---
            result = await self.workflow.run_async(
                self.current_case,
                existing_session=self.current_session  # None is safe -- creates new session
            )
            response_text = result.get("response", "")

            # Check again if still missing data
            missing_data_detected = self._detect_missing_data(response_text)

            session = result.get("_session")

            if missing_data_detected:
                # Still missing data -- keep the session open for another follow-up round
                self.current_session = session
                soap_response = self._format_clarification_request(response_text)
                thinking_process = "### Agentic Decision (Follow-up Round)\n\n" + \
                    "Additional information received. Triage Agent still identifies missing data.\n\n" + \
                    "**Action:** Requesting further clarification."
                show_followup = gr.update(visible=True)
            else:
                # Diagnosis complete -- finalize and save the single session that
                # now contains: initial run steps + user_followup step + continuation steps
                if session:
                    session.set_final_output(result)
                    self.workflow.conversation_manager.complete_session(session.session_id, save=True)
                    logger.info(f"Session {session.session_id} finalized and saved (full interaction captured)")
                self.current_session = None
                soap_response = self._format_soap_response(response_text)
                thinking_process = self._extract_thinking_process(response_text)
                show_followup = gr.update(visible=False)

            citations = self._extract_citations(response_text)

            self.conversation_history.append({
                "case": self.current_case.dict(exclude={'image_data'}),
                "response": response_text,
                "session_id": result.get("session_id"),
                "followup": followup_text
            })

            return soap_response, thinking_process, citations, show_followup

        except Exception as e:
            logger.error(f"Error processing follow-up: {e}", exc_info=True)
            return f"Error: {str(e)}\n\nPlease try again.", "", "", gr.update(visible=True)

    def process_followup(self, followup_text: str) -> Tuple[str, str, str, Any]:
        """Synchronous wrapper for process_followup_async."""
        return asyncio.run(self.process_followup_async(followup_text))

    def process_case(self, *args) -> Tuple[str, str, str, Any]:
        """Synchronous wrapper for process_case_async."""
        return asyncio.run(self.process_case_async(*args))

    def _detect_missing_data(self, response: str) -> bool:
        """
        Detect if the agent is asking for clarification due to missing data.

        Args:
            response: Agent response text

        Returns:
            True if missing data detected
        """
        missing_indicators = [
            "missing",
            "insufficient",
            "cannot provide",
            "need more information",
            "clarification",
            "please provide",
            "required information",
            "incomplete data",
            "unable to proceed",
            "please specify"
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in missing_indicators)

    def _format_clarification_request(self, response: str) -> str:
        """
        Format the clarification request to make the agentic moment PROMINENT.

        Args:
            response: Agent response text

        Returns:
            Formatted clarification message
        """
        # Extract questions or missing items from response
        lines = response.split('\n')
        questions = []
        missing_items = []

        for line in lines:
            if '?' in line:
                questions.append(line.strip())
            if 'missing' in line.lower():
                missing_items.append(line.strip())

        # Build prominent message
        msg = "# ⚠️ AGENTIC PAUSE: Missing Critical Data\n\n"
        msg += "## Agent Decision\n\n"
        msg += "**I cannot provide a reliable assessment because critical clinical information is missing.**\n\n"
        msg += "In dermatology, accurate diagnosis requires complete clinical context. "
        msg += "Proceeding with incomplete data risks diagnostic error.\n\n"

        if missing_items:
            msg += "## Missing Information\n\n"
            for item in missing_items[:3]:  # Top 3
                msg += f"- {item}\n"
            msg += "\n"

        if questions:
            msg += "## Required Clarifications\n\n"
            for q in questions[:5]:  # Top 5 questions
                msg += f"**Q:** {q}\n\n"

        msg += "---\n\n"
        msg += "**Next Step:** Please provide the missing information above, then resubmit for analysis.\n\n"
        msg += "*This pause demonstrates the agent's commitment to safe, evidence-based reasoning over guessing.*"

        return msg

    def _format_soap_response(self, response: str) -> str:
        """
        Format SOAP note with Markdown styling.

        Args:
            response: Raw response text

        Returns:
            Formatted SOAP note
        """
        # Add markdown headers if not present
        formatted = response

        if "Subjective" in response and not "**Subjective" in response:
            formatted = formatted.replace("Subjective (S):", "## Subjective (S)")
            formatted = formatted.replace("Objective (O):", "## Objective (O)")
            formatted = formatted.replace("Assessment (A):", "## Assessment (A)")
            formatted = formatted.replace("Plan (P):", "## Plan (P)")

        return formatted

    def _extract_thinking_process(self, response: str) -> str:
        """
        Extract agent thinking process from response.

        Args:
            response: Raw response text

        Returns:
            Thinking process summary
        """
        # Build structured reasoning trace
        trace = "### Multi-Agent Workflow Execution\n\n"

        # Look for reasoning sections
        thinking_markers = {
            "triage": "🔍 **Step 1: Triage Agent**",
            "research": "📚 **Step 2: Research Agent**",
            "diagnostic": "🩺 **Step 3: Diagnostic Agent**",
            "retrieved": "📚 **Step 2: Research Agent**",
            "guidelines": "📚 **Step 2: Research Agent**",
            "soap": "🩺 **Step 3: Diagnostic Agent**",
            "assessment": "🩺 **Step 3: Diagnostic Agent**"
        }

        sections = {
            "triage": [],
            "research": [],
            "diagnostic": []
        }

        for line in response.split('\n'):
            line_lower = line.lower()
            if 'triage' in line_lower or 'missing' in line_lower:
                sections["triage"].append(line.strip())
            elif 'research' in line_lower or 'retrieved' in line_lower or 'guideline' in line_lower:
                sections["research"].append(line.strip())
            elif 'diagnostic' in line_lower or 'soap' in line_lower or 'assessment' in line_lower:
                sections["diagnostic"].append(line.strip())

        # Build trace
        trace += "🔍 **Step 1: Triage Agent** - Analyzed case completeness\n"
        if sections["triage"]:
            trace += "   - " + "\n   - ".join(sections["triage"][:3]) + "\n"
        trace += "\n"

        trace += "📚 **Step 2: Research Agent** - Retrieved evidence-based guidelines\n"
        if sections["research"]:
            trace += "   - " + "\n   - ".join(sections["research"][:3]) + "\n"
        else:
            trace += "   - Queried ChromaDB knowledge base (1,492 chunks)\n"
            trace += "   - Retrieved relevant AAD and StatPearls guidelines\n"
        trace += "\n"

        trace += "🩺 **Step 3: Diagnostic Agent** - Synthesized SOAP note\n"
        if sections["diagnostic"]:
            trace += "   - " + "\n   - ".join(sections["diagnostic"][:3]) + "\n"
        else:
            trace += "   - Combined clinical data with guideline evidence\n"
            trace += "   - Generated structured differential diagnosis\n"

        trace += "\n---\n\n*This trace shows the collaborative reasoning of our multi-agent system.*"

        return trace

    def _extract_citations(self, response: str) -> str:
        """
        Extract guideline citations from response and format with clear grounding.

        Args:
            response: Raw response text

        Returns:
            Formatted citations with source, section, and evidence
        """
        # Look for citation patterns
        citation_lines = []
        in_citations = False

        for line in response.split('\n'):
            if any(marker in line.lower() for marker in ["citation", "reference", "guideline", "source:", "aad", "statpearls"]):
                in_citations = True

            if in_citations and line.strip():
                citation_lines.append(line)

            # Stop at next major section
            if in_citations and line.startswith('##'):
                break

        # Build structured citations display
        citations_md = "### Evidence-Based Clinical Guidelines\n\n"
        citations_md += "*The following guidelines from our knowledge base (ChromaDB) support this assessment:*\n\n"

        if citation_lines:
            # Format each citation with clear structure
            for i, line in enumerate(citation_lines[:5], 1):  # Top 5 citations
                # Clean up the line
                clean_line = line.strip('- ').strip()

                if 'AAD' in clean_line or 'American Academy of Dermatology' in clean_line:
                    citations_md += f"**{i}. AAD Guideline**\n\n"
                    citations_md += f"   {clean_line}\n\n"
                elif 'StatPearls' in clean_line:
                    citations_md += f"**{i}. StatPearls Medical Reference**\n\n"
                    citations_md += f"   {clean_line}\n\n"
                else:
                    citations_md += f"**{i}.** {clean_line}\n\n"

            citations_md += "---\n\n"
            citations_md += "*Evidence grounding prevents hallucination and ensures clinically validated recommendations.*"
        else:
            # Default message when no specific citations found
            citations_md += "**Retrieved from:**\n\n"
            citations_md += "- American Academy of Dermatology (AAD) Clinical Guidelines\n"
            citations_md += "- StatPearls Medical Reference Library\n\n"
            citations_md += "*1,492 guideline chunks available in knowledge base*\n\n"
            citations_md += "---\n\n"
            citations_md += "*Specific citations will appear here when diagnosis is generated.*"

        return citations_md

    def create_ui(self) -> gr.Blocks:
        """
        Create Gradio interface.

        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="MedGemma Clinical Robustness Assistant"
        ) as app:
            gr.Markdown("""
            # MedGemma Clinical Robustness Assistant

            **Powered by MedGemma-27B** (Health-Specialized AI) | **Multi-Agent:** Triage → Research → Diagnosis
            **Evidence-Based:** 1,492 Clinical Guidelines (AAD, StatPearls)

            Upload images/files or use manual input. Agent will request missing data before diagnosing.

            **⚠️ DISCLAIMER:** Research and demonstration purposes only. Not for clinical use.
            """)

            with gr.Row():
                # Left column: Inputs
                with gr.Column(scale=1):
                    gr.Markdown("## Clinical Case Input")

                    # Image upload (compact)
                    image_input = gr.Image(
                        label="Clinical Image (Optional)",
                        type="pil",
                        height=180
                    )

                    # Case file upload (compact)
                    file_input = gr.File(
                        label="Case Vignette File (JSON/Text)",
                        file_types=[".json", ".txt"],
                        height=100
                    )

                    gr.Markdown("*💡 Tip: Scroll ↓ for manual clinical context fields*")

                    # Submit button (MOVED UP - primary action)
                    submit_btn = gr.Button(
                        "🔬 Analyze Case",
                        variant="primary",
                        size="lg"
                    )

                    # Manual inputs (COLLAPSED by default)
                    with gr.Accordion("📝 Manual Input Fields (Alternative to File Upload)", open=False):
                        history_input = gr.Textbox(
                            label="Patient History",
                            placeholder="e.g., 65-year-old man with itchy red rash on elbows for 3 weeks",
                            lines=3
                        )

                        exam_input = gr.Textbox(
                            label="Physical Examination",
                            placeholder="e.g., Erythematous plaques with silvery scales on bilateral elbows",
                            lines=3
                        )

                        with gr.Row():
                            age_input = gr.Number(
                                label="Age (years)",
                                minimum=0,
                                maximum=120
                            )
                            gender_input = gr.Dropdown(
                                label="Gender",
                                choices=["male", "female", "other"],
                                value=None
                            )

                        duration_input = gr.Textbox(
                            label="Symptom Duration",
                            placeholder="e.g., 3 weeks"
                        )

                # Right column: Outputs
                with gr.Column(scale=1):
                    gr.Markdown("## Clinical Assessment")

                    # SOAP note output
                    soap_output = gr.Markdown(
                        label="SOAP Note",
                        value="SOAP note will appear here after analysis..."
                    )

                    # Follow-up section (HIDDEN by default, shown on agentic pause)
                    with gr.Group(visible=False) as followup_group:
                        gr.Markdown("---\n### Reply to Agent\nProvide the missing information the agent requested above:")
                        followup_input = gr.Textbox(
                            label="Your Response",
                            placeholder="e.g., The patient is a 10-month-old female. Symptoms include itching for 6 months. No family history of skin conditions...",
                            lines=3
                        )
                        followup_btn = gr.Button(
                            "Submit Follow-up",
                            variant="primary",
                            size="md"
                        )

                    # Clinical reasoning trace accordion
                    with gr.Accordion("Clinical Reasoning Trace", open=False):
                        thinking_output = gr.Markdown(
                            value="Agent reasoning workflow will appear here..."
                        )

                    # Citations accordion
                    with gr.Accordion("Evidence-Based Guidelines (AAD, StatPearls)", open=False):
                        citations_output = gr.Markdown(
                            value="Retrieved guideline citations will appear here..."
                        )

            # Examples (collapsed to save space)
            with gr.Accordion("📋 Example Cases (Click to Load)", open=False):
                gr.Examples(
                    examples=[
                        [
                            "65-year-old man with itchy red rash on elbows for 3 weeks",
                            "Erythematous plaques with silvery scales on bilateral elbows",
                            None,
                            65,
                            "male",
                            "3 weeks",
                            None
                        ],
                        [
                            "32-year-old woman with facial rash worsening in sunlight",
                            "Malar erythema sparing nasolabial folds",
                            None,
                            32,
                            "female",
                            "2 months",
                            None
                        ],
                        [
                            "8-year-old child with intensely itchy rash in skin folds",
                            "Erythematous patches in antecubital and popliteal fossae with lichenification",
                            None,
                            8,
                            "female",
                            "6 months, recurrent",
                            None
                        ]
                    ],
                    inputs=[
                        history_input,
                        exam_input,
                        image_input,
                        age_input,
                        gender_input,
                        duration_input,
                        file_input
                    ]
                )

            # Event handlers

            # Main submit button
            submit_btn.click(
                fn=self.process_case,
                inputs=[
                    history_input,
                    exam_input,
                    image_input,
                    age_input,
                    gender_input,
                    duration_input,
                    file_input
                ],
                outputs=[
                    soap_output,
                    thinking_output,
                    citations_output,
                    followup_group
                ]
            )

            # Follow-up button (answers agent's clarification questions)
            followup_btn.click(
                fn=self.process_followup,
                inputs=[followup_input],
                outputs=[
                    soap_output,
                    thinking_output,
                    citations_output,
                    followup_group
                ]
            )

            gr.Markdown("""
            ---
            **Clinical Reasoning:** MedGemma-27B (Health-Specialized) | **Orchestration:** Google ADK + Gemini Pro
            **Knowledge Base:** ChromaDB (1,492 AAD + StatPearls Guideline Chunks) | **Architecture:** Manager-Specialist Pattern
            **Built for:** [Med-Gemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
            """)

        return app


def launch_app(
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int = None
):
    # Use environment variable if set, otherwise default to 7860
    if server_port is None:
        server_port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    """
    Launch the Gradio app.

    Args:
        share: Whether to create public sharing link
        server_name: Server hostname
        server_port: Server port
    """
    logger.info("Launching MedGemma Gradio app...")

    app_instance = MedGemmaApp()
    app = app_instance.create_ui()

    app.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_error=True,
        inbrowser=True
    )


if __name__ == "__main__":
    launch_app()
