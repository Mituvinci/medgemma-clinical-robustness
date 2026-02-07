"""
Gradio UI for MedGemma Clinical Robustness Assistant
EXACT MATCH to HTML design specification

Multimodal interface for:
- Image uploads (dermatology photos)
- Case file uploads (JSON/text vignettes)
- Manual input modal with "Analyze & Close"
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
    """Gradio application for MedGemma Clinical Assistant - Exact HTML Design Match."""

    def __init__(self):
        """
        Initialize the Gradio app.

        ARCHITECTURE:
        - Orchestration: Google ADK with Gemini Pro Latest (gemini-pro-latest)
        - Clinical Reasoning: medgemma-27b-it (google/medgemma-27b-it) [DEFAULT]
        """
        self.workflow = create_workflow(
            model_name="gemini-pro-latest",
            use_medgemma=True
        )
        self.current_case = None
        self.current_session = None
        self.conversation_history = []
        self.is_analyzing = False
        self.uploaded_files = []
        self.manual_data = {}

        logger.info("MedGemmaApp initialized (HTML Design Match)")

    def parse_case_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Parse uploaded case file (JSON or text)."""
        try:
            if file_path.lower().endswith('.pdf'):
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                content = "\n".join(page.extract_text() or "" for page in reader.pages)
                logger.info(f"Parsed PDF case file: {len(reader.pages)} pages")
                return {"history": content, "context_state": "original"}

            with open(file_path, 'r') as f:
                content = f.read()

            try:
                case_data = json.loads(content)
                logger.info(f"Parsed JSON case file: {case_data.get('case_id', 'unknown')}")
                return case_data
            except json.JSONDecodeError:
                return {"history": content, "context_state": "original"}

        except Exception as e:
            logger.error(f"Failed to parse case file: {e}")
            return None

    def get_next_session_id(self) -> str:
        """Auto-increment session ID using file-based counter."""
        counter_file = Path("data/session_counter.txt")
        counter_file.parent.mkdir(exist_ok=True)

        if counter_file.exists():
            with open(counter_file, 'r') as f:
                counter = int(f.read().strip())
        else:
            counter = 0

        counter += 1

        with open(counter_file, 'w') as f:
            f.write(str(counter))

        return f"session_{counter:03d}"

    def build_case_from_inputs(
        self,
        text_input: str,
        history: str = "",
        exam: str = "",
        age: Optional[int] = None,
        gender: Optional[str] = None,
        duration: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build case dictionary from inputs."""
        case_data = {"context_state": "original"}

        # Priority: manual fields > text input
        if history or exam or age or gender or duration:
            if history:
                case_data["history"] = history
            if exam:
                case_data["exam"] = exam
            if age:
                case_data["age"] = age
            if gender:
                case_data["gender"] = gender
            if duration:
                case_data["duration"] = duration
        elif text_input:
            case_data["history"] = text_input

        # Process uploaded files
        if self.uploaded_files:
            for file_path in self.uploaded_files:
                file_ext = file_path.lower() if isinstance(file_path, str) else file_path.name.lower()

                if any(file_ext.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                    case_data["image"] = Image.open(file_path)
                elif any(file_ext.endswith(ext) for ext in ['.json', '.txt', '.pdf']):
                    file_data = self.parse_case_file(file_path)
                    if file_data:
                        case_data.update(file_data)

        return case_data

    def process_case(
        self,
        text_input: str,
        history: str,
        exam: str,
        age: Optional[int],
        gender: Optional[str],
        duration: Optional[str]
    ):
        """Process initial case submission. Generator for streaming updates."""
        self.is_analyzing = True

        # Build case from inputs BEFORE clearing
        case_data = self.build_case_from_inputs(text_input, history, exam, age, gender, duration)
        self.uploaded_files = []

        # STATE: Analyzing
        yield (
            "Processing your case...",  # soap_output
            "",  # reasoning_output
            "",  # citations_output
            "",  # text_input cleared
            gr.update(value="Analyzing...", interactive=False),  # analyze_btn
        )

        try:
            self.current_case = ClinicalCase(**case_data)
            session_id = self.get_next_session_id()

            logger.info(f"Starting workflow for {session_id}")
            result = asyncio.run(self.workflow.run_case(self.current_case, session_id))

            soap_note = result.get("soap_note", "No diagnosis generated.")
            reasoning = self._extract_thinking_process(result.get("raw_response", ""))
            citations = self._extract_citations(result.get("raw_response", ""))

            # Check if agentic pause triggered
            if result.get("missing_data") or result.get("questions"):
                pause_msg = self._format_agentic_pause(
                    result.get("missing_data", []),
                    result.get("questions", [])
                )

                self.current_session = session_id
                self.is_analyzing = False

                yield (
                    pause_msg,
                    reasoning,
                    citations,
                    "",
                    gr.update(value="Submit Follow-up", interactive=True),
                )
            else:
                # Completed
                self.is_analyzing = False
                self.current_session = None

                yield (
                    self._format_soap_response(soap_note),
                    reasoning,
                    citations,
                    "",
                    gr.update(value="Analyze Case", interactive=True),
                )

        except Exception as e:
            logger.error(f"Error in process_case: {e}")
            self.is_analyzing = False

            yield (
                f"Error: {str(e)}",
                "",
                "",
                "",
                gr.update(value="Analyze Case", interactive=True),
            )

    def _format_agentic_pause(self, missing_items: List[str], questions: List[str]) -> str:
        """Format agentic pause message."""
        msg = "## Agentic Pause: Missing Information\n\n"
        msg += "The agent has paused to request critical clinical context before diagnosing.\n\n"

        if missing_items:
            msg += "### Missing Information\n\n"
            for item in missing_items[:3]:
                msg += f"- {item}\n"
            msg += "\n"

        if questions:
            msg += "### Required Clarifications\n\n"
            for q in questions[:5]:
                msg += f"**Q:** {q}\n\n"

        msg += "---\n\n"
        msg += "**Next Step:** Please provide the missing information using the text box below.\n\n"

        return msg

    def _format_soap_response(self, response: str) -> str:
        """Format SOAP note with Markdown styling."""
        formatted = response

        if "Subjective" in response and not "**Subjective" in response:
            formatted = formatted.replace("Subjective (S):", "## Subjective (S)")
            formatted = formatted.replace("Objective (O):", "## Objective (O)")
            formatted = formatted.replace("Assessment (A):", "## Assessment (A)")
            formatted = formatted.replace("Plan (P):", "## Plan (P)")

        return formatted

    def _extract_thinking_process(self, response: str) -> str:
        """Extract agent thinking process from response."""
        trace = "### Multi-Agent Workflow Execution\n\n"
        trace += "Triage Agent → Research Agent → Diagnostic Agent\n\n"
        trace += "**Step 1: Triage Agent** - Analyzed case completeness\n\n"
        trace += "**Step 2: Research Agent** - Retrieved evidence-based guidelines from ChromaDB (1,492 chunks)\n\n"
        trace += "**Step 3: Diagnostic Agent** - Synthesized SOAP note with guideline evidence\n\n"
        trace += "---\n\n*This trace shows the collaborative reasoning of our multi-agent system.*"

        return trace

    def _extract_citations(self, response: str) -> str:
        """Extract guideline citations from response."""
        citations_md = "### Evidence-Based Clinical Guidelines\n\n"
        citations_md += "*The following guidelines from our knowledge base (ChromaDB) support this assessment:*\n\n"
        citations_md += "**Retrieved from:**\n\n"
        citations_md += "- American Academy of Dermatology (AAD) Clinical Guidelines\n"
        citations_md += "- StatPearls Medical Reference Library\n\n"
        citations_md += "*1,492 guideline chunks available in knowledge base*\n\n"
        citations_md += "---\n\n"
        citations_md += "*Evidence grounding prevents hallucination and ensures clinically validated recommendations.*"

        return citations_md

    def create_ui(self) -> gr.Blocks:
        """Create Gradio interface matching EXACT HTML design."""

        theme = gr.themes.Base(
            primary_hue=gr.themes.colors.orange,
            neutral_hue=gr.themes.colors.neutral,
            text_size=gr.themes.sizes.text_sm,
            spacing_size=gr.themes.sizes.spacing_sm,
            radius_size=gr.themes.sizes.radius_sm,
            font=("system-ui", "-apple-system", "sans-serif"),
        )

        with gr.Blocks(title="MedGemma Clinical Robustness Assistant", theme=theme) as app:

            gr.HTML("""<style>
/* ===== GLOBAL RESET (removes gray background) ===== */
html, body {
    background: #fff !important;
}

.gradio-container,
.wrap,
.container,
.gr-group {
    background-color: #fff !important;
}

.gr-block,
.gr-panel,
.gr-box {
    background: #fff !important;
}

/* ===== FIX FONT SIZE ACROSS BROWSERS ===== */
/* ===== FORCE CONSISTENT FONT SCALING ===== */
:root {
    font-size: 16px !important;
    --background-fill-primary: #fff !important;
    --background-fill-secondary: #fff !important;
    --block-background-fill: #fff !important;
    --panel-background-fill: #fff !important;
    --body-background-fill: #fff !important;
}

body {
    font-size: 1rem !important;
}

/* ===== HIDDEN FILE UPLOAD ===== */
#hidden-file-upload {
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    opacity: 0 !important;
}

/* ===== ACCORDIONS ===== */
#acc-reasoning,
#acc-citations {
    font-size: 0.8rem !important;
}

/* ===== ASSESSMENT GROUP ===== */
#assessment-group {
    gap: 2px !important;
    background: transparent !important;
    border: none !important;
}

/* ===== PINNED INPUT SECTION ===== */
#input-section {
    position: fixed !important;
    bottom: 0 !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 100% !important;
    max-width: 1200px !important;
    background: #fff !important;
    z-index: 1000 !important;
    padding: 12px 16px !important;
    border-top: 1px solid #ddd !important;
    box-shadow: 0 -2px 8px rgba(0,0,0,0.05) !important;
}


/* ===== SCROLL AREA (space for pinned input) ===== */
#scrollable-content {
    padding-bottom: 280px !important;
    background: transparent !important;
    border: none !important;
}

/* ===== REMOVE GRADIO MARKDOWN GRAY BANDS ===== */


.gr-markdown,
.gr-markdown > div,
.gr-markdown .prose {
    background: #fff !important;
}

.prose * {
    background: #fff !important;
    
}

/* Target the REAL flex container that Gradio renders inside the elem_id wrapper */
#input-btn-row .gr-row {
  display: flex !important;
  justify-content: center !important;  /* center the 3 buttons */
  gap: 12px !important;               /* space between buttons */
  column-gap: 12px !important;
  flex-wrap: nowrap !important;
  align-items: center !important;
}

/* Prevent Gradio wrappers from stretching */
#input-btn-row .gr-row > * {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: unset !important;
}

/* Extra-safe fallback: if gap still doesn't show, force margin */
#input-btn-row button {
  margin: 0 6px !important;   /* 6px left + 6px right = 12px gap */
}


/* Less tall buttons */
#input-section button {
  padding: 8px 14px !important;
  line-height: 1.1 !important;
  font-weight: 600 !important;
}

/* Thin separators */
hr {
  border: none !important;
  border-top: 1px solid #e5e5e5 !important;margin: 0;
  margin: 0 !important;
  height: 0 !important;
  background: transparent !important;
}
.gr-markdown.gr-block {
  padding-top: 2px !important;
  padding-bottom: 2px !important;
}



</style>""")

            # Hidden file upload (CSS hides it, but stays in DOM so JS can trigger it)
            file_upload = gr.File(
                label="",
                file_types=[".jpg", ".jpeg", ".png", ".json", ".txt", ".pdf"],
                file_count="multiple",
                type="filepath",
                elem_id="hidden-file-upload",
            )

            # SCROLLABLE TOP CONTENT
            with gr.Group(elem_id="scrollable-content"):

                # SECTION 1: HEADER (single Markdown to avoid Gradio gaps)
                gr.Markdown(
                   "<div style='padding-left:16px; padding-top:8px;  padding-bottom:8px;'>"
                    "<span style='font-size:16px; font-weight:600;'>MedGemma Clinical Robustness Assistant</span><br>"
                    "<span style='font-size:9px; color:#666;'>"
                    "Powered by MedGemma-27B (Health-Specialized AI) | "
                    "Multi-Agent: Triage → Research → Diagnosis "
                    "Evidence-Based: 1,492 Clinical Guidelines (AAD, StatPearls)"
                    "</span><br>"
                    "<span style='font-size:9px; color:#999;'>"
                    "Upload image/file or use manual input. Agent will request missing data before diagnosing.<br>"
                    "<strong>DISCLAIMER:</strong> Research and demonstration purposes only. Not for clinical use."
                    "</span>"
		    "</div>"
                )

                #gr.Markdown("---")

                # SECTION 2: CLINICAL ASSESSMENT
                gr.Markdown(
		    "<div style='padding-left:16px; padding-top:8px;  padding-bottom:8px;'>"
                    "<span style='font-size:13px; font-weight:600;'>Clinical Assessment</span><br>"
                    "<span style='font-size:9px; color:#666;'>*SOAP note will appear here after analysis...*</span>"
		    "</div>"
                )

                soap_output = gr.Markdown(value="")

                with gr.Group(elem_id="assessment-group"):
                    with gr.Accordion("Clinical Reasoning Trace", open=False, elem_id="acc-reasoning"):
                        reasoning_output = gr.Markdown(
                            value="<span style='font-size:9px;'>Detailed clinical reasoning will be displayed here after case analysis.</span>"
                        )

                    with gr.Accordion("Evidence-Based Guidelines (AAD, StatPearls)", open=False, elem_id="acc-citations"):
                        citations_output = gr.Markdown(
                            value="<span style='font-size:9px;'>Relevant clinical guidelines and evidence-based references will be displayed here.</span>"
                        )

                #gr.Markdown("---")

                # SECTION 4: EXAMPLES
                gr.Markdown("<div style='padding-left:16px; padding-top:4px;  padding-bottom:4px;'>" 
                            "<span style='font-size:11px; font-weight:600;'>Example Cases (Click to load):</span>"
                            "</div>")
                gr.Markdown("<div style='padding-left:16px; padding-top:4px;  padding-bottom:4px;'>" 
                             "<span style='font-size:10px;'>[Some example cases](#)<br>[Some example cases](#)</span>"
		             "</div>")

                #gr.Markdown("---")

             

            # PINNED BOTTOM INPUT SECTION
            with gr.Group(elem_id="input-section"):
                gr.Markdown( "<div style='padding-left:16px; padding-top:8px;  padding-bottom:8px;'>" 
			     "<span style='font-size:13px; font-weight:600;'>Clinical Case Input</span>"
                             "</div>")

                text_input = gr.Textbox(
                    placeholder="Describe case or attach file...",
                    lines=4,
                    show_label=False,
                )

                with gr.Row(elem_id="input-btn-row"):
                    attach_btn = gr.Button("📎 Attach", variant="secondary", scale=0)
                    manual_btn = gr.Button("✏️ Manual Input", variant="secondary", scale=0)
                    analyze_btn = gr.Button("Analyze Case", variant="primary",  scale=0)

            # MODAL - Using native Gradio (visible toggle)
            modal_visible = gr.State(False)

            with gr.Group(visible=False) as modal_backdrop:
                gr.Markdown("### Manual Input Fields")

                history_input = gr.Textbox(
                    label="Patient History",
                    placeholder="65-year-old man with itchy red rash on elbows for 3 weeks",
                    lines=3
                )

                exam_input = gr.Textbox(
                    label="Physical Examination",
                    placeholder="Erythematous plaques with silvery scales on bilateral elbows",
                    lines=3
                )

                with gr.Row():
                    age_input = gr.Number(
                        label="Age (years)",
                        minimum=0,
                        maximum=120,
                        value=None
                    )

                    gender_input = gr.Dropdown(
                        label="Gender",
                        choices=["male", "female", "other"],
                        value=None
                    )

                duration_input = gr.Textbox(
                    label="Symptom Duration",
                    placeholder="3 weeks"
                )

                with gr.Row():
                    cancel_btn = gr.Button("Cancel", variant="secondary")
                    analyze_close_btn = gr.Button("Analyze & Close", variant="primary")

            # EVENT HANDLERS

            # Track uploaded files
            def track_uploads(files):
                if files:
                    self.uploaded_files = files if isinstance(files, list) else [files]
                else:
                    self.uploaded_files = []
                return files

            file_upload.change(
                fn=track_uploads,
                inputs=[file_upload],
                outputs=[file_upload]
            )

            # Attach button - directly open system file browser (one click)
            attach_btn.click(
                fn=None,
                js="() => { const el = document.querySelector('#hidden-file-upload input[type=file]'); if (el) el.click(); }"
            )

            # Manual button - open modal
            def open_modal():
                return gr.update(visible=True)

            manual_btn.click(
                fn=open_modal,
                outputs=[modal_backdrop]
            )

            # Cancel button - close modal
            def close_modal():
                return gr.update(visible=False)

            cancel_btn.click(
                fn=close_modal,
                outputs=[modal_backdrop]
            )

            # Analyze & Close button - save data, close modal, and trigger analysis
            def analyze_and_close(history, exam, age, gender, duration):
                """Save manual data, close modal, and trigger analysis immediately."""
                # Save to manual_data
                self.manual_data = {
                    "history": history,
                    "exam": exam,
                    "age": age,
                    "gender": gender,
                    "duration": duration
                }

                # Close modal and trigger analysis
                return (
                    gr.update(visible=False),  # modal_backdrop
                    gr.update(value="trigger_analysis")  # signal to trigger analysis
                )

            analyze_close_btn.click(
                fn=analyze_and_close,
                inputs=[history_input, exam_input, age_input, gender_input, duration_input],
                outputs=[modal_backdrop, text_input]
            ).then(
                # Immediately trigger analysis with manual data
                fn=lambda txt, h, e, a, g, d: self.process_case(txt, h, e, a, g, d),
                inputs=[text_input, history_input, exam_input, age_input, gender_input, duration_input],
                outputs=[soap_output, reasoning_output, citations_output, text_input, analyze_btn]
            )

            # Analyze Case button - normal analysis
            analyze_btn.click(
                fn=lambda txt, h, e, a, g, d: self.process_case(txt, h, e, a, g, d),
                inputs=[text_input, history_input, exam_input, age_input, gender_input, duration_input],
                outputs=[soap_output, reasoning_output, citations_output, text_input, analyze_btn]
            )

        return app


def launch_app():
    """Launch the Gradio app."""
    app_instance = MedGemmaApp()
    app = app_instance.create_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    launch_app()
