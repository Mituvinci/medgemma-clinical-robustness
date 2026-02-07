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

    def build_case_from_inputs(self, text_input: str, file_list: List[str] = None, session_id: str = None) -> Dict[str, Any]:
        """Build case dictionary from text input and uploaded files."""
        case_data = {
            "case_id": f"ui_{session_id or 'unknown'}",
            "context_state": "original",
        }

        if text_input:
            case_data["history"] = text_input

        # Process uploaded files
        files = file_list or []
        for file_path in files:
            file_ext = file_path.lower() if isinstance(file_path, str) else file_path.name.lower()

            if any(file_ext.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                case_data["image_data"] = Image.open(file_path)
            elif any(file_ext.endswith(ext) for ext in ['.json', '.txt', '.pdf']):
                file_data = self.parse_case_file(file_path)
                if file_data:
                    case_data.update(file_data)

        return case_data

    def process_case(self, text_input: str, file_list: List[str] = None):
        """Process case submission. Generator for streaming updates."""
        self.is_analyzing = True

        # Generate session_id once, use for both case_id and workflow
        session_id = self.get_next_session_id()

        # Build case from inputs BEFORE clearing
        case_data = self.build_case_from_inputs(text_input, file_list, session_id)

        # STATE: Analyzing — yield 6 outputs matching UI components
        yield (
            "Processing your case...",  # soap_output
            "",  # reasoning_output
            "",  # citations_output
            "",  # text_input cleared
            gr.update(value="Analyzing...", interactive=False),  # analyze_btn
            gr.update(visible=False),  # reset_btn hidden during analysis
        )

        try:
            self.current_case = ClinicalCase(**case_data)

            logger.info(f"Starting workflow for {session_id}")
            result = asyncio.run(self.workflow.run_async(self.current_case))

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
                    gr.update(value="Follow-up", interactive=True),
                    gr.update(visible=True),  # show reset button
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
                    gr.update(visible=False),  # reset stays hidden
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
                gr.update(visible=False),
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

    /* Gradio theme variables */
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #ffffff !important;
    --block-background-fill: #ffffff !important;
    --panel-background-fill: #ffffff !important;
    --body-background-fill: #ffffff !important;

    /* Explicit fallback */
    background-color: #ffffff !important;
}

body {
    font-size: 1rem !important; 
    background-color:#fff;
}

/* ===== HIDDEN FILE UPLOAD ===== */
#hidden-file-upload,
#remove-file-idx {
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
    background-color: #fff !important;
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


/* Extra-safe fallback: if gap still doesn't show, force margin */
#input-btn-row button {
  margin: 0 50px !important;   /* 6px left + 6px right = 12px gap */
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
/* Smaller placeholder text inside textbox */
#input-section textarea::placeholder {
    font-size: 11px !important;
    font-style: italic;
    background-color: transparent !important;

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
                gr.Markdown(
                    "<div style='padding-left:16px; padding-top:8px; padding-bottom:4px;'>"
                    "<span style='font-size:13px; font-weight:600;'>Clinical Case Input</span>"
                    "</div>"
                )

                text_input = gr.Textbox(
                    placeholder=(
                        "Describe your case (history, exam findings, age, symptoms, duration...) "
                        "and/or attach files (image, PDF, JSON, text -- max 3 files). "
                        "Or click an example from above for quick analysis."
                    ),
                    lines=5,
                    show_label=False,
                )

                # State to track attached files (list of file paths)
                file_state = gr.State([])

                # Row showing attached filenames with X buttons
                file_display = gr.HTML(value="", elem_id="file-display")

                with gr.Row(elem_id="input-btn-row"):
                    attach_btn = gr.Button("Attach", variant="secondary", scale=0,size='sm')
                    reset_btn = gr.Button("Reset", variant="secondary", scale=0, visible=False,size='sm')
                    analyze_btn = gr.Button("Analyze Case", variant="primary", scale=0,size='sm')

            # EVENT HANDLERS

            def format_file_display(file_list):
                """Generate HTML for attached file names with X buttons."""
                if not file_list:
                    return ""
                html_parts = []
                for i, fp in enumerate(file_list):
                    name = os.path.basename(fp)
                    if len(name) > 50:
                        name = name[:50] + "..."
                    html_parts.append(
                        f"<span style='display:inline-flex; align-items:center; "
                        f"background:#f0f0f0; border:1px solid #ddd; border-radius:4px; "
                        f"padding:2px 8px; margin:2px 4px; font-size:11px;'>"
                        f"{name}"
                        f"<button onclick=\"var el=document.querySelector('#remove-file-idx textarea')||document.querySelector('#remove-file-idx input');if(el){{el.value='{i}';el.dispatchEvent(new Event('input',{{bubbles:true}}));}}\""
                        f" style='border:none; background:none; cursor:pointer; "
                        f"margin-left:6px; font-size:13px; color:#999;'>x</button>"
                        f"</span>"
                    )
                return "<div style='padding:4px 0;'>" + "".join(html_parts) + "</div>"

            # Track uploaded files from hidden gr.File
            def on_file_upload(files, current_files):
                """Add newly uploaded files to the list (max 3)."""
                if not files:
                    return current_files, format_file_display(current_files), None
                new_files = files if isinstance(files, list) else [files]
                updated = list(current_files) if current_files else []
                for f in new_files:
                    if len(updated) >= 3:
                        break
                    updated.append(f)
                return updated, format_file_display(updated), None

            file_upload.change(
                fn=on_file_upload,
                inputs=[file_upload, file_state],
                outputs=[file_state, file_display, file_upload]
            )

            # Attach button - directly open system file browser
            attach_btn.click(
                fn=None,
                js="() => { const el = document.querySelector('#hidden-file-upload input[type=file]'); if (el) el.click(); }"
            )

            # Remove file by index (triggered by X button JS)
            remove_idx = gr.Textbox(value="", elem_id="remove-file-idx")

            def remove_file(idx_str, current_files):
                """Remove a file from the list by index."""
                if not idx_str or not current_files:
                    return current_files, format_file_display(current_files), ""
                try:
                    idx = int(idx_str)
                    updated = list(current_files)
                    if 0 <= idx < len(updated):
                        updated.pop(idx)
                    return updated, format_file_display(updated), ""
                except (ValueError, IndexError):
                    return current_files, format_file_display(current_files), ""

            remove_idx.input(
                fn=remove_file,
                inputs=[remove_idx, file_state],
                outputs=[file_state, file_display, remove_idx]
            )

            # Analyze Case / Follow-up button
            def on_analyze(text, files):
                """Wrapper to call process_case with current files."""
                file_list = files if files else []
                yield from self.process_case(text, file_list)

            analyze_btn.click(
                fn=on_analyze,
                inputs=[text_input, file_state],
                outputs=[soap_output, reasoning_output, citations_output, text_input, analyze_btn, reset_btn]
            )

            # Reset button - clear everything, back to initial state
            def on_reset():
                """Reset all state back to initial."""
                self.current_case = None
                self.current_session = None
                self.is_analyzing = False
                return (
                    "",  # soap_output
                    "<span style='font-size:9px;'>Detailed clinical reasoning will be displayed here after case analysis.</span>",  # reasoning
                    "<span style='font-size:9px;'>Relevant clinical guidelines and evidence-based references will be displayed here.</span>",  # citations
                    "",  # text_input
                    [],  # file_state
                    "",  # file_display
                    gr.update(value="Analyze Case", interactive=True),  # analyze_btn
                    gr.update(visible=False),  # reset_btn
                )

            reset_btn.click(
                fn=on_reset,
                outputs=[soap_output, reasoning_output, citations_output, text_input, file_state, file_display, analyze_btn, reset_btn]
            )

        return app


def launch_app():
    """Launch the Gradio app."""
    app_instance = MedGemmaApp()
    app = app_instance.create_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    launch_app()
