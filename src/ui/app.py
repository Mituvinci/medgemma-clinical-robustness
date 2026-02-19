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
        - Clinical Reasoning: google/medgemma-1.5-4b-it (local GPU, weights cached)
          Switch to "medgemma-vertex" for Vertex AI demo recording.
        """
        # agent_model options:
        #   "medgemma-1.5-4b"  — local GPU, cached weights (testing)
        #   "medgemma-hf"      — HuggingFace Inference API, login node, no GPU
        #   "medgemma-vertex"  — Vertex AI endpoint (demo recording)
        agent_model = settings.agent_model
        self.workflow = create_workflow(
            model_name="gemini-2.0-flash",  # RPD=1000 (gemini-pro-latest = RPD=50, too low)
            use_medgemma=True,
            agent_model=agent_model
        )
        logger.info(f"MedGemmaApp initialized with agent_model={agent_model}")
        self.current_case = None
        self.current_session = None  # Session object (not just ID)
        self.conversation_history = []
        self.is_analyzing = False
        self.in_followup_mode = False  # Track if we're waiting for follow-up
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

        # STATE: Analyzing — keep user's text visible so they know their input was received
        yield (
            "Processing your case...",  # soap_output
            "",  # reasoning_output
            "",  # citations_output
            gr.update(interactive=False),  # text_input: keep text, just lock it
            gr.update(value="Analyzing...", interactive=False),  # analyze_btn
            gr.update(visible=False),  # reset_btn
            gr.update(interactive=False),  # camera_btn
            gr.update(interactive=False),  # attach_btn
        )

        try:
            self.current_case = ClinicalCase(**case_data)

            logger.info(f"Starting workflow for {session_id}")
            result = asyncio.run(self.workflow.run_async(self.current_case))

            # run_async returns "response" (full agent output), not "soap_note"
            response_text = result.get("response", "No diagnosis generated.")
            soap_note = response_text
            is_pause = self._detect_missing_data(response_text)
            reasoning = self._extract_thinking_process(response_text, result)
            citations = self._extract_citations(response_text, is_pause=is_pause, result=result)
            if is_pause:
                pause_msg = response_text  # Agent's own message asking for info

                # Store the session OBJECT (not just ID) for continuation
                self.current_session = result.get("_session")
                self.in_followup_mode = True
                self.is_analyzing = False

                yield (
                    self._format_clarification_request(pause_msg),
                    reasoning,
                    citations,
                    gr.update(value="", interactive=True),
                    gr.update(value="Submit Follow-up", interactive=True),
                    gr.update(visible=True),
                    gr.update(interactive=True),  # camera_btn
                    gr.update(interactive=True),  # attach_btn
                )
            else:
                # Completed
                self.is_analyzing = False
                self.in_followup_mode = False
                self.current_session = None

                yield (
                    self._format_soap_response(soap_note),
                    reasoning,
                    citations,
                    gr.update(value="", interactive=True),
                    gr.update(value="Analyze Case", interactive=True),
                    gr.update(visible=False),
                    gr.update(interactive=True),  # camera_btn
                    gr.update(interactive=True),  # attach_btn
                )

        except Exception as e:
            logger.error(f"Error in process_case: {e}")
            self.is_analyzing = False
            self.in_followup_mode = False
            self.current_session = None

            yield (
                f"Error: {str(e)}",
                "",
                "",
                gr.update(value="", interactive=True),
                gr.update(value="Analyze Case", interactive=True),
                gr.update(visible=False),
                gr.update(interactive=True),  # camera_btn
                gr.update(interactive=True),  # attach_btn
            )

    def process_followup(self, user_response: str):
        """
        Process follow-up response after agentic pause.
        Continues existing session instead of creating new one.
        """
        if not self.in_followup_mode or not self.current_session:
            logger.error("process_followup called but not in follow-up mode")
            yield (
                "Error: Not in follow-up mode. Please start a new case.",
                "",
                "",
                "",
                gr.update(value="Analyze Case", interactive=True),
                gr.update(visible=False),
            )
            return

        self.is_analyzing = True

        # STATE: Analyzing — keep user's follow-up text visible while processing
        yield (
            "Processing your follow-up response...",
            "",
            "",
            gr.update(interactive=False),  # lock textbox, keep text visible
            gr.update(value="Analyzing...", interactive=False),
            gr.update(visible=False),
            gr.update(interactive=False),  # camera_btn
            gr.update(interactive=False),  # attach_btn
        )

        try:
            logger.info(f"Continuing session {self.current_session.session_id} with follow-up")

            # Continue existing session with user's response
            result = asyncio.run(
                self.workflow.run_async(
                    self.current_case,
                    user_message=f"Additional information from user: {user_response}",
                    existing_session=self.current_session
                )
            )

            response_text = result.get("response", "No response generated.")
            soap_note = response_text
            # Check if STILL paused (might need more info)
            is_still_paused = self._detect_missing_data(response_text)
            reasoning = self._extract_thinking_process(response_text, result)
            citations = self._extract_citations(response_text, is_pause=is_still_paused, result=result)

            if is_still_paused:
                # Still missing data
                logger.info("Agent still requesting more information")
                self.is_analyzing = False

                yield (
                    self._format_clarification_request(response_text),
                    reasoning,
                    citations,
                    gr.update(value=None, interactive=True),
                    gr.update(value="Submit Follow-up", interactive=True),
                    gr.update(visible=True),
                    gr.update(interactive=True),  # camera_btn
                    gr.update(interactive=True),  # attach_btn
                )
            else:
                # Completed! Save the session now
                logger.info(f"Follow-up complete. Saving session {self.current_session.session_id}")

                result["_session"].set_final_output(result)
                from src.agents.conversation_manager import get_conversation_manager
                get_conversation_manager().complete_session(self.current_session.session_id, save=True)

                self.is_analyzing = False
                self.in_followup_mode = False
                self.current_session = None

                yield (
                    self._format_soap_response(soap_note),
                    reasoning,
                    citations,
                    gr.update(value=None, interactive=True),
                    gr.update(value="Analyze Case", interactive=True),
                    gr.update(visible=False),
                    gr.update(interactive=True),  # camera_btn
                    gr.update(interactive=True),  # attach_btn
                )

        except Exception as e:
            logger.error(f"Error in process_followup: {e}")
            self.is_analyzing = False
            self.in_followup_mode = False
            self.current_session = None

            yield (
                f"Error processing follow-up: {str(e)}",
                "",
                "",
                gr.update(value=None, interactive=True),
                gr.update(value="Analyze Case", interactive=True),
                gr.update(visible=False),
                gr.update(interactive=True),  # camera_btn
                gr.update(interactive=True),  # attach_btn
            )

    def _detect_missing_data(self, response: str) -> bool:
        """
        Detect if agent is requesting missing data (agentic pause).

        IMPROVED DETECTION:
        - Checks for questions (contains '?')
        - Checks for missing SOAP sections (no complete diagnosis)
        - Checks for explicit missing data keywords

        Returns True if agent needs more information.
        """
        # Never treat error messages or very short garbage as a pause
        if len(response.strip()) < 50:
            return False
        if response.lower().startswith("error") or "medgemma" in response[:60].lower() and "failed" in response[:120].lower():
            return False

        response_lower = response.lower()

        # Check 1: Does response contain questions?
        has_questions = "?" in response

        # Check 2: Does response have complete SOAP sections?
        has_subjective = "subjective" in response_lower
        has_assessment = "assessment" in response_lower
        has_complete_soap = has_subjective and has_assessment

        # Check 3: Explicit missing data keywords
        missing_keywords = [
            "missing", "insufficient", "clarification",
            "please provide", "could you", "would you",
            "need more", "require additional", "lacking"
        ]
        has_missing_keywords = any(keyword in response_lower[:500] for keyword in missing_keywords)

        # Agentic pause if:
        # - Has questions AND no complete SOAP, OR
        # - Has explicit missing data keywords
        is_pause = (has_questions and not has_complete_soap) or has_missing_keywords

        if is_pause:
            logger.info(f"Agentic pause detected: questions={has_questions}, soap={has_complete_soap}, keywords={has_missing_keywords}")

        return is_pause

    def _format_clarification_request(self, response: str) -> str:
        """Format agent's clarification request — show ONLY the questions, strip internal orchestration text."""
        import json as _json
        stripped = response.strip()
        if stripped.startswith('{'):
            try:
                data = _json.loads(stripped)
                for val in data.values():
                    if isinstance(val, dict):
                        response = val.get("soap_note") or val.get("differential_diagnoses") or response
                        break
            except Exception:
                pass
        clean_response = self._strip_orchestration_text(response)

        formatted = "## Please Provide Additional Details\n\n"
        formatted += "To give you an accurate assessment, the specialist needs a bit more information:\n\n"
        formatted += "---\n\n"
        formatted += clean_response
        formatted += "\n\n---\n\n"
        formatted += "_Please type your answers in the box above and click **Submit Follow-up**._\n"
        return formatted

    def _format_agentic_pause(self, missing_items: List[str], questions: List[str]) -> str:
        """Format agentic pause message (legacy - keeping for compatibility)."""
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

    def _strip_orchestration_text(self, text: str) -> str:
        """Remove internal ADK orchestration lines that Gemini echoes back from its system prompt."""
        import re
        clean_lines = []
        for line in text.split('\n'):
            s = line.strip()
            if re.match(r'^Step\s+\d+\s*:', s):
                continue
            if s.upper().startswith('CALL '):
                continue
            if re.match(r'^okay,?\s+i\s+will\b', s, re.IGNORECASE):
                continue
            if 'transfer_to_agent' in s.lower():
                continue
            if re.search(r'medgemma_\w+\s+tool', s, re.IGNORECASE):
                continue
            clean_lines.append(line)
        result = '\n'.join(clean_lines).strip()
        return re.sub(r'\n{3,}', '\n\n', result)

    def _format_soap_response(self, response: str) -> str:
        """Format SOAP note with Markdown styling, stripping orchestration noise and MedGemma preambles."""
        import re

        import json as _json

        # If Gemini dumped the raw tool-result JSON, extract the SOAP note from it
        stripped = response.strip()
        if stripped.startswith('{'):
            try:
                data = _json.loads(stripped)
                # ADK wraps as {"medgemma_clinical_diagnosis_response": {...}}
                for val in data.values():
                    if isinstance(val, dict):
                        # Prefer differential_diagnoses (actual output), fall back to soap_note
                        soap = val.get("differential_diagnoses") or val.get("soap_note", "")
                        if soap:
                            response = soap
                        break
            except Exception:
                pass  # not valid JSON, proceed with original

        formatted = self._strip_orchestration_text(response)

        # Strip prompt echo markers that Vertex AI one-click-deploy adds
        for marker in ["Final Output:\n", "Final Output:\r\n"]:
            if formatted.startswith(marker):
                formatted = formatted[len(marker):].strip()

        # Strip Gemini's wrapper phrase: "Here is the complete SOAP note from MedGemma specialist:"
        formatted = re.sub(
            r'(?i)here\s+is\s+the\s+complete\s+soap\s+note\s+from\s+\w+.*?:\s*', '', formatted
        )

        # Strip MedGemma small-model preambles:
        formatted = re.sub(r'(?i)ensure\s+all\s+required\s+elements.*?\n', '', formatted)
        formatted = re.sub(
            r'(?i)\*?\*?Constraint Checklist.*?(?=\n---|\Z)',
            '', formatted, flags=re.DOTALL
        )
        formatted = re.sub(r'(?i)strategizing\s+complete\..*?\n', '', formatted)

        # Strip header metadata block with placeholder values
        # e.g. **Patient Name:** [Patient Name] / **Date of Visit:** [Date] / **MRN:** [MRN]
        formatted = re.sub(r'(?m)^\*\*Patient Name:\*\*.*$', '', formatted)
        formatted = re.sub(r'(?m)^\*\*Date of Visit:\*\*.*$', '', formatted)
        formatted = re.sub(r'(?m)^\*\*Referring Physician.*:\*\*.*$', '', formatted)
        formatted = re.sub(r'(?m)^\*\*Medical Record Number.*:\*\*.*$', '', formatted)
        # Also strip any line that is just a [placeholder] bracket value
        formatted = re.sub(r'(?m)^\*\*[\w\s]+:\*\*\s*\[[\w\s]+\]\s*$', '', formatted)

        # Remove leading "---" separators and collapse blank lines
        formatted = re.sub(r'^\s*---\s*\n', '', formatted, flags=re.MULTILINE)
        formatted = re.sub(r'\n{3,}', '\n\n', formatted).strip()

        # Bold SOAP section headers if plain text
        if "Subjective" in formatted and "**Subjective" not in formatted:
            formatted = formatted.replace("Subjective (S):", "## Subjective (S)")
            formatted = formatted.replace("Objective (O):", "## Objective (O)")
            formatted = formatted.replace("Assessment (A):", "## Assessment (A)")
            formatted = formatted.replace("Plan (P):", "## Plan (P)")

        return formatted

    def _extract_thinking_process(self, response: str, result: dict = None) -> str:
        """Build a readable, scrollable agent trace from actual agent_steps in the result."""
        result = result or {}
        model_name = result.get("model", "gemini-2.0-flash")
        agent_steps = result.get("agent_steps", [])

        trace = "### Clinical Reasoning Pipeline\n\n"
        trace += f"**Orchestrator**: {model_name} (Google ADK)  \n"
        trace += f"**Clinical AI**: MedGemma (google/medgemma-1.5-4b-it via Vertex AI)\n\n"
        trace += "---\n\n"

        TOOL_LABELS = {
            "analyze_case_completeness":  "🔍 Assembling case data",
            "medgemma_triage_analysis":   "🧠 MedGemma: Triage analysis",
            "retrieve_clinical_guidelines": "📚 RAG: Retrieving clinical guidelines",
            "medgemma_guideline_synthesis": "🧠 MedGemma: Synthesizing guidelines",
            "medgemma_clinical_diagnosis":  "🧠 MedGemma: Generating SOAP note",
            "transfer_to_agent":           "→ Transferring to next agent",
        }

        if not agent_steps:
            trace += "_No agent steps recorded._\n"
            return trace

        for step in agent_steps:
            agent = step.get("agent", "Unknown")
            fcs = step.get("function_calls", [])
            step_text = step.get("response", "").strip()

            if not fcs and not step_text:
                continue

            trace += f"**{agent}**\n"

            for fc in fcs:
                fc_name = fc.get("name", "")
                label = TOOL_LABELS.get(fc_name, f"⚙️ {fc_name}")

                # For transfer, show destination
                if fc_name == "transfer_to_agent":
                    args = fc.get("args", "")
                    dest = ""
                    import re
                    m = re.search(r"'agent_name':\s*'([^']+)'", args)
                    if m:
                        dest = m.group(1)
                    trace += f"- {label} **{dest}**\n"
                else:
                    trace += f"- {label}\n"

            # For MedGemma steps, show a snippet of what MedGemma returned
            if any("medgemma" in fc.get("name", "").lower() for fc in fcs):
                # Pull MedGemma output from the stored _run_outputs (not step text)
                # step_text here is usually the tool-call line, not the response
                # Show first 300 chars of meaningful text if available
                clean = step_text.replace("[Tool Calls:", "").strip()
                if clean and len(clean) > 20 and "medgemma_" not in clean[:30]:
                    trace += f"\n> {clean[:300]}{'...' if len(clean) > 300 else ''}\n"
            trace += "\n"

        trace += "---\n\n"
        trace += f"_{len(agent_steps)} agent interactions logged_"
        return trace

    def _extract_citations(self, response: str, is_pause: bool = False, result: dict = None) -> str:
        """Extract RAG citations from agent_steps (RAG tool args) and from MedGemma response text."""
        import re
        result = result or {}
        agent_steps = result.get("agent_steps", [])

        citations_md = "### Evidence-Based Clinical Guidelines\n\n"

        if is_pause:
            citations_md += "*No guidelines retrieved — pipeline paused at Triage stage.*\n\n"
            citations_md += "RAG retrieval only runs when clinical data is sufficient for diagnosis.\n\n"
            citations_md += "Provide the requested information and the Research Agent will query:\n"
            citations_md += "- AAD Clinical Practice Guidelines\n"
            citations_md += "- StatPearls Medical Reference\n"
            citations_md += "- JAAD Case Reports\n"
            return citations_md

        # Extract guideline titles from the RAG step's tool call args (most reliable source)
        rag_docs = []
        for step in agent_steps:
            for fc in step.get("function_calls", []):
                if fc.get("name") == "retrieve_clinical_guidelines":
                    # The args string contains the query used
                    args = fc.get("args", "")
                    m = re.search(r"'query':\s*'([^']+)'", args)
                    if m:
                        rag_docs.append(("Query used", m.group(1)))
                elif fc.get("name") == "medgemma_guideline_synthesis":
                    # Extract doc titles from args
                    args = fc.get("args", "")
                    titles = re.findall(r"'title':\s*'([^']+)'", args)
                    sources = re.findall(r"'source':\s*'([^']+)'", args)
                    scores = re.findall(r"'similarity_score':\s*([\d.]+)", args)
                    for i, title in enumerate(titles):
                        src = sources[i] if i < len(sources) else "AAD"
                        score = float(scores[i]) if i < len(scores) else 0.0
                        rag_docs.append((src, title, score))

        # Also search response text for inline citations
        inline_sources = re.findall(r'Source:\s*([^\n\r]+)', response, re.IGNORECASE)
        aad_inline = re.findall(r'(?:AAD Guidelines?|American Academy of Dermatology)[:\-]\s*([^\n\r\.]+)', response, re.IGNORECASE)
        sp_inline = re.findall(r'StatPearls[:\-]\s*([^\n\r\.]+)', response, re.IGNORECASE)
        inline_all = list(dict.fromkeys([s.strip() for s in inline_sources + aad_inline + sp_inline if s.strip()]))

        if rag_docs:
            citations_md += "*Documents retrieved from RAG corpus (Vertex AI RAG):*\n\n"
            seen_titles = set()
            for item in rag_docs:
                if item[0] == "Query used":
                    citations_md += f"**Search query**: `{item[1]}`\n\n"
                    citations_md += "**Retrieved documents:**\n\n"
                elif len(item) == 3:
                    src, title, score = item
                    # Strip PDF extension for readability
                    clean_title = re.sub(r'\.pdf$', '', title, flags=re.IGNORECASE).replace('_', ' ').replace('-', ' ')
                    if clean_title not in seen_titles:
                        seen_titles.add(clean_title)
                        citations_md += f"- **{src}** — {clean_title} *(similarity: {score:.2f})*\n"
            citations_md += "\n"

        if inline_all:
            citations_md += "*Cited by MedGemma in response:*\n\n"
            for src in inline_all:
                citations_md += f"- {src}\n"
            citations_md += "\n"

        if not rag_docs and not inline_all:
            citations_md += "*No citations extracted.*\n\n"
            citations_md += "**Knowledge bases available:**\n"
            citations_md += "- AAD Clinical Practice Guidelines\n"
            citations_md += "- StatPearls Medical Reference\n"
            citations_md += "- JAAD Case Reports\n"

        citations_md += "\n---\n\n"
        citations_md += "*RAG corpus: AAD guidelines, StatPearls, JAAD Case Reports (Vertex AI RAG, us-west1)*"
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
#acc-reasoning button,
#acc-citations button {
    font-size: 18px !important;
}

/* ===== CAMERA & ATTACH BUTTONS — light gray background ===== */
#camera-btn button, #attach-btn button {
    background-color: #e8e8e8 !important;
    border-color: #d0d0d0 !important;
}
#camera-btn button:hover, #attach-btn button:hover {
    background-color: #dcdcdc !important;
}

/* ===== ASSESSMENT GROUP ===== */
#assessment-group {
    gap: 2px !important;
    background: transparent !important;
    font-size: 18px !important;
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

/* ===== SOAP OUTPUT — scrollable box, never expands the page ===== */
#soap-output {
    max-height: 45vh !important;
    overflow-y: auto !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
    background: #fafafa !important;
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
                    "<span style='font-size:24px; font-weight:600;  padding-bottom:8px;'>MedGemma Clinical Robustness Assistant</span><br>"
                    "<span style='font-size:16px; color:#666;'>"
                    "Powered by MedGemma Multimodal Specialist Suite (Health-Specialized AI) | "
                    "Multi-Agent: Triage → Research → Diagnosis "
                    "Evidence-Based: 3,091 Clinical Guidelines (AAD, StatPearls, JAADCR)"
                    "</span><br>"
                    "<span style='font-size:16px; color:#999;  padding-bottom:8px;'>"
                    "Upload image/file or use manual input. Agent will request missing data before diagnosing.<br><br>"
                    "<strong>DISCLAIMER:</strong> Research and demonstration purposes only. Not for clinical use."
                    "</span>"
		    "</div>"
                )

                #gr.Markdown("---")

                # SECTION 2: CLINICAL ASSESSMENT
                gr.Markdown(
		    "<div style='padding-left:16px; padding-top:8px;  padding-bottom:8px;'>"
                    "<span style='font-size:18px; font-weight:600;'>Clinical Assessment</span><br>"
                    "<span style='font-size:16px; color:#666;'>*SOAP note will appear here after analysis...*</span>"
		    "</div>"
                )

                soap_output = gr.Markdown(value="", elem_id="soap-output")

                with gr.Group(elem_id="assessment-group"):
                    with gr.Accordion("Clinical Reasoning Trace", open=False, elem_id="acc-reasoning"):
                        reasoning_output = gr.Markdown(
                            value="<span style='font-size:16px;'>Detailed clinical reasoning will be displayed here after case analysis.</span>"
                        )

                    with gr.Accordion("Evidence-Based Guidelines (AAD, StatPearls, JAADCR)", open=False, elem_id="acc-citations"):
                        citations_output = gr.Markdown(
                            value="<span style='font-size:16px;'>Relevant clinical guidelines and evidence-based references will be displayed here.</span>"
                        )

                #gr.Markdown("---")

                # SECTION 4: EXAMPLES
                # Plain Gradio buttons — no JS tricks, no hidden buttons, just direct .click() handlers.
                # CSS makes them look like text rows (full-width, left-aligned, small font).
                _CASE1_FULL = (
                    "A 73-year-old man presented to the hospital with a rash of 2 and a half-weeks duration. "
                    "It started 2 days after receiving his first Pfizer COVID-19 vaccine in his left deltoid. "
                    "He reported associated pruritus and burning sensation. He denied systemic symptoms and denied "
                    "known history of SARS-CoV-2 infection. He denied recent intake of new medications or exposure "
                    "to potential allergens. Physical exam demonstrated morbilliform papules that coalesced into an "
                    "erythematous plaque with sharp demarcation on his left proximal arm and trunk with fine overlying "
                    "desquamation and relative sparing of the axillary vault. There was no mucosal, facial, or acral "
                    "involvement. He was hemodynamically stable, afebrile, and routine laboratory studies were "
                    "unremarkable on admission. Act as a dermatologist and determine what is the diagnosis."
                )
                _CASE2_FULL = (
                    "A 10-year-old previously healthy male presented with a 5-day history of itchy, tense bullae "
                    "to arms, trunk, and face. New blisters can be seen at the periphery of older lesions forming "
                    "annular and arciform configurations with central crusting. He had no recent history of infections "
                    "or exposure to medications, including antibiotics or nonsteroidal anti-inflammatory drugs. "
                    "Act as a dermatologist and determine what is the diagnosis."
                )

                gr.HTML("""<style>
                /* Style example buttons as clickable text rows, flush to left */
                #example-btn-0, #example-btn-1,
                #example-btn-0 *, #example-btn-1 * {
                    padding-left: 0 !important;
                    margin-left: 0 !important;
                }
                #example-btn-0 button, #example-btn-1 button {
                    width: 100% !important;
                    text-align: left !important;
                    white-space: normal !important;
                    word-wrap: break-word !important;
                    font-size: 10px !important;
                    font-weight: normal !important;
                    padding: 6px 0 !important;
                    background: transparent !important;
                    border: none !important;
                    border-radius: 0 !important;
                    color: #333 !important;
                    box-shadow: none !important;
                    line-height: 1.4 !important;
                    display: block !important;
    		    padding-left: 0 !important;
                    margin-left: 0 !important;
                    text-align: left !important;
                }
                #example-btn-0 button:hover, #example-btn-1 button:hover {
                    background: #f5f5f5 !important;
                    padding-left: 0 !important;
                    margin-left: 0 !important;
                    text-align: left !important;
                }
                #example-divider {
                    border: none;
                    border-top: 1px solid #e8e8e8;
                    margin: 0;
                }
               #example-btn-0 .gr-button,
               #example-btn-1 .gr-button {
                     padding-left: 0 !important;
               }
                </style>
                <div style='padding:4px;'>
                  <span style='font-size:18px; font-weight:600;'>Example Cases (Click to load):</span>
                </div>""")

                example_btn_0 = gr.Button(
                    value=(
                        "Case 1 : 73M, post-vaccine rash:  "
                        + _CASE1_FULL[:200] + "..."
                    ),
                    variant="secondary",
                    elem_id="example-btn-0",
                )
                gr.HTML('<hr id="example-divider">')
                example_btn_1 = gr.Button(
                    value=(
                        "Case 2 : 10M, tense bullae:  "
                        + _CASE2_FULL[:200] + "..."
                    ),
                    variant="secondary",
                    elem_id="example-btn-1",
                )
                #gr.Markdown("---")

             

            # CAMERA POPUP — appears above input section when visible (gr.Group toggle pattern)
            with gr.Group(visible=False, elem_id="camera-popup") as camera_popup:
                gr.Markdown(
                    "<div style='padding:6px 16px 2px;'>"
                    "<span style='font-size:11px; font-weight:600;'>📷 Capture Photo</span>"
                    "</div>"
                )
                webcam_img = gr.Image(
                    sources=["webcam"],
                    type="filepath",
                    label=None,
                    show_label=False,
                    height=220,
                )
                with gr.Row():
                    camera_use_btn = gr.Button("Use Photo", variant="primary", size="sm", scale=1)
                    camera_cancel_btn = gr.Button("Cancel", variant="secondary", size="sm", scale=1)

            # PINNED BOTTOM INPUT SECTION
            with gr.Group(elem_id="input-section"):
                gr.Markdown(
                    "<div style='padding-left:16px; padding-top:8px; padding-bottom:4px;'>"
                    "<span style='font-size:18px; font-weight:600;'>Clinical Case Input</span>"
                    "</div>"
                )

                text_input = gr.Textbox(
                    placeholder=(
                        "Describe your case (history, exam findings, age, symptoms, duration...) "
                        "and/or attach files (image, PDF, JSON, text -- max 3 files). "
                        "Or click an example from above for quick analysis."
                    ),
                    lines=5,
                    max_lines=5,
                    show_label=False,
                    elem_id="case-input",
                )

                # State to track attached files (list of file paths)
                file_state = gr.State([])

                # Row showing attached filenames/thumbnails with X buttons
                file_display = gr.HTML(value="", elem_id="file-display")

                # Hidden remove buttons (one per slot, max 3 files)
                with gr.Row(visible=False):
                    remove_btn_0 = gr.Button("x0", elem_id="remove-btn-0", size="sm")
                    remove_btn_1 = gr.Button("x1", elem_id="remove-btn-1", size="sm")
                    remove_btn_2 = gr.Button("x2", elem_id="remove-btn-2", size="sm")

                with gr.Row(elem_id="input-btn-row"):
                    camera_btn = gr.Button("📷 Camera", variant="secondary", scale=1, size="lg", elem_id="camera-btn")
                    attach_btn = gr.Button("📎 Attach", variant="secondary", scale=1, size="lg", elem_id="attach-btn")
                    analyze_btn = gr.Button("Analyze Case", variant="primary", scale=1, size="lg")
                    reset_btn = gr.Button("Reset", variant="secondary", scale=0, visible=False, size="lg")

            # EVENT HANDLERS

            _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

            def format_file_display(file_list):
                """Generate HTML chips for attached files.
                - Images: tiny 28×28 square thumbnail + truncated name
                - Other files: filename chip (truncated at 80 chars)
                X button clicks the hidden Gradio remove button for that slot.
                """
                if not file_list:
                    return ""
                html_parts = []
                for i, fp in enumerate(file_list):
                    name = os.path.basename(fp)
                    ext = os.path.splitext(name)[1].lower()
                    is_image = ext in _IMAGE_EXTS

                    # Truncate name at 80 chars
                    display_name = name if len(name) <= 80 else name[:80] + "..."

                    thumb_html = ""
                    if is_image:
                        thumb_html = (
                            f"<img src='/file={fp}' "
                            f"style='width:28px;height:28px;object-fit:cover;"
                            f"border-radius:3px;margin-right:5px;vertical-align:middle;'>"
                        )

                    html_parts.append(
                        f"<span style='display:inline-flex; align-items:center; "
                        f"background:#f0f0f0; border:1px solid #ddd; border-radius:4px; "
                        f"padding:2px 6px; margin:2px 4px; font-size:11px; max-width:280px;'>"
                        f"{thumb_html}"
                        f"<span style='overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                        f"max-width:180px;vertical-align:middle;'>{display_name}</span>"
                        f"<button onclick=\"document.querySelector('#remove-btn-{i} button').click();\""
                        f" style='border:none;background:none;cursor:pointer;"
                        f"margin-left:5px;font-size:14px;color:#999;line-height:1;flex-shrink:0;'>×</button>"
                        f"</span>"
                    )
                return "<div style='padding:4px 0; display:flex; flex-wrap:wrap;'>" + "".join(html_parts) + "</div>"

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

            # Camera button — show camera popup
            camera_btn.click(
                fn=lambda: gr.update(visible=True),
                outputs=[camera_popup]
            )

            # Cancel — hide camera popup without saving
            camera_cancel_btn.click(
                fn=lambda: (gr.update(visible=False), None),
                outputs=[camera_popup, webcam_img]
            )

            # Use Photo — save captured image with timestamp name, add to file_state
            import shutil, datetime as _dt

            def on_use_photo(img_path, current_files):
                """Save captured image, add to file list, hide camera popup."""
                if not img_path:
                    return current_files, format_file_display(current_files), gr.update(visible=False), None
                os.makedirs("data/temp_images", exist_ok=True)
                timestamp = _dt.datetime.now().strftime("%H-%M-%S")
                dest = os.path.join("data/temp_images", f"captured_{timestamp}.jpeg")
                shutil.copy2(img_path, dest)
                updated = list(current_files) if current_files else []
                if len(updated) < 3:
                    updated.append(dest)
                return updated, format_file_display(updated), gr.update(visible=False), None

            camera_use_btn.click(
                fn=on_use_photo,
                inputs=[webcam_img, file_state],
                outputs=[file_state, file_display, camera_popup, webcam_img]
            )

            # Attach button - directly open system file browser
            attach_btn.click(
                fn=None,
                js="() => { const el = document.querySelector('#hidden-file-upload input[type=file]'); if (el) el.click(); }"
            )

            # Remove file handlers — one per slot (max 3 files)
            def make_remove_fn(slot_idx):
                def remove_file(current_files):
                    updated = list(current_files) if current_files else []
                    if 0 <= slot_idx < len(updated):
                        updated.pop(slot_idx)
                    return updated, format_file_display(updated)
                return remove_file

            for _slot, _btn in enumerate([remove_btn_0, remove_btn_1, remove_btn_2]):
                _btn.click(
                    fn=make_remove_fn(_slot),
                    inputs=[file_state],
                    outputs=[file_state, file_display]
                )

            # Example case buttons — load full case text into textbox
            example_btn_0.click(fn=lambda: gr.update(value=_CASE1_FULL), outputs=[text_input])
            example_btn_1.click(fn=lambda: gr.update(value=_CASE2_FULL), outputs=[text_input])

            # Analyze Case / Follow-up button (routes based on mode)
            def on_analyze(text, files):
                """
                Route to either process_case (new case) or process_followup (continuation).
                Determines routing based on in_followup_mode flag.
                """
                if self.in_followup_mode:
                    # Follow-up mode: continue existing session
                    yield from self.process_followup(text)
                else:
                    # New case mode
                    file_list = files if files else []
                    yield from self.process_case(text, file_list)

            analyze_btn.click(
                fn=on_analyze,
                inputs=[text_input, file_state],
                outputs=[soap_output, reasoning_output, citations_output, text_input, analyze_btn, reset_btn, camera_btn, attach_btn]
            )

            # Reset button - clear everything, back to initial state
            def on_reset():
                """Reset all state back to initial."""
                self.current_case = None
                self.current_session = None
                self.is_analyzing = False
                self.in_followup_mode = False
                return (
                    "",  # soap_output
                    "<span style='font-size:9px;'>Detailed clinical reasoning will be displayed here after case analysis.</span>",
                    "<span style='font-size:9px;'>Relevant clinical guidelines and evidence-based references will be displayed here.</span>",
                    "",  # text_input
                    [],  # file_state
                    "",  # file_display
                    gr.update(value="Analyze Case", interactive=True),  # analyze_btn
                    gr.update(visible=False),  # reset_btn
                    gr.update(interactive=True),  # camera_btn
                    gr.update(interactive=True),  # attach_btn
                )

            reset_btn.click(
                fn=on_reset,
                outputs=[soap_output, reasoning_output, citations_output, text_input, file_state, file_display, analyze_btn, reset_btn, camera_btn, attach_btn]
            )

        return app


def launch_app():
    """Launch the Gradio app."""
    app_instance = MedGemmaApp()
    app = app_instance.create_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    launch_app()
