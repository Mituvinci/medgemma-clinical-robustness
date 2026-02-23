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
import datetime as _dt
import html as _html
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image

# Constants
MAX_TEXT_LENGTH = 5_000        # ~1,250 tokens — covers any clinical case
MAX_IMAGE_PDF_SIZE_MB = 5      # 5MB for images and PDFs
MAX_TEXT_FILE_SIZE_KB = 5      # 5KB for .json and .txt files
MAX_FILES = 3
TEMP_IMAGE_DIR = "data/temp_images"
TEMP_IMAGE_MAX_AGE_HOURS = 2   # Clean up temp images older than this

_PLACEHOLDER_DEFAULT = (
    "Describe your case (history, exam findings, age, symptoms, duration...) "
    "and/or attach files (image, PDF, JSON, text -- max 3 files). "
    "Or click an example from above for quick analysis."
)
_PLACEHOLDER_WORKING = "MedGemma Clinical Robustness is analyzing your case... please wait"

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
        self.is_analyzing = False
        self.in_followup_mode = False  # Track if we're waiting for follow-up

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

    @staticmethod
    def _validate_text(text: str) -> str:
        """Validate and sanitize text input. Returns cleaned text or raises ValueError."""
        if not text:
            return ""
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text input too long ({len(text):,} chars). Maximum is {MAX_TEXT_LENGTH:,} characters.")
        return text.strip()

    @staticmethod
    def _validate_file(file_path: str) -> str:
        """Validate uploaded file size by type. Returns empty string if valid, error message if not."""
        try:
            name = os.path.basename(file_path)
            ext = os.path.splitext(name)[1].lower()
            size_bytes = os.path.getsize(file_path)
            size_kb = size_bytes / 1024
            size_mb = size_bytes / (1024 * 1024)

            if ext in ('.json', '.txt'):
                if size_kb > MAX_TEXT_FILE_SIZE_KB:
                    return f"File '{name}' is too large ({size_kb:.1f}KB). Text/JSON files must be under {MAX_TEXT_FILE_SIZE_KB}KB."
            elif ext in ('.jpg', '.jpeg', '.png', '.pdf'):
                if size_mb > MAX_IMAGE_PDF_SIZE_MB:
                    return f"File '{name}' is too large ({size_mb:.1f}MB). Image/PDF files must be under {MAX_IMAGE_PDF_SIZE_MB}MB."
            return ""
        except OSError:
            return f"Cannot read file: {os.path.basename(file_path)}"

    # ── Session persistence (follow-up survives browser refresh) ─────────

    ACTIVE_SESSION_FILE = Path("data/active_session.json")

    def _save_active_session(self, response_text: str, reasoning_text: str, citations_text: str):
        """Persist follow-up state to disk so it survives browser refresh."""
        try:
            self.ACTIVE_SESSION_FILE.parent.mkdir(exist_ok=True)
            # Flush ConversationSession to logs/sessions/ so .load() can find it
            if self.current_session:
                self.current_session.save()
            data = {
                "version": 1,
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
                "in_followup_mode": True,
                "session_id": self.current_session.session_id if self.current_session else None,
                "case_data": self.current_case.dict(exclude={"image_data"}) if self.current_case else None,
                "last_response_text": response_text,
                "last_reasoning_text": reasoning_text,
                "last_citations_text": citations_text,
            }
            with open(self.ACTIVE_SESSION_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved active session to {self.ACTIVE_SESSION_FILE}")
        except Exception as e:
            logger.error(f"Failed to save active session: {e}")

    def _clear_active_session(self):
        """Remove active session file (case completed or reset)."""
        try:
            if self.ACTIVE_SESSION_FILE.exists():
                self.ACTIVE_SESSION_FILE.unlink()
                logger.info("Cleared active session file")
        except OSError as e:
            logger.debug(f"Could not clear active session file: {e}")

    def _restore_active_session(self):
        """Restore follow-up state from disk on page load.

        Returns 8-tuple matching yield pattern, or None if nothing to restore.
        """
        if not self.ACTIVE_SESSION_FILE.exists():
            return None
        try:
            with open(self.ACTIVE_SESSION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") != 1 or not data.get("in_followup_mode"):
                self._clear_active_session()
                return None
            if not data.get("session_id") or not data.get("case_data"):
                self._clear_active_session()
                return None
            # Staleness check (> 24 hours)
            try:
                saved_time = _dt.datetime.fromisoformat(data["timestamp"].rstrip("Z"))
                age_hours = (_dt.datetime.utcnow() - saved_time).total_seconds() / 3600
                if age_hours > 24:
                    logger.info(f"Active session is {age_hours:.1f}h old, clearing")
                    self._clear_active_session()
                    return None
            except (ValueError, KeyError):
                pass
            # Restore ClinicalCase
            self.current_case = ClinicalCase(**data["case_data"])
            # Restore ConversationSession from logs/sessions/
            from src.agents.conversation_manager import ConversationSession, get_conversation_manager
            self.current_session = ConversationSession.load(data["session_id"])
            get_conversation_manager().active_sessions[data["session_id"]] = self.current_session
            self.in_followup_mode = True
            self.is_analyzing = False
            logger.info(f"Restored active session: {data['session_id']}")
            return (
                data.get("last_response_text", ""),
                data.get("last_reasoning_text", ""),
                data.get("last_citations_text", ""),
                gr.Textbox(value="", interactive=True, placeholder=_PLACEHOLDER_DEFAULT),
                gr.Button(visible=False),                     # analyze_btn: hidden
                gr.Group(visible=True),                       # followup_group: SHOW
                gr.Button(visible=True),                      # reset_btn
                gr.Button(interactive=True),                  # camera_btn
                gr.Button(interactive=True),                  # attach_btn
            )
        except Exception as e:
            logger.error(f"Failed to restore active session: {e}")
            self._clear_active_session()
            return None

    # ── End session persistence ──────────────────────────────────────────

    def build_case_from_inputs(self, text_input: str, file_list: List[str] = None, session_id: str = None) -> Dict[str, Any]:
        """Build case dictionary from text input and uploaded files.
        Raises ValueError with user-friendly message if any file exceeds size limits.
        """
        case_data = {
            "case_id": f"ui_{session_id or 'unknown'}",
            "context_state": "original",
        }

        if text_input:
            case_data["history"] = self._validate_text(text_input)

        # Validate ALL files first — raise on first error so user sees the message
        files = file_list or []
        for file_path in files[:MAX_FILES]:
            error = self._validate_file(file_path)
            if error:
                raise ValueError(error)

        # Process uploaded files (all validated)
        for file_path in files[:MAX_FILES]:
            file_ext = file_path.lower() if isinstance(file_path, str) else file_path.name.lower()

            if any(file_ext.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                case_data["image_path"] = file_path  # path for workflow image attachment
                case_data["image_data"] = Image.open(file_path)
            elif any(file_ext.endswith(ext) for ext in ['.json', '.txt', '.pdf']):
                file_data = self.parse_case_file(file_path)
                if file_data:
                    case_data.update(file_data)

        return case_data

    def process_case(self, text_input: str, file_list: List[str] = None):
        """Process case submission. Generator for streaming updates."""
        # Guard: require at least some input before calling any agent
        has_text = bool(text_input and text_input.strip())
        has_files = bool(file_list and len(file_list) > 0)
        if not has_text and not has_files:
            yield (
                "**Please provide clinical information before submitting.**\n\n"
                "You can:\n"
                "- Type a clinical case description in the text field\n"
                "- Attach a case file (.txt, .json, .pdf)\n"
                "- Attach a clinical image (.jpg, .png)",
                "", "",
                gr.Textbox(interactive=True, placeholder=_PLACEHOLDER_DEFAULT),
                gr.Button(visible=True, interactive=True),   # analyze_btn
                gr.Group(visible=False),                      # followup_group: hidden
                gr.Button(visible=False),                     # reset_btn
                gr.Button(interactive=True),                  # camera_btn
                gr.Button(interactive=True),                  # attach_btn
            )
            return

        # If user uploaded only an image (no text, no case files), inject default text.
        # Track this so we can force an agentic pause regardless of what the model generates —
        # an image alone is never sufficient clinical context for a diagnosis.
        is_image_only_submission = False
        if not has_text and has_files:
            image_exts = ('.jpg', '.jpeg', '.png')
            only_images = all(
                (f.lower() if isinstance(f, str) else f.name.lower()).endswith(image_exts)
                for f in file_list
            )
            if only_images:
                n_images = len(file_list)
                img_word = "image" if n_images == 1 else f"{n_images} images"
                text_input = f"[Image-only submission: {img_word} attached, no case description provided]"
                is_image_only_submission = True

        self.is_analyzing = True

        # Generate session_id once, use for both case_id and workflow
        session_id = self.get_next_session_id()

        # Build case from inputs BEFORE clearing
        case_data = self.build_case_from_inputs(text_input, file_list, session_id)

        # STATE: Analyzing — if user typed text, keep it visible (locked).
        # For image-only submissions the textbox shows a status value (placeholder
        # updates are not reliable in Gradio 6.x generators; setting value is).
        _processing_textbox = (
            gr.Textbox(value=_PLACEHOLDER_WORKING, interactive=False)
            if is_image_only_submission
            else gr.Textbox(interactive=False)
        )
        yield (
            "Processing your case...",  # soap_output
            "",  # reasoning_output
            "",  # citations_output
            _processing_textbox,
            gr.Button(visible=False, interactive=False),  # analyze_btn: hide during processing
            gr.Group(visible=False),                       # followup_group: hidden
            gr.Button(visible=True, interactive=True),     # reset_btn: SHOW during processing
            gr.Button(interactive=False),                  # camera_btn
            gr.Button(interactive=False),                  # attach_btn
        )

        try:
            self.current_case = ClinicalCase(**case_data)

            logger.info(f"Starting workflow for {session_id}")
            import time as _time
            _result = None
            for _attempt in range(3):
                try:
                    _result = asyncio.run(self.workflow.run_async(self.current_case))
                    break
                except Exception as _e:
                    if ("429" in str(_e) or "RESOURCE_EXHAUSTED" in str(_e)) and _attempt < 2:
                        _wait = 20 * (_attempt + 1)
                        logger.warning(f"Pipeline 429 rate limit, retrying in {_wait}s (attempt {_attempt+1}/3)")
                        _time.sleep(_wait)
                    else:
                        raise
            result = _result

            # run_async returns "response" (full agent output), not "soap_note"
            response_text = result.get("response", "No diagnosis generated.")
            soap_note = response_text
            is_pause = self._detect_missing_data(response_text)
            # Image-only submissions must always pause — images alone are never enough
            # for a diagnosis regardless of what the model generates.
            if is_image_only_submission:
                is_pause = True
            reasoning = self._extract_thinking_process(response_text, result)
            citations = self._extract_citations(response_text, is_pause=is_pause, result=result)
            if is_pause:
                pause_msg = response_text  # Agent's own message asking for info

                # Store the session OBJECT (not just ID) for continuation
                self.current_session = result.get("_session")
                self.in_followup_mode = True
                self.is_analyzing = False

                formatted_pause = self._format_clarification_request(pause_msg)
                self._save_active_session(formatted_pause, reasoning, citations)

                logger.info("Yielding PAUSE state — showing 'Submit Follow-up' button")
                yield (
                    formatted_pause,
                    reasoning,
                    citations,
                    gr.Textbox(value="", interactive=True, placeholder=_PLACEHOLDER_DEFAULT),
                    gr.Button(visible=False),                     # analyze_btn: HIDE
                    gr.Group(visible=True),                       # followup_group: SHOW
                    gr.Button(visible=True),                      # reset_btn
                    gr.Button(interactive=True),                  # camera_btn
                    gr.Button(interactive=True),                  # attach_btn
                )
            else:
                # Completed
                self.is_analyzing = False
                self.in_followup_mode = False
                self.current_session = None
                self._clear_active_session()

                yield (
                    self._format_soap_response(soap_note),
                    reasoning,
                    citations,
                    gr.Textbox(value="", interactive=True, placeholder=_PLACEHOLDER_DEFAULT),
                    gr.Button(visible=True, interactive=True),    # analyze_btn: SHOW
                    gr.Group(visible=False),                      # followup_group: hidden
                    gr.Button(visible=True),                      # reset_btn: SHOW after SOAP
                    gr.Button(interactive=True),                  # camera_btn
                    gr.Button(interactive=True),                  # attach_btn
                )

        except Exception as e:
            logger.error(f"Error in process_case: {e}")
            self.is_analyzing = False
            self.in_followup_mode = False
            self.current_session = None
            self._clear_active_session()

            yield (
                f"Error: {str(e)}",
                "",
                "",
                gr.Textbox(value="", interactive=True, placeholder=_PLACEHOLDER_DEFAULT),
                gr.Button(visible=True, interactive=True),    # analyze_btn
                gr.Group(visible=False),                      # followup_group: hidden
                gr.Button(visible=False),                     # reset_btn
                gr.Button(interactive=True),                  # camera_btn
                gr.Button(interactive=True),                  # attach_btn
            )

    def process_followup(self, user_response: str, file_list: List[str] = None):
        """
        Process follow-up response after agentic pause.
        Continues existing session (same case session) instead of creating new one.
        Also processes any newly attached files (text/image) and includes their content.
        """
        if not self.in_followup_mode or not self.current_session:
            logger.error("process_followup called but not in follow-up mode")
            yield (
                "Error: Not in follow-up mode. Please start a new case.",
                "",
                "",
                gr.Textbox(interactive=True, placeholder=_PLACEHOLDER_DEFAULT),
                gr.Button(visible=True, interactive=True),    # analyze_btn
                gr.Group(visible=False),                      # followup_group: hidden
                gr.Button(visible=False),                     # reset_btn
                gr.Button(interactive=True),                  # camera_btn
                gr.Button(interactive=True),                  # attach_btn
            )
            return

        self.is_analyzing = True

        # STATE: Analyzing — keep user's follow-up text visible while processing.
        # If textbox was empty (file-only follow-up), show status as value.
        _has_followup_text = bool(user_response and user_response.strip())
        _fu_processing_textbox = (
            gr.Textbox(value=_PLACEHOLDER_WORKING, interactive=False)
            if not _has_followup_text
            else gr.Textbox(interactive=False)
        )
        yield (
            "Processing your follow-up response...",
            "",
            "",
            _fu_processing_textbox,
            gr.Button(visible=False),                         # analyze_btn: hidden
            gr.Group(visible=False),                          # followup_group: hidden
            gr.Button(visible=True, interactive=True),        # reset_btn: SHOW during processing
            gr.Button(interactive=False),                     # camera_btn
            gr.Button(interactive=False),                     # attach_btn
        )

        try:
            logger.info(f"Continuing session {self.current_session.session_id} with follow-up")

            # Build follow-up message from text input + any attached files
            parts = []
            if user_response and user_response.strip():
                parts.append(user_response.strip())

            for fp in (file_list or []):
                file_ext = fp.lower() if isinstance(fp, str) else ""
                if any(file_ext.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                    # Update image on current case so workflow can use it
                    self.current_case.image_path = fp
                    self.current_case.image_data = Image.open(fp)
                    parts.append("[New clinical image attached]")
                elif any(file_ext.endswith(ext) for ext in ['.json', '.txt', '.pdf']):
                    file_data = self.parse_case_file(fp)
                    if file_data and 'history' in file_data:
                        parts.append(file_data['history'])

            combined_message = "\n\n".join(parts) if parts else user_response or ""

            # Continue existing session with user's response
            result = asyncio.run(
                self.workflow.run_async(
                    self.current_case,
                    user_message=f"Additional information from user: {combined_message}",
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

                formatted_followup = self._format_clarification_request(response_text)
                self._save_active_session(formatted_followup, reasoning, citations)

                yield (
                    formatted_followup,
                    reasoning,
                    citations,
                    gr.Textbox(value="", interactive=True, placeholder=_PLACEHOLDER_DEFAULT),
                    gr.Button(visible=False),                     # analyze_btn: hidden
                    gr.Group(visible=True),                       # followup_group: SHOW
                    gr.Button(visible=True),                      # reset_btn
                    gr.Button(interactive=True),                  # camera_btn
                    gr.Button(interactive=True),                  # attach_btn
                )
            else:
                # Completed! Save the session now
                logger.info(f"Follow-up complete. Saving session {self.current_session.session_id}")

                # Remove non-serializable _session before saving to JSON
                from src.agents.conversation_manager import get_conversation_manager
                session_obj = result.pop("_session", None)
                if session_obj:
                    session_obj.set_final_output(result)
                    session_obj.save(get_conversation_manager().storage_dir)
                # complete_session removes from active_sessions; safe if already removed
                get_conversation_manager().complete_session(self.current_session.session_id, save=True)

                self.is_analyzing = False
                self.in_followup_mode = False
                self.current_session = None
                self._clear_active_session()

                yield (
                    self._format_soap_response(soap_note),
                    reasoning,
                    citations,
                    gr.Textbox(value="", interactive=True, placeholder=_PLACEHOLDER_DEFAULT),
                    gr.Button(visible=True, interactive=True),    # analyze_btn: SHOW
                    gr.Group(visible=False),                      # followup_group: hidden
                    gr.Button(visible=True),                      # reset_btn: SHOW after SOAP
                    gr.Button(interactive=True),                  # camera_btn
                    gr.Button(interactive=True),                  # attach_btn
                )

        except Exception as e:
            logger.error(f"Error in process_followup: {e}")
            self.is_analyzing = False
            self.in_followup_mode = False
            self.current_session = None
            self._clear_active_session()

            yield (
                f"Error processing follow-up: {str(e)}",
                "",
                "",
                gr.Textbox(value="", interactive=True, placeholder=_PLACEHOLDER_DEFAULT),
                gr.Button(visible=True, interactive=True),    # analyze_btn
                gr.Group(visible=False),                      # followup_group: hidden
                gr.Button(visible=False),                     # reset_btn
                gr.Button(interactive=True),                  # camera_btn
                gr.Button(interactive=True),                  # attach_btn
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

        # Agentic pause if response has no complete SOAP AND (has questions OR missing keywords).
        # has_complete_soap always wins -- a full SOAP is never a pause, even if keywords appear.
        is_pause = not has_complete_soap and (has_questions or has_missing_keywords)

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

        # MedGemma often returns questions on one line: "LACK INFORMATION:1.What...2.What..."
        # Insert line breaks before numbered items so they display as a list
        import re as _re
        clean_response = _re.sub(r'(\d+)\.\s*', r'\n\n\1. ', clean_response)
        # Strip "LACK INFORMATION:" prefix — already shown in the heading
        clean_response = _re.sub(r'^.*?LACK\s+INFORMATION\s*:\s*', '', clean_response, flags=_re.IGNORECASE).strip()

        # Cap at 5 most important questions to avoid overwhelming the user
        lines = clean_response.split('\n')
        question_lines = [l for l in lines if '?' in l]
        non_question_lines = [l for l in lines if '?' not in l and l.strip()]
        if len(question_lines) > 5:
            question_lines = question_lines[:5]
            # Renumber 1-5
            renumbered = []
            for i, q in enumerate(question_lines, 1):
                q_text = _re.sub(r'^\s*\d+\.\s*', '', q).strip()
                renumbered.append(f"{i}. {q_text}")
            clean_response = '\n\n'.join(renumbered)

        formatted = "## Please Provide Additional Details\n\n"
        formatted += "To give you an accurate assessment, the specialist needs a bit more information:\n\n"
        formatted += "---\n\n"
        formatted += clean_response
        formatted += "\n\n---\n\n"
        formatted += "_Please type your answers in the **Clinical Case Input** field below and click **Submit Follow-up**._\n"
        return formatted

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

    # ── SOAP formatting: v2 (section-extraction) is active, v1 (regex-strip) commented below ──

    def _format_soap_response(self, response: str) -> str:
        """v2: Extract SOAP sections directly instead of stripping noise patterns."""
        import re
        import json as _json

        # Step 1: Unwrap ADK JSON wrapper if present
        stripped = response.strip()
        if stripped.startswith('{'):
            try:
                data = _json.loads(stripped)
                for val in data.values():
                    if isinstance(val, dict):
                        soap = val.get("differential_diagnoses") or val.get("soap_note", "")
                        if soap:
                            response = soap
                        break
            except Exception:
                pass

        text = self._strip_orchestration_text(response)

        # Step 2: Try to extract SOAP sections by header
        soap_headers = [
            (r'(?:#+\s*)?(?:\*\*)?Subjective\s*(?:\(S\))?(?:\*\*)?[:\s]*', '## Subjective (S)\n\n'),
            (r'(?:#+\s*)?(?:\*\*)?Objective\s*(?:\(O\))?(?:\*\*)?[:\s]*',  '## Objective (O)\n\n'),
            (r'(?:#+\s*)?(?:\*\*)?Assessment\s*(?:\(A\))?(?:\*\*)?[:\s]*', '## Assessment (A)\n\n'),
            (r'(?:#+\s*)?(?:\*\*)?Plan\s*(?:\(P\))?(?:\*\*)?[:\s]*',      '## Plan (P)\n\n'),
        ]

        # Find positions of each SOAP header
        sections = []
        for pattern, header in soap_headers:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                sections.append((m.start(), m.end(), header))

        if len(sections) >= 3:
            # We found enough SOAP sections — extract content between headers
            sections.sort(key=lambda x: x[0])
            formatted_parts = []
            for i, (start, end, header) in enumerate(sections):
                next_start = sections[i + 1][0] if i + 1 < len(sections) else len(text)
                content = text[end:next_start].strip()
                # Strip stray bold markers MedGemma leaves at content boundaries
                content = re.sub(r'^\*\*\s*', '', content)
                content = re.sub(r'\s*\*\*\s*$', '', content)
                formatted_parts.append(header + content)
            return '\n\n'.join(formatted_parts)

        # Step 3: Fallback — strip known noise patterns (same as v1 logic)
        formatted = text
        for marker in ["Final Output:\n", "Final Output:\r\n"]:
            if formatted.startswith(marker):
                formatted = formatted[len(marker):].strip()
        formatted = re.sub(r'(?i)here\s+is\s+the\s+complete\s+soap\s+note\s+from\s+\w+.*?:\s*', '', formatted)
        formatted = re.sub(r'(?i)ensure\s+all\s+required\s+elements.*?\n', '', formatted)
        formatted = re.sub(r'(?i)\*?\*?Constraint Checklist.*?(?=\n---|\Z)', '', formatted, flags=re.DOTALL)
        formatted = re.sub(r'(?i)strategizing\s+complete\..*?\n', '', formatted)
        formatted = re.sub(r'(?m)^\*\*Patient Name:\*\*.*$', '', formatted)
        formatted = re.sub(r'(?m)^\*\*Date of Visit:\*\*.*$', '', formatted)
        formatted = re.sub(r'(?m)^\*\*Referring Physician.*:\*\*.*$', '', formatted)
        formatted = re.sub(r'(?m)^\*\*Medical Record Number.*:\*\*.*$', '', formatted)
        formatted = re.sub(r'(?m)^\*\*[\w\s]+:\*\*\s*\[[\w\s]+\]\s*$', '', formatted)
        formatted = re.sub(r'^\s*---\s*\n', '', formatted, flags=re.MULTILINE)
        formatted = re.sub(r'\n{3,}', '\n\n', formatted).strip()
        if "Subjective" in formatted and "**Subjective" not in formatted:
            formatted = formatted.replace("Subjective (S):", "## Subjective (S)")
            formatted = formatted.replace("Objective (O):", "## Objective (O)")
            formatted = formatted.replace("Assessment (A):", "## Assessment (A)")
            formatted = formatted.replace("Plan (P):", "## Plan (P)")
        return formatted

    # def _format_soap_response_v1(self, response: str) -> str:
    #     """v1 (original): Format SOAP note by stripping noise with regex patterns."""
    #     import re
    #     import json as _json
    #     stripped = response.strip()
    #     if stripped.startswith('{'):
    #         try:
    #             data = _json.loads(stripped)
    #             for val in data.values():
    #                 if isinstance(val, dict):
    #                     soap = val.get("differential_diagnoses") or val.get("soap_note", "")
    #                     if soap:
    #                         response = soap
    #                     break
    #         except Exception:
    #             pass
    #     formatted = self._strip_orchestration_text(response)
    #     for marker in ["Final Output:\n", "Final Output:\r\n"]:
    #         if formatted.startswith(marker):
    #             formatted = formatted[len(marker):].strip()
    #     formatted = re.sub(r'(?i)here\s+is\s+the\s+complete\s+soap\s+note\s+from\s+\w+.*?:\s*', '', formatted)
    #     formatted = re.sub(r'(?i)ensure\s+all\s+required\s+elements.*?\n', '', formatted)
    #     formatted = re.sub(r'(?i)\*?\*?Constraint Checklist.*?(?=\n---|\Z)', '', formatted, flags=re.DOTALL)
    #     formatted = re.sub(r'(?i)strategizing\s+complete\..*?\n', '', formatted)
    #     formatted = re.sub(r'(?m)^\*\*Patient Name:\*\*.*$', '', formatted)
    #     formatted = re.sub(r'(?m)^\*\*Date of Visit:\*\*.*$', '', formatted)
    #     formatted = re.sub(r'(?m)^\*\*Referring Physician.*:\*\*.*$', '', formatted)
    #     formatted = re.sub(r'(?m)^\*\*Medical Record Number.*:\*\*.*$', '', formatted)
    #     formatted = re.sub(r'(?m)^\*\*[\w\s]+:\*\*\s*\[[\w\s]+\]\s*$', '', formatted)
    #     formatted = re.sub(r'^\s*---\s*\n', '', formatted, flags=re.MULTILINE)
    #     formatted = re.sub(r'\n{3,}', '\n\n', formatted).strip()
    #     if "Subjective" in formatted and "**Subjective" not in formatted:
    #         formatted = formatted.replace("Subjective (S):", "## Subjective (S)")
    #         formatted = formatted.replace("Objective (O):", "## Objective (O)")
    #         formatted = formatted.replace("Assessment (A):", "## Assessment (A)")
    #         formatted = formatted.replace("Plan (P):", "## Plan (P)")
    #     return formatted

    def _extract_thinking_process(self, response: str, result: dict = None) -> str:
        """Build a readable, scrollable agent trace from actual agent_steps in the result."""
        result = result or {}
        model_name = result.get("model", "gemini-2.0-flash")
        agent_steps = result.get("agent_steps", [])

        from src.agents.adk_agents import _specialist_display_name
        trace = "### Clinical Reasoning Pipeline\n\n"
        trace += f"**Orchestrator**: {model_name} (Google ADK)  \n"
        trace += f"**Clinical AI**: MedGemma ({_specialist_display_name()})\n\n"
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

    # ── Citation extraction: v2 (ast.literal_eval fallback) active, v1 (regex fallback) commented below ──

    def _extract_citations(self, response: str, is_pause: bool = False, result: dict = None) -> str:
        """v2: Extract RAG citations — uses structured data first, ast.literal_eval fallback instead of regex."""
        import re
        import ast
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

        # Primary source: structured data stored directly by the tool
        rag_docs = []
        stored_guidelines = result.get("retrieved_guidelines", [])
        rag_query = result.get("rag_query", "")

        if rag_query:
            rag_docs.append(("Query used", rag_query))

        for g in stored_guidelines:
            src = g.get("source", "Unknown")
            title = g.get("title", "Untitled")
            score = g.get("similarity_score", 0.0)
            rag_docs.append((src, title, score))

        # Fallback: parse function call args via ast.literal_eval (not regex)
        if not stored_guidelines:
            for step in agent_steps:
                for fc in step.get("function_calls", []):
                    args_str = fc.get("args", "")
                    if not args_str:
                        continue
                    try:
                        args_dict = ast.literal_eval(args_str) if isinstance(args_str, str) else args_str
                    except (ValueError, SyntaxError):
                        args_dict = {}
                    if not isinstance(args_dict, dict):
                        continue

                    if fc.get("name") == "retrieve_clinical_guidelines" and not rag_query:
                        query = args_dict.get("query", "")
                        if query:
                            rag_docs.insert(0, ("Query used", query))

                    elif fc.get("name") == "medgemma_guideline_synthesis":
                        guidelines = args_dict.get("guidelines", [])
                        if isinstance(guidelines, list):
                            for g in guidelines:
                                if isinstance(g, dict):
                                    rag_docs.append((
                                        g.get("source", "AAD"),
                                        g.get("title", "Untitled"),
                                        float(g.get("similarity_score", 0.0)),
                                    ))

        # Also search response text for inline citations (simple, stable patterns)
        inline_sources = re.findall(r'Source:\s*([^\n\r]+)', response, re.IGNORECASE)
        aad_inline = re.findall(r'(?:AAD Guidelines?|American Academy of Dermatology)[:\-]\s*([^\n\r\.]+)', response, re.IGNORECASE)
        sp_inline = re.findall(r'StatPearls[:\-]\s*([^\n\r\.]+)', response, re.IGNORECASE)
        inline_all = list(dict.fromkeys([s.strip() for s in inline_sources + aad_inline + sp_inline if s.strip()]))

        # Format output
        if rag_docs:
            citations_md += "*Documents retrieved from RAG corpus (Vertex AI RAG):*\n\n"
            seen_titles = set()
            has_docs = False
            for item in rag_docs:
                if item[0] == "Query used":
                    citations_md += f"**Search query**: `{item[1]}`\n\n"
                    citations_md += "**Retrieved documents:**\n\n"
                elif len(item) == 3:
                    src, title, score = item
                    clean_title = re.sub(r'\.pdf$', '', str(title), flags=re.IGNORECASE).replace('_', ' ').replace('-', ' ')
                    if clean_title not in seen_titles and clean_title.strip() and clean_title.lower() not in ("untitled", "unknown"):
                        seen_titles.add(clean_title)
                        citations_md += f"- **{src}** — {clean_title} *(similarity: {score:.2f})*\n"
                        has_docs = True
                    elif clean_title not in seen_titles:
                        seen_titles.add(clean_title)
                        citations_md += f"- **{src}** — *(similarity: {score:.2f})*\n"
                        has_docs = True
            if not has_docs:
                citations_md += "*No matching guideline documents retrieved for this query.*\n"
                citations_md += "*MedGemma still generated the SOAP note — diagnosis is based on the clinical case data.*\n"
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

    # def _extract_citations_v1(self, response: str, is_pause: bool = False, result: dict = None) -> str:
    #     """v1 (original): Extract RAG citations with regex fallback for function call args."""
    #     import re
    #     result = result or {}
    #     agent_steps = result.get("agent_steps", [])
    #     citations_md = "### Evidence-Based Clinical Guidelines\n\n"
    #     if is_pause:
    #         citations_md += "*No guidelines retrieved — pipeline paused at Triage stage.*\n\n"
    #         citations_md += "RAG retrieval only runs when clinical data is sufficient for diagnosis.\n\n"
    #         citations_md += "Provide the requested information and the Research Agent will query:\n"
    #         citations_md += "- AAD Clinical Practice Guidelines\n- StatPearls Medical Reference\n- JAAD Case Reports\n"
    #         return citations_md
    #     rag_docs = []
    #     stored_guidelines = result.get("retrieved_guidelines", [])
    #     rag_query = result.get("rag_query", "")
    #     if rag_query:
    #         rag_docs.append(("Query used", rag_query))
    #     for g in stored_guidelines:
    #         rag_docs.append((g.get("source", "Unknown"), g.get("title", "Untitled"), g.get("similarity_score", 0.0)))
    #     if not stored_guidelines:
    #         for step in agent_steps:
    #             for fc in step.get("function_calls", []):
    #                 if fc.get("name") == "retrieve_clinical_guidelines" and not rag_query:
    #                     args = fc.get("args", "")
    #                     m = re.search(r"'query':\s*'([^']+)'", args)
    #                     if m:
    #                         rag_docs.insert(0, ("Query used", m.group(1)))
    #                 elif fc.get("name") == "medgemma_guideline_synthesis":
    #                     args = fc.get("args", "")
    #                     titles = re.findall(r"'title':\s*'([^']+)'", args)
    #                     sources = re.findall(r"'source':\s*'([^']+)'", args)
    #                     scores = re.findall(r"'similarity_score':\s*([\d.]+)", args)
    #                     for i, title in enumerate(titles):
    #                         src_item = sources[i] if i < len(sources) else "AAD"
    #                         score_item = float(scores[i]) if i < len(scores) else 0.0
    #                         rag_docs.append((src_item, title, score_item))
    #     inline_sources = re.findall(r'Source:\s*([^\n\r]+)', response, re.IGNORECASE)
    #     aad_inline = re.findall(r'(?:AAD Guidelines?|American Academy of Dermatology)[:\-]\s*([^\n\r\.]+)', response, re.IGNORECASE)
    #     sp_inline = re.findall(r'StatPearls[:\-]\s*([^\n\r\.]+)', response, re.IGNORECASE)
    #     inline_all = list(dict.fromkeys([s.strip() for s in inline_sources + aad_inline + sp_inline if s.strip()]))
    #     if rag_docs:
    #         citations_md += "*Documents retrieved from RAG corpus (Vertex AI RAG):*\n\n"
    #         seen_titles = set()
    #         has_docs = False
    #         for item in rag_docs:
    #             if item[0] == "Query used":
    #                 citations_md += f"**Search query**: `{item[1]}`\n\n**Retrieved documents:**\n\n"
    #             elif len(item) == 3:
    #                 src, title, score = item
    #                 clean_title = re.sub(r'\.pdf$', '', title, flags=re.IGNORECASE).replace('_', ' ').replace('-', ' ')
    #                 if clean_title not in seen_titles and clean_title.strip() and clean_title.lower() not in ("untitled", "unknown"):
    #                     seen_titles.add(clean_title)
    #                     citations_md += f"- **{src}** — {clean_title} *(similarity: {score:.2f})*\n"
    #                     has_docs = True
    #                 elif clean_title not in seen_titles:
    #                     seen_titles.add(clean_title)
    #                     citations_md += f"- **{src}** — *(similarity: {score:.2f})*\n"
    #                     has_docs = True
    #         if not has_docs:
    #             citations_md += "*No matching guideline documents retrieved for this query.*\n"
    #             citations_md += "*MedGemma still generated the SOAP note — diagnosis is based on the clinical case data.*\n"
    #         citations_md += "\n"
    #     if inline_all:
    #         citations_md += "*Cited by MedGemma in response:*\n\n"
    #         for src in inline_all:
    #             citations_md += f"- {src}\n"
    #         citations_md += "\n"
    #     if not rag_docs and not inline_all:
    #         citations_md += "*No citations extracted.*\n\n**Knowledge bases available:**\n"
    #         citations_md += "- AAD Clinical Practice Guidelines\n- StatPearls Medical Reference\n- JAAD Case Reports\n"
    #     citations_md += "\n---\n\n*RAG corpus: AAD guidelines, StatPearls, JAAD Case Reports (Vertex AI RAG, us-west1)*"
    #     return citations_md

    def create_ui(self) -> gr.Blocks:
        """Create Gradio interface matching EXACT HTML design."""

        theme = gr.themes.Base(
            primary_hue=gr.themes.colors.orange,
            neutral_hue=gr.themes.colors.neutral,
            text_size=gr.themes.sizes.text_sm,
            spacing_size=gr.themes.sizes.spacing_sm,
            radius_size=gr.themes.sizes.radius_sm,
            font=("Segoe UI",),
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

/* ===== HIDDEN SLOT TEXTBOX (for file removal) ===== */
#remove-slot { display: none !important; }

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


</style>
""")

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
                /* Zero out the group wrapper and ALL inner Gradio .block divs */
                #example-btns-group {
                    padding: 0 !important;
                    margin: 0 !important;
                    gap: 0 !important;
                    border: none !important;
                    background: transparent !important;
                }
                #example-btns-group > * {
                    padding: 0 !important;
                    margin: 0 !important;
                }
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
                    font-family: monospace !important;
                }
                #example-btn-0 button:hover, #example-btn-1 button:hover {
                    background: #f5f5f5 !important;
                    padding-left: 0 !important;
                    margin-left: 0 !important;
                    text-align: left !important;
                }
                /* Divider between example cases — CSS border instead of a separate HTML block */
                #example-btn-0 button {
                    border-bottom: 1px solid #e8e8e8 !important;
		    font-family: monospace !important;
                }
               #example-btn-0 .gr-button.gr-button {
                     padding-left: 0 !important;
                     font-family: monospace !important;
               }

               #example-btn-1 .gr-button {
                     padding-left: 0 !important;
               }
                </style>
                <div style='padding:4px;'>
                  <span style='font-size:18px; font-weight:600;'>Example Cases (Click to load):</span>
                </div>""")

                with gr.Group(elem_id="example-btns-group"):
                    example_btn_0 = gr.Button(
                        value=(
                            "Case 1 : 73M, post-vaccine rash:  "
                            + _CASE1_FULL[:215] + "..."
                        ),
                        variant="secondary",
                        elem_id="example-btn-0",
                    )
                    example_btn_1 = gr.Button(
                        value=(
                            "Case 2 : 10M, tense bullae:  "
                            + _CASE2_FULL[:217] + "..."
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

                # Hidden textbox — JS writes the slot index here to trigger removal
                # (visible=True + CSS display:none, so Gradio event binding stays active)
                remove_slot = gr.Textbox(value="", elem_id="remove-slot", visible=True)

                with gr.Row(elem_id="input-btn-row"):
                    camera_btn = gr.Button("📷 Camera", variant="secondary", scale=1, size="lg", elem_id="camera-btn")
                    attach_btn = gr.Button("📎 Attach", variant="secondary", scale=1, size="lg", elem_id="attach-btn")
                    analyze_btn = gr.Button("Analyze Case", variant="primary", scale=1, size="lg", visible=True)
                    with gr.Group(visible=False) as followup_group:
                        followup_btn = gr.Button("Submit Follow-up", variant="primary", size="lg")
                    reset_btn = gr.Button("Reset", variant="secondary", scale=0, visible=False, size="lg")

            # EVENT HANDLERS

            _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

            def format_file_display(file_list):
                """Generate HTML chips for attached files.
                - Images: tiny 28×28 square thumbnail + truncated name
                - Other files: filename chip (truncated at 80 chars)
                X button clicks the hidden Gradio remove button for that slot.
                Filenames are HTML-escaped to prevent injection.
                """
                if not file_list:
                    return ""
                html_parts = []
                for i, fp in enumerate(file_list):
                    name = os.path.basename(fp)
                    ext = os.path.splitext(name)[1].lower()
                    is_image = ext in _IMAGE_EXTS

                    # Truncate name at 80 chars, then HTML-escape
                    display_name = name if len(name) <= 80 else name[:80] + "..."
                    display_name = _html.escape(display_name)

                    thumb_html = ""
                    if is_image:
                        try:
                            import base64 as _b64
                            from io import BytesIO as _BytesIO
                            _thumb = Image.open(fp)
                            _thumb.thumbnail((56, 56))  # 2x for retina, displayed at 28x28
                            _buf = _BytesIO()
                            _thumb.save(_buf, format="JPEG", quality=60)
                            _b64_str = _b64.b64encode(_buf.getvalue()).decode("ascii")
                            thumb_html = (
                                f"<img src='data:image/jpeg;base64,{_b64_str}' "
                                f"style='width:28px;height:28px;object-fit:cover;"
                                f"border-radius:3px;margin-right:5px;vertical-align:middle;'>"
                            )
                        except Exception:
                            thumb_html = ""

                    html_parts.append(
                        f"<span style='display:inline-flex; align-items:center; "
                        f"background:#f0f0f0; border:1px solid #ddd; border-radius:4px; "
                        f"padding:2px 6px; margin:2px 4px; font-size:11px; max-width:280px;'>"
                        f"{thumb_html}"
                        f"<span style='overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                        f"max-width:180px;vertical-align:middle;'>{display_name}</span>"
                        f"<button data-remove-slot='{i}'"
                        f" style='border:none;background:none;cursor:pointer;"
                        f"margin-left:5px;font-size:14px;color:#999;line-height:1;flex-shrink:0;'>×</button>"
                        f"</span>"
                    )
                return "<div style='padding:4px 0; display:flex; flex-wrap:wrap;'>" + "".join(html_parts) + "</div>"

            # Track uploaded files from hidden gr.File
            def on_file_upload(files, current_files):
                """Add newly uploaded files to the list (max 3), with file size validation."""
                if not files:
                    return current_files, format_file_display(current_files), None
                new_files = files if isinstance(files, list) else [files]
                updated = list(current_files) if current_files else []
                for f in new_files:
                    if len(updated) >= MAX_FILES:
                        break
                    error = MedGemmaApp._validate_file(f)
                    if error:
                        logger.warning(error)
                        continue
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

            # Use Photo — save captured image with UUID name, add to file_state
            import shutil, datetime as _dt, uuid as _uuid

            def _cleanup_old_temp_images():
                """Remove temp captured images older than TEMP_IMAGE_MAX_AGE_HOURS."""
                try:
                    temp_dir = Path(TEMP_IMAGE_DIR)
                    if not temp_dir.exists():
                        return
                    import time
                    cutoff = time.time() - (TEMP_IMAGE_MAX_AGE_HOURS * 3600)
                    for f in temp_dir.glob("captured_*.jpeg"):
                        try:
                            if f.stat().st_mtime < cutoff:
                                f.unlink()
                                logger.debug(f"Cleaned up old temp image: {f.name}")
                        except OSError:
                            pass
                except Exception as e:
                    logger.debug(f"Temp image cleanup error (non-critical): {e}")

            # Clean up old temp images on app startup
            _cleanup_old_temp_images()

            def on_use_photo(img_path, current_files):
                """Save captured image, add to file list, hide camera popup."""
                if not img_path:
                    return current_files, format_file_display(current_files), gr.update(visible=False), None
                try:
                    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
                    # UUID filename prevents collisions (two captures in same second)
                    unique_id = _uuid.uuid4().hex[:8]
                    timestamp = _dt.datetime.now().strftime("%H-%M-%S")
                    dest = os.path.join(TEMP_IMAGE_DIR, f"captured_{timestamp}_{unique_id}.jpeg")
                    shutil.copy2(img_path, dest)
                    updated = list(current_files) if current_files else []
                    if len(updated) < MAX_FILES:
                        updated.append(dest)
                    # Clean up old temp images while we're at it
                    _cleanup_old_temp_images()
                    return updated, format_file_display(updated), gr.update(visible=False), None
                except Exception as e:
                    logger.error(f"Failed to save captured photo: {e}")
                    return current_files, format_file_display(current_files), gr.update(visible=False), None

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

            # Remove file handler — JS writes slot index to hidden textbox, this fires
            def on_remove_file(slot_str, current_files):
                try:
                    idx = int(slot_str)
                except (ValueError, TypeError):
                    return current_files, format_file_display(current_files), ""
                updated = list(current_files) if current_files else []
                if 0 <= idx < len(updated):
                    removed = updated.pop(idx)
                    logger.info(f"Removed file at slot {idx}: {os.path.basename(removed)}")
                return updated, format_file_display(updated), ""

            remove_slot.change(
                fn=on_remove_file,
                inputs=[remove_slot, file_state],
                outputs=[file_state, file_display, remove_slot]
            )

            # Example case buttons — load full case text into textbox
            example_btn_0.click(fn=lambda: gr.update(value=_CASE1_FULL), outputs=[text_input])
            example_btn_1.click(fn=lambda: gr.update(value=_CASE2_FULL), outputs=[text_input])

            # Single action button — dispatches based on current mode
            def on_action(text, files):
                file_list = files if files else []
                if self.in_followup_mode:
                    yield from self.process_followup(text, file_list)
                else:
                    yield from self.process_case(text, file_list)

            _outputs = [soap_output, reasoning_output, citations_output, text_input, analyze_btn, followup_group, reset_btn, camera_btn, attach_btn]

            analyze_btn.click(
                fn=on_action,
                inputs=[text_input, file_state],
                outputs=_outputs
            )

            # Follow-up button — same handler, just wired to the second button
            followup_btn.click(
                fn=on_action,
                inputs=[text_input, file_state],
                outputs=_outputs
            )

            # Reset button - clear everything, back to initial state
            def on_reset():
                """Reset all state back to initial."""
                self.current_case = None
                self.current_session = None
                self.is_analyzing = False
                self.in_followup_mode = False
                self._clear_active_session()
                return (
                    "",  # soap_output
                    "<span style='font-size:9px;'>Detailed clinical reasoning will be displayed here after case analysis.</span>",
                    "<span style='font-size:9px;'>Relevant clinical guidelines and evidence-based references will be displayed here.</span>",
                    gr.Textbox(value="", placeholder=_PLACEHOLDER_DEFAULT),  # text_input: fully reset
                    [],  # file_state
                    "",  # file_display
                    gr.Button(visible=True, interactive=True),   # analyze_btn
                    gr.Group(visible=False),                      # followup_group: hidden
                    gr.Button(visible=False),                     # reset_btn
                    gr.Button(interactive=True),                  # camera_btn
                    gr.Button(interactive=True),                  # attach_btn
                )

            reset_btn.click(
                fn=on_reset,
                outputs=[soap_output, reasoning_output, citations_output, text_input, file_state, file_display, analyze_btn, followup_group, reset_btn, camera_btn, attach_btn]
            )

            # Restore active follow-up session on page load (survives browser refresh)
            def on_page_load():
                restored = self._restore_active_session()
                if restored:
                    return restored
                return (gr.skip(),) * 9  # no-op

            app.load(
                fn=on_page_load,
                outputs=_outputs
            )

            # JS event delegation for file remove chips
            # Clicking X writes the slot index into a hidden Gradio textbox,
            # which triggers the .change() handler to remove the file.
            app.load(
                fn=None,
                js="""
                () => {
                    document.addEventListener('click', function(e) {
                        var btn = e.target.closest('[data-remove-slot]');
                        if (!btn) return;
                        // Block removal if processing is active
                        if (window.__mgProcessing) { e.preventDefault(); e.stopPropagation(); return; }
                        e.preventDefault();
                        e.stopPropagation();
                        var slot = btn.getAttribute('data-remove-slot');
                        var el = document.querySelector('#remove-slot textarea, #remove-slot input');
                        if (el) {
                            var nativeSetter = Object.getOwnPropertyDescriptor(
                                window.HTMLTextAreaElement.prototype || window.HTMLInputElement.prototype, 'value'
                            ).set || Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                            nativeSetter.call(el, slot);
                            el.dispatchEvent(new Event('input', {bubbles: true}));
                        }
                    });
                }
                """
            )

            # JS: disable file chips + show placeholder animation during processing.
            # <script> tags inside gr.HTML do NOT execute in Gradio 6 — must use app.load(js=...).
            app.load(
                fn=None,
                js="""
                () => {
                    window.__mgProcessing = false;
                    var _sawDisabled = false;
                    var WORKING_PH = "MedGemma Clinical Robustness is analyzing your case... please wait";

                    function getTA() { return document.querySelector('#case-input textarea'); }

                    function setChipsDisabled(off) {
                        var fd = document.querySelector('#file-display');
                        if (fd) { fd.style.pointerEvents = off ? 'none' : ''; fd.style.opacity = off ? '0.5' : ''; }
                    }

                    function applyPlaceholder() {
                        var ta = getTA();
                        if (ta && !ta.value.trim()) ta.placeholder = WORKING_PH;
                    }

                    document.addEventListener('click', function(e) {
                        var btn = e.target.closest('button');
                        if (!btn) return;
                        var t = btn.textContent.trim();
                        if (t.indexOf('Analyze Case') === -1 && t.indexOf('Submit Follow-up') === -1) return;
                        window.__mgProcessing = true;
                        _sawDisabled = false;
                        setChipsDisabled(true);
                        applyPlaceholder();
                    }, true);

                    setInterval(function() {
                        if (!window.__mgProcessing) return;
                        var ta = getTA();
                        if (!ta) return;
                        var isOff = ta.disabled || ta.readOnly;
                        if (isOff) {
                            _sawDisabled = true;
                            setChipsDisabled(true);
                            applyPlaceholder();
                        }
                        if (!isOff && _sawDisabled) {
                            window.__mgProcessing = false;
                            _sawDisabled = false;
                            setChipsDisabled(false);
                        }
                    }, 300);

                    setTimeout(function() {
                        if (window.__mgProcessing) {
                            window.__mgProcessing = false; _sawDisabled = false; setChipsDisabled(false);
                        }
                    }, 120000);
                }
                """
            )

        return app


def launch_app():
    """Launch the Gradio app."""
    app_instance = MedGemmaApp()
    app = app_instance.create_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    launch_app()
