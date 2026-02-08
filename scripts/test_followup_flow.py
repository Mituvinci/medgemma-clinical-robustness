"""
Integration Test: Full Follow-up Flow

This script tests the complete agentic conversation:
1. Submit incomplete case (image_only)
2. Agent pauses and asks for missing data
3. Automatically provide history
4. If agent still pauses, provide exam
5. Get final SOAP note
6. Verify session saved correctly

Usage:
    python scripts/test_followup_flow.py
    python scripts/test_followup_flow.py --case-id 01_02_25
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
from typing import Dict, Any
import re

from src.agents.adk_agents import create_workflow
from src.utils.schemas import ClinicalCase
from src.agents.conversation_manager import get_conversation_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FollowupFlowTester:
    """Tests automatic follow-up flow with NEJIM cases."""

    def __init__(self, nejim_dir: str = "NEJIM/image_challenge_input"):
        """Initialize tester."""
        self.workflow = create_workflow(
            model_name="gemini-pro-latest",
            use_medgemma=True
        )
        self.nejim_dir = Path(nejim_dir)
        self.conversation_manager = get_conversation_manager()

    def _detect_pause(self, response: str) -> bool:
        """Detect if agent paused."""
        response_lower = response.lower()
        has_questions = "?" in response
        has_soap = "subjective" in response_lower and "assessment" in response_lower
        missing_keywords = [
            "missing", "insufficient", "clarification",
            "please provide", "could you", "need more"
        ]
        has_missing = any(kw in response_lower[:500] for kw in missing_keywords)
        return (has_questions and not has_soap) or has_missing

    async def test_case_with_followup(self, case_id: str) -> Dict[str, Any]:
        """
        Test full follow-up flow for a case.

        Flow:
        1. Submit image_only → expect pause
        2. Provide history → check if paused or complete
        3. If paused, provide exam → should complete
        4. Verify final SOAP note
        5. Verify single session file

        Args:
            case_id: Case ID like "01_02_25"

        Returns:
            Test result dict
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing follow-up flow for case: {case_id}")
        logger.info(f"{'='*60}\n")

        # Load case files
        image_only_file = self.nejim_dir / f"{case_id}_image_only.txt"
        history_file = self.nejim_dir / f"{case_id}_history.txt"
        exam_file = self.nejim_dir / f"{case_id}_exam.txt"
        original_file = self.nejim_dir / f"{case_id}_original.txt"

        if not image_only_file.exists():
            logger.error(f"Case {case_id} not found in {self.nejim_dir}")
            return {"error": "Case not found"}

        image_only_text = image_only_file.read_text().strip()
        history_text = history_file.read_text().strip() if history_file.exists() else None
        exam_text = exam_file.read_text().strip() if exam_file.exists() else None

        # Step 1: Submit image_only
        logger.info("STEP 1: Submitting image_only case...")
        clinical_case = ClinicalCase(
            case_id=f"TEST_{case_id}_followup",
            history=image_only_text,
            physical_exam=None,
            image_data=None,
            context_state="image_only"
        )

        result1 = await self.workflow.run_async(clinical_case)
        response1 = result1.get("response", "")
        session1 = result1.get("_session")

        logger.info(f"Response length: {len(response1)} chars")

        # Check if paused
        paused_step1 = self._detect_pause(response1)

        if paused_step1:
            logger.info("✓ Agent PAUSED as expected (missing data)")
            logger.info(f"Agent's request:\n{response1[:300]}...")

            # Step 2: Provide history
            if history_text:
                logger.info("\nSTEP 2: Providing history as follow-up...")

                result2 = await self.workflow.run_async(
                    clinical_case,
                    user_message=f"Additional information: {history_text}",
                    existing_session=session1
                )

                response2 = result2.get("response", "")
                session2 = result2.get("_session")

                logger.info(f"Response length: {len(response2)} chars")

                # Check if still paused
                paused_step2 = self._detect_pause(response2)

                if paused_step2:
                    logger.info("Agent STILL PAUSED (needs more data)")
                    logger.info(f"Agent's request:\n{response2[:300]}...")

                    # Step 3: Provide exam
                    if exam_text:
                        logger.info("\nSTEP 3: Providing exam findings as follow-up...")

                        result3 = await self.workflow.run_async(
                            clinical_case,
                            user_message=f"Physical examination: {exam_text}",
                            existing_session=session2
                        )

                        response3 = result3.get("response", "")
                        session3 = result3.get("_session")

                        logger.info(f"Response length: {len(response3)} chars")

                        paused_step3 = self._detect_pause(response3)

                        if not paused_step3:
                            logger.info("✓ Agent provided FINAL DIAGNOSIS after exam")

                            # Save session
                            session3.set_final_output(result3)
                            self.conversation_manager.complete_session(session3.session_id, save=True)

                            final_response = response3
                            final_session = session3
                            steps = 3
                        else:
                            logger.warning("⚠ Agent STILL paused after exam (unexpected)")
                            final_response = response3
                            final_session = session3
                            steps = 3
                    else:
                        logger.warning("No exam.txt file available")
                        final_response = response2
                        final_session = session2
                        steps = 2
                else:
                    logger.info("✓ Agent provided DIAGNOSIS after history")

                    # Save session
                    session2.set_final_output(result2)
                    self.conversation_manager.complete_session(session2.session_id, save=True)

                    final_response = response2
                    final_session = session2
                    steps = 2
            else:
                logger.warning("No history.txt file available")
                final_response = response1
                final_session = session1
                steps = 1
        else:
            logger.warning("⚠ Agent DID NOT PAUSE on image_only (unexpected)")
            logger.info("Agent provided diagnosis with minimal data")

            final_response = response1
            final_session = session1
            steps = 1

        # Verify session saved
        session_id = final_session.session_id
        logger.info(f"\nFinal session ID: {session_id}")

        # Check for session files
        session_files = list(Path("logs/sessions").glob(f"*{session_id}*"))
        logger.info(f"Session files found: {len(session_files)}")
        for f in session_files:
            logger.info(f"  - {f.name}")

        # Check if SOAP note present
        has_soap = "subjective" in final_response.lower() and "assessment" in final_response.lower()

        result_summary = {
            "case_id": case_id,
            "steps": steps,
            "paused_initially": paused_step1,
            "final_has_soap": has_soap,
            "session_id": session_id,
            "session_files_count": len(session_files),
            "final_response_length": len(final_response),
            "success": has_soap and len(session_files) >= 1
        }

        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY:")
        logger.info(f"  Steps taken: {steps}")
        logger.info(f"  Initial pause: {'YES' if paused_step1 else 'NO'}")
        logger.info(f"  Final SOAP note: {'YES' if has_soap else 'NO'}")
        logger.info(f"  Session saved: {'YES' if len(session_files) >= 1 else 'NO'}")
        logger.info(f"  Overall: {'✓ PASS' if result_summary['success'] else '✗ FAIL'}")
        logger.info(f"{'='*60}\n")

        return result_summary


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test follow-up flow")
    parser.add_argument(
        "--case-id",
        default="01_02_25",
        help="Case ID to test (e.g., 01_02_25)"
    )
    parser.add_argument(
        "--input",
        default="NEJIM/image_challenge_input",
        help="Path to NEJIM folder"
    )

    args = parser.parse_args()

    tester = FollowupFlowTester(nejim_dir=args.input)
    result = await tester.test_case_with_followup(args.case_id)

    if result.get("success"):
        print("\n✓ Follow-up flow test PASSED!")
        print(f"Session saved to logs/sessions/ with ID: {result['session_id']}")
    else:
        print("\n✗ Follow-up flow test FAILED")
        print(f"Check logs for details")


if __name__ == "__main__":
    asyncio.run(main())
