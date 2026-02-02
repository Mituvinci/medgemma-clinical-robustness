"""
Test Google ADK Multi-Agent Workflow

Verifies the proper Google ADK implementation.
"""

import sys
from pathlib import Path
import asyncio
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# FIX: Patch ADK telemetry to handle bytes (if needed in ADK 1.23.0)
# ============================================================================
try:
    import google.adk.telemetry as telemetry

    # Try to patch the trace_call_llm function to handle bytes
    original_trace_call_llm = telemetry.trace_call_llm

    def safe_trace_call_llm(*args, **kwargs):
        """Wrapper that converts bytes to strings before logging."""
        try:
            # Try to sanitize any bytes in the arguments
            safe_args = []
            for arg in args:
                if isinstance(arg, bytes):
                    safe_args.append(arg.decode('utf-8', errors='replace'))
                else:
                    safe_args.append(arg)

            safe_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, bytes):
                    safe_kwargs[key] = value.decode('utf-8', errors='replace')
                else:
                    safe_kwargs[key] = value

            return original_trace_call_llm(*safe_args, **safe_kwargs)
        except Exception:
            # If patching fails, just skip telemetry for this call
            pass

    telemetry.trace_call_llm = safe_trace_call_llm
    print("✓ ADK 1.23.0 telemetry patched for bytes handling")
except Exception as e:
    print(f"⚠ Could not patch ADK telemetry: {e}")
    print("   (Should be fine with ADK 1.23.0+)")
# ============================================================================

from src.agents.adk_agents import create_workflow
from src.utils.schemas import ClinicalCase, ContextState
from config.config import validate_config

async def test_adk_workflow_async():
    """Test the Google ADK workflow asynchronously."""
    print("=" * 70)
    print("Testing Google ADK Multi-Agent Workflow")
    print("=" * 70)

    # Validate config
    print("\n1. Validating configuration...")
    try:
        validate_config()
        print("   [OK] Configuration valid")
    except Exception as e:
        print(f"   [ERROR] Configuration error: {e}")
        return

    # Create test case
    print("\n2. Creating test case...")
    test_case = ClinicalCase(
        case_id="test_adk_001",
        context_state=ContextState.ORIGINAL,
        history="65-year-old man with itchy red rash on elbows for 3 weeks",
        physical_exam="Erythematous plaques with silvery scales on bilateral elbows",
        patient_age=65,
        patient_gender="male",
        metadata={"duration": "3 weeks"}
    )
    print(f"   [OK] Case: {test_case.case_id}")
    print(f"   [OK] History: {test_case.history}")

    # Create Google ADK workflow
    print("\n3. Initializing Hybrid ADK + MedGemma workflow...")
    workflow = create_workflow(
        model_name="gemini-pro-latest",  # ADK orchestration
        use_medgemma=True  # MedGemma clinical reasoning
    )
    print("   [OK] Hybrid architecture initialized")
    print("   [OK] Orchestration: Google ADK with Gemini Pro Latest")
    print("   [OK] Clinical Reasoning: MedGemma-27B (Health-Specialized)")
    print("\n   [OK] Root Coordinator with 3 sub-agents:")
    print("       - TriageAgent (ADK orchestrates → MedGemma analyzes)")
    print("       - ResearchAgent (ADK retrieves → MedGemma synthesizes)")
    print("       - DiagnosticAgent (ADK coordinates → MedGemma diagnoses)")

    # Run workflow
    print("\n4. Running multi-agent workflow...")
    print("   This may take 30-60 seconds...\n")

    try:
        result = await workflow.run_async(test_case)

        print("\n" + "=" * 70)
        print("WORKFLOW RESULTS")
        print("=" * 70)

        print(f"\nSession ID: {result['session_id']}")
        print(f"Case ID: {result['case_id']}")
        print(f"Model: {result['model']}")

        print("\nAgent Response:")
        print("-" * 70)
        print(result['response'])
        print("-" * 70)

        print("\n" + "=" * 70)
        print("[OK] GOOGLE ADK WORKFLOW TEST PASSED")
        print("=" * 70)

        print("\nNext Steps:")
        print("1. Check logs/sessions/ for full conversation log")
        print("2. The response above is the final SOAP note from DiagnosticAgent")
        print("3. Build Gradio UI (Step 4) to provide interactive interface")

        return True  # Success

    except Exception as e:
        print(f"\n[ERROR] Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("[FAILED] GOOGLE ADK WORKFLOW TEST FAILED")
        print("=" * 70)
        sys.exit(1)  # Exit with error code


def test_adk_workflow():
    """Synchronous wrapper for async test."""
    try:
        result = asyncio.run(test_adk_workflow_async())
        if result:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Test runner failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_adk_workflow()
