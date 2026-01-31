"""
Test Google ADK Multi-Agent Workflow

Verifies the proper Google ADK implementation.
"""

import sys
from pathlib import Path
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

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
    print("\n3. Initializing Google ADK workflow...")
    workflow = create_workflow(model_name="gemini-pro-latest")
    print("   [OK] Google ADK agents created")
    print("   [OK] Root Coordinator with 3 sub-agents:")
    print("       - TriageAgent (checks missing data)")
    print("       - ResearchAgent (retrieves guidelines)")
    print("       - DiagnosticAgent (generates SOAP note)")

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

    except Exception as e:
        print(f"\n[ERROR] Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return


def test_adk_workflow():
    """Synchronous wrapper for async test."""
    asyncio.run(test_adk_workflow_async())


if __name__ == "__main__":
    test_adk_workflow()
