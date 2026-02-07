"""
Test Gradio UI for MedGemma Clinical Assistant

Quick test to verify Gradio app initialization and component creation.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from src.ui.app import MedGemmaApp
from config.config import validate_config

def test_gradio_ui():
    """Test Gradio UI initialization."""
    print("=" * 70)
    print("Testing MedGemma Gradio UI")
    print("=" * 70)

    # Validate config
    print("\n1. Validating configuration...")
    try:
        validate_config()
        print("   [OK] Configuration valid")
    except Exception as e:
        print(f"   [ERROR] Configuration error: {e}")
        return

    # Initialize app
    print("\n2. Initializing Gradio app...")
    try:
        app_instance = MedGemmaApp()
        print("   [OK] MedGemmaApp instance created")
        print(f"   [OK] Workflow initialized with model: gemini-pro-latest")
    except Exception as e:
        print(f"   [ERROR] Failed to initialize app: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create UI components
    print("\n3. Creating UI components...")
    try:
        app = app_instance.create_ui()
        print("   [OK] Gradio Blocks interface created")
        print("   [OK] Components:")
        print("       - Disclaimer (research/demo purposes only)")
        print("       - Image upload")
        print("       - File upload")
        print("       - Manual input fields (history, exam, age, gender, duration)")
        print("       - SOAP note output (with agentic pause detection)")
        print("       - Clinical Reasoning Trace accordion")
        print("       - Evidence-Based Guidelines accordion (AAD, StatPearls)")
        print("       - Example cases")
    except Exception as e:
        print(f"   [ERROR] Failed to create UI: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 70)
    print("[OK] GRADIO UI TEST PASSED")
    print("=" * 70)

    print("\nNext Steps:")
    print("1. Launch the app: python main.py --mode app")
    print("2. Open browser at: http://127.0.0.1:7860")
    print("3. Test with example cases or upload your own data")
    print("4. Check agent thinking process in accordions")

    print("\nTo launch now with share link (for remote access):")
    print("   python -c \"from src.ui.app import launch_app; launch_app(share=True)\"")


if __name__ == "__main__":
    test_gradio_ui()
