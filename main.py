"""
MedGemma Clinical Robustness Assistant - Main Entry Point

Usage:
    python main.py --mode app              # Launch Gradio UI
    python main.py --mode ingest           # Run data ingestion pipeline
    python main.py --mode evaluate         # Run evaluation on 25 test cases
    python main.py --mode test             # Run test suite
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.config import settings, validate_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def run_app():
    """Launch the Gradio UI application."""
    logger.info("Starting Gradio application...")
    try:
        from src.ui.app import launch_app
        launch_app()
    except ImportError as e:
        logger.error(f"Failed to import UI module: {e}")
        logger.info("Make sure you've completed Step 4 (Gradio UI implementation)")
        sys.exit(1)


def run_ingestion():
    """Run the data ingestion pipeline to populate ChromaDB."""
    logger.info("Starting data ingestion pipeline...")
    try:
        from src.rag.ingestion import run_ingestion_from_config

        total_chunks = run_ingestion_from_config()

        if total_chunks > 0:
            logger.info(f"Ingestion completed successfully! Total chunks: {total_chunks}")
        else:
            logger.warning("No chunks were ingested. Check your data directories.")

    except ImportError as e:
        logger.error(f"Failed to import ingestion module: {e}")
        logger.info("Make sure you've completed Step 2 (Data Ingestion)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        sys.exit(1)


def run_evaluation():
    """Run evaluation on the 25 test cases."""
    logger.info("Starting evaluation pipeline...")
    try:
        from src.evaluation.evaluator import run_evaluation
        run_evaluation()
    except ImportError as e:
        logger.error(f"Failed to import evaluation module: {e}")
        logger.info("Make sure you've completed Step 5 (Evaluation Setup)")
        sys.exit(1)


def run_tests():
    """Run the test suite."""
    logger.info("Running test suite...")
    import pytest
    exit_code = pytest.main(["-v", "tests/"])
    sys.exit(exit_code)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="MedGemma Clinical Robustness Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode app          Launch the Gradio UI
  python main.py --mode ingest       Ingest guidelines into ChromaDB
  python main.py --mode evaluate     Run evaluation on test cases
  python main.py --mode test         Run test suite
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["app", "ingest", "evaluate", "test"],
        default="app",
        help="Execution mode (default: app)"
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip API key validation (for testing)"
    )

    args = parser.parse_args()

    # Validate configuration unless skipped
    if not args.skip_validation:
        try:
            validate_config()
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            logger.info("Please check your .env file and ensure all required API keys are set")
            sys.exit(1)

    # Route to appropriate function
    mode_handlers = {
        "app": run_app,
        "ingest": run_ingestion,
        "evaluate": run_evaluation,
        "test": run_tests
    }

    handler = mode_handlers.get(args.mode)
    if handler:
        handler()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
