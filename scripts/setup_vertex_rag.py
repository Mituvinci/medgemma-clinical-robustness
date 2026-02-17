"""
Vertex AI RAG Corpus Setup Script

One-time setup to create a RAG corpus and import documents from GCS.
After running, copy the corpus resource name into .env as VERTEX_RAG_CORPUS.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import vertexai
from vertexai.preview import rag
from google.api_core import retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = "medgemma-kaggle-487619"
LOCATION = "us-west1"  # us-central1/us-east4 restricted, us-east1 had capacity issues
GCS_BUCKET = "gs://aad_state_pearl_jaad_cr_derm_data_1"
CORPUS_DISPLAY_NAME = "medgemma-derm-guidelines"

# Chunking config (matching current ChromaDB settings)
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100


def create_rag_corpus():
    """Create a new RAG corpus with text-embedding-005."""
    logger.info(f"Creating RAG corpus: {CORPUS_DISPLAY_NAME}")
    logger.info(f"Project: {PROJECT_ID}, Location: {LOCATION}")

    try:
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        # Create corpus
        corpus = rag.create_corpus(
            display_name=CORPUS_DISPLAY_NAME,
            description="Dermatology clinical guidelines from AAD, StatPearls, and JAAD case reports",
            embedding_model_config=rag.EmbeddingModelConfig(
                publisher_model="publishers/google/models/text-embedding-005"
            ),
        )

        logger.info(f"✓ Corpus created successfully!")
        logger.info(f"  Corpus name: {corpus.name}")
        logger.info(f"  Resource ID: {corpus.name.split('/')[-1]}")

        return corpus

    except Exception as e:
        logger.error(f"Failed to create corpus: {e}")
        raise


def import_documents_from_gcs(corpus_name: str):
    """Import documents from GCS bucket into the RAG corpus."""
    folders = ["AAD", "JAADCR", "StatPearls"]

    logger.info(f"\nImporting documents from: {GCS_BUCKET}")
    logger.info(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")

    for folder in folders:
        try:
            gcs_path = f"{GCS_BUCKET}/{folder}/"
            logger.info(f"\n  Importing {folder}/...")

            # Import files from GCS folder
            response = rag.import_files(
                corpus_name=corpus_name,
                paths=[gcs_path],
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                max_embedding_requests_per_min=900,  # Rate limit
            )

            logger.info(f"  ✓ {folder}/ imported successfully")
            logger.info(f"    Imported files: {response.imported_rag_files_count}")

        except Exception as e:
            logger.error(f"  ✗ Failed to import {folder}/: {e}")
            # Continue with other folders

    logger.info("\n✓ Document import complete!")


def verify_corpus(corpus_name: str):
    """Verify corpus was created and populated."""
    try:
        logger.info(f"\nVerifying corpus: {corpus_name}")

        # Get corpus info
        corpus = rag.get_corpus(name=corpus_name)

        logger.info(f"  Display name: {corpus.display_name}")
        logger.info(f"  Description: {corpus.description}")
        logger.info(f"  Corpus name: {corpus.name}")

        # List files in corpus
        rag_files = list(rag.list_files(corpus_name=corpus_name))
        logger.info(f"  Total files: {len(rag_files)}")

        if rag_files:
            logger.info(f"\n  Sample files:")
            for i, rag_file in enumerate(rag_files[:5]):
                logger.info(f"    {i+1}. {rag_file.display_name}")

        logger.info("\n✓ Corpus verification successful!")
        return True

    except Exception as e:
        logger.error(f"Corpus verification failed: {e}")
        return False


def print_env_instructions(corpus_name: str):
    """Print instructions for updating .env file."""
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("\n1. Copy this corpus name into your .env file:")
    print(f"\n   VERTEX_RAG_CORPUS={corpus_name}")
    print("\n2. Update your .env to use Vertex RAG:")
    print("\n   RAG_BACKEND=vertex")
    print("\n3. Test the setup:")
    print("\n   python scripts/evaluate_nejim_cases.py \\")
    print("     --input NEJIM/image_challenge_input \\")
    print("     --agent-model medgemma-vertex \\")
    print("     --max-cases 1 \\")
    print("     --output logs/evaluation_test_vertex_rag")
    print("\n4. To revert to ChromaDB (if needed):")
    print("\n   RAG_BACKEND=chroma")
    print("\n" + "="*70 + "\n")


def main():
    """Main setup workflow."""
    try:
        logger.info("="*70)
        logger.info("VERTEX AI RAG CORPUS SETUP")
        logger.info("="*70)

        # Step 1: Create corpus
        corpus = create_rag_corpus()
        corpus_name = corpus.name

        # Step 2: Import documents
        import_documents_from_gcs(corpus_name)

        # Step 3: Verify setup
        verify_corpus(corpus_name)

        # Step 4: Print instructions
        print_env_instructions(corpus_name)

        logger.info("Setup completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Check credentials: gcloud auth application-default login")
        logger.error("2. Verify GCS bucket exists: gcloud storage ls gs://aad_state_pearl_jaad_cr_derm_data/")
        logger.error("3. Check project permissions: gcloud projects get-iam-policy gen-lang-client-0316668095")
        return 1


if __name__ == "__main__":
    sys.exit(main())
