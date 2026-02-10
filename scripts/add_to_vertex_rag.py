"""
Add New Documents to Existing Vertex AI RAG Corpus

Similar to add_new_knowledge.py but for Vertex RAG instead of ChromaDB.
Appends new documents from GCS or local files to the existing corpus.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import vertexai
from vertexai.preview import rag
from config.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_documents_to_vertex_rag(
    corpus_name: str,
    source_path: str,
    is_gcs: bool = False,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
):
    """
    Add new documents to an existing Vertex RAG corpus.

    Args:
        corpus_name: Full resource name of the RAG corpus
        source_path: Path to documents (GCS bucket path or local directory)
        is_gcs: True if source_path is a GCS path (gs://...), False for local
        chunk_size: Chunk size for splitting documents
        chunk_overlap: Overlap between chunks
    """
    logger.info("="*70)
    logger.info("ADD DOCUMENTS TO VERTEX AI RAG CORPUS")
    logger.info("="*70)
    logger.info(f"Corpus: {corpus_name}")
    logger.info(f"Source: {source_path}")
    logger.info(f"Type: {'GCS' if is_gcs else 'Local'}")
    logger.info(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    logger.info("")

    # Initialize Vertex AI
    project_id = settings.google_cloud_project
    location = settings.vertex_rag_location
    vertexai.init(project=project_id, location=location)

    try:
        if is_gcs:
            # Import from GCS
            logger.info(f"Importing from GCS: {source_path}")
            response = rag.import_files(
                corpus_name=corpus_name,
                paths=[source_path],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_embedding_requests_per_min=900,
            )
            logger.info(f"✓ Import complete!")
            logger.info(f"  Files imported: {response.imported_rag_files_count}")

        else:
            # Upload local files to GCS first, then import
            logger.error("Local file upload not yet implemented.")
            logger.error("Please upload files to GCS first, then use --gcs flag.")
            logger.error(f"Example: gsutil -m cp -r {source_path} gs://your-bucket/path/")
            return False

        # Verify
        logger.info(f"\nVerifying corpus...")
        corpus = rag.get_corpus(name=corpus_name)
        rag_files = list(rag.list_files(corpus_name=corpus_name))
        logger.info(f"✓ Corpus now has {len(rag_files)} total files")

        logger.info("\n" + "="*70)
        logger.info("DOCUMENTS ADDED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info("\nThe RAG system will automatically use the new documents.")
        logger.info("No need to restart the app or update .env")
        logger.info("="*70 + "\n")

        return True

    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Check GCS path is correct and accessible")
        logger.error("2. Verify corpus name matches .env: VERTEX_RAG_CORPUS")
        logger.error("3. Check permissions: gcloud projects get-iam-policy")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Add new documents to Vertex AI RAG corpus"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=settings.vertex_rag_corpus,
        help="Full corpus resource name (defaults to VERTEX_RAG_CORPUS from .env)",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to documents (GCS: gs://bucket/path/ or local directory)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size for document splitting (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap (default: 100)",
    )

    args = parser.parse_args()

    # Check if corpus is configured
    if not args.corpus:
        logger.error("No corpus configured!")
        logger.error("Set VERTEX_RAG_CORPUS in .env or use --corpus flag")
        return 1

    # Determine if path is GCS
    is_gcs = args.path.startswith("gs://")

    # Add documents
    success = add_documents_to_vertex_rag(
        corpus_name=args.corpus,
        source_path=args.path,
        is_gcs=is_gcs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
