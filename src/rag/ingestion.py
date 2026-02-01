"""
Data Ingestion Pipeline

Orchestrates the complete ingestion workflow:
1. Load documents from AAD/ and StatPearls/ directories
2. Chunk documents with sliding window
3. Generate embeddings
4. Store in ChromaDB

Provides progress tracking and error handling.
"""

from pathlib import Path
from typing import List, Optional
import logging
from tqdm import tqdm

from src.rag.document_processor import DocumentProcessor
from src.rag.chunking import TextChunker
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStore
from config.config import settings

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Complete document ingestion pipeline."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        doc_processor: Optional[DocumentProcessor] = None,
        chunker: Optional[TextChunker] = None,
        embedder: Optional[EmbeddingGenerator] = None
    ):
        """
        Initialize ingestion pipeline.

        Args:
            vector_store: VectorStore instance (creates new if None)
            doc_processor: DocumentProcessor instance
            chunker: TextChunker instance
            embedder: EmbeddingGenerator instance
        """
        self.vector_store = vector_store or VectorStore()
        self.doc_processor = doc_processor or DocumentProcessor()
        self.chunker = chunker or TextChunker()
        self.embedder = embedder or EmbeddingGenerator()

        logger.info("Initialized IngestionPipeline")

    def ingest_directory(
        self,
        directory: Path,
        recursive: bool = True,
        reset_collection: bool = False
    ) -> int:
        """
        Ingest all documents from a directory.

        Args:
            directory: Path to directory containing documents
            recursive: Process subdirectories
            reset_collection: If True, reset collection before ingestion

        Returns:
            Number of chunks ingested
        """
        directory = Path(directory)

        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return 0

        logger.info(f"Starting ingestion from {directory}")

        # Reset collection if requested
        if reset_collection:
            logger.warning("Resetting vector store collection")
            self.vector_store.reset_collection()

        # Step 1: Load documents
        logger.info("Step 1/4: Loading documents...")
        documents = self.doc_processor.process_directory(
            directory=directory,
            recursive=recursive
        )

        if not documents:
            logger.warning(f"No documents found in {directory}")
            return 0

        logger.info(f"Loaded {len(documents)} documents")

        # Step 2: Chunk documents
        logger.info("Step 2/4: Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)

        if not chunks:
            logger.warning("No chunks created")
            return 0

        logger.info(f"Created {len(chunks)} chunks")

        # Step 3: Generate embeddings
        logger.info("Step 3/4: Generating embeddings...")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.encode(texts, show_progress=True)

        logger.info(f"Generated {len(embeddings)} embeddings")

        # Step 4: Store in vector database
        logger.info("Step 4/4: Storing in ChromaDB...")
        self._store_chunks(chunks, embeddings)

        logger.info(f"Successfully ingested {len(chunks)} chunks from {directory}")
        return len(chunks)

    def ingest_multiple_directories(
        self,
        directories: List[Path],
        reset_collection: bool = False
    ) -> int:
        """
        Ingest documents from multiple directories.

        Args:
            directories: List of directory paths
            reset_collection: If True, reset collection before first ingestion

        Returns:
            Total number of chunks ingested
        """
        total_chunks = 0

        for idx, directory in enumerate(directories):
            # Only reset on first directory
            should_reset = reset_collection and idx == 0

            chunks_count = self.ingest_directory(
                directory=directory,
                reset_collection=should_reset
            )
            total_chunks += chunks_count

        logger.info(f"Total chunks ingested: {total_chunks}")
        return total_chunks

    def _store_chunks(self, chunks: List, embeddings: List[List[float]]) -> None:
        """
        Store chunks and embeddings in vector database.

        Args:
            chunks: List of TextChunk objects
            embeddings: List of embedding vectors
        """
        # Prepare data for ChromaDB
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        chunk_texts = [chunk.text for chunk in chunks]
        chunk_metadatas = [chunk.metadata for chunk in chunks]

        # Store in batches to avoid memory issues
        batch_size = 100
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        with tqdm(total=len(chunks), desc="Storing chunks") as pbar:
            for i in range(0, len(chunks), batch_size):
                batch_end = min(i + batch_size, len(chunks))

                self.vector_store.add_documents(
                    documents=chunk_texts[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    metadatas=chunk_metadatas[i:batch_end],
                    ids=chunk_ids[i:batch_end]
                )

                pbar.update(batch_end - i)

        logger.info(f"Stored {len(chunks)} chunks in vector database")

    def get_stats(self) -> dict:
        """
        Get ingestion pipeline statistics.

        Returns:
            Dict with collection stats and model info
        """
        collection_stats = self.vector_store.get_collection_stats()

        return {
            "collection": collection_stats,
            "embedder": {
                "model": self.embedder.model_name,
                "dimension": self.embedder.get_embedding_dimension(),
                "device": self.embedder.device
            },
            "chunker": {
                "chunk_size": self.chunker.chunk_size,
                "overlap": self.chunker.chunk_overlap,
                "preserve_sentences": self.chunker.preserve_sentences
            }
        }


def run_ingestion_from_config() -> int:
    """
    Run ingestion using paths from config.

    Returns:
        Number of chunks ingested
    """
    logger.info("Running ingestion from configuration")

    pipeline = IngestionPipeline()

    # Get directories from SourceKnowledgeBase
    project_root = Path(settings.BASE_DIR)
    aad_dir = project_root / "SourceKnowledgeBase" / "AAD"
    statpearls_dir = project_root / "SourceKnowledgeBase" / "StatPearls"

    directories = []

    if aad_dir.exists():
        directories.append(aad_dir)
        logger.info(f"Found AAD directory: {aad_dir}")
    else:
        logger.warning(f"AAD directory not found: {aad_dir}")

    if statpearls_dir.exists():
        directories.append(statpearls_dir)
        logger.info(f"Found StatPearls directory: {statpearls_dir}")
    else:
        logger.warning(f"StatPearls directory not found: {statpearls_dir}")

    if not directories:
        logger.error("No source directories found!")
        return 0

    # Run ingestion (reset collection to start fresh)
    total_chunks = pipeline.ingest_multiple_directories(
        directories=directories,
        reset_collection=True
    )

    # Print stats
    stats = pipeline.get_stats()
    logger.info(f"Ingestion complete! Stats: {stats}")

    return total_chunks
