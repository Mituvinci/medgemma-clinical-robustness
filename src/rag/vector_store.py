"""
ChromaDB Vector Store Wrapper

Provides a clean interface for ChromaDB operations including:
- Collection management (create, get, delete)
- Document persistence to disk
- Add/query/delete operations with metadata
"""

import chromadb
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path

from config.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB wrapper for managing dermatology guideline embeddings."""

    def __init__(
        self,
        collection_name: str = "dermatology_guidelines",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize ChromaDB client with persistence.

        Args:
            collection_name: Name of the collection to use
            persist_directory: Path to persist ChromaDB data (defaults to config)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR

        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence (for ChromaDB >= 0.4.0)
        try:
            # Try ChromaDB 1.x API first
            self.client = chromadb.PersistentClient(
                path=self.persist_directory
            )
            logger.info(f"Initialized ChromaDB 1.x at {self.persist_directory}")
        except AttributeError:
            # Fallback to ChromaDB 0.4.x API
            from chromadb.config import Settings
            self.client = chromadb.Client(Settings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False
            ))
            logger.info(f"Initialized ChromaDB 0.4.x at {self.persist_directory}")

        # Collection will be created/loaded lazily
        self._collection = None

    @property
    def collection(self):
        """Lazy load collection."""
        if self._collection is None:
            self._collection = self.get_or_create_collection()
        return self._collection

    def get_or_create_collection(self) -> chromadb.Collection:
        """
        Get existing collection or create new one.

        Returns:
            ChromaDB collection
        """
        try:
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Dermatology clinical guidelines (AAD, StatPearls)"}
            )
            logger.info(f"Loaded collection '{self.collection_name}' with {collection.count()} documents")
            return collection
        except Exception as e:
            logger.error(f"Error getting/creating collection: {e}")
            raise

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> None:
        """
        Add documents with embeddings to the collection.

        Args:
            documents: List of document text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts (title, source, section, etc.)
            ids: List of unique document IDs
        """
        try:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query collection for similar documents.

        Args:
            query_embeddings: Query embedding vectors
            n_results: Number of results to return
            where: Metadata filter (e.g., {"source": "AAD"})
            where_document: Document content filter

        Returns:
            Dict with 'ids', 'distances', 'documents', 'metadatas'
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            logger.debug(f"Query returned {len(results['ids'][0])} results")
            return results
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            raise

    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self._collection = None
            logger.info(f"Deleted collection '{self.collection_name}'")
        except ValueError as e:
            # Collection doesn't exist, which is fine
            if "does not exist" in str(e):
                logger.info(f"Collection '{self.collection_name}' does not exist, nothing to delete")
                self._collection = None
            else:
                logger.error(f"Error deleting collection: {e}")
                raise
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.

        Returns:
            Dict with count and sample metadata
        """
        try:
            count = self.collection.count()

            # Get a sample document if collection is not empty
            sample_metadata = None
            if count > 0:
                sample = self.collection.peek(limit=1)
                if sample['metadatas']:
                    sample_metadata = sample['metadatas'][0]

            return {
                "name": self.collection_name,
                "count": count,
                "persist_directory": self.persist_directory,
                "sample_metadata": sample_metadata
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            raise

    def reset_collection(self) -> None:
        """Delete and recreate collection (fresh start)."""
        logger.warning(f"Resetting collection '{self.collection_name}'")
        try:
            self.delete_collection()
        except Exception as e:
            logger.warning(f"Could not delete collection (may not exist): {e}")

        # Reset the cached collection
        self._collection = None

        # Create a fresh collection
        self._collection = self.get_or_create_collection()
        logger.info(f"Collection '{self.collection_name}' reset successfully")
