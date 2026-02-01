"""
Retriever

Semantic search interface for querying ChromaDB.
Converts natural language queries to embeddings and retrieves relevant guideline chunks.
"""

from typing import List, Dict, Optional, Any
import logging

from src.rag.vector_store import VectorStore
from src.rag.embeddings import EmbeddingGenerator
from src.utils.schemas import RetrievedDocument

logger = logging.getLogger(__name__)


class Retriever:
    """Semantic search retriever for clinical guidelines."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedder: Optional[EmbeddingGenerator] = None
    ):
        """
        Initialize retriever.

        Args:
            vector_store: VectorStore instance (creates new if None)
            embedder: EmbeddingGenerator instance (creates new if None)
        """
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or EmbeddingGenerator()

        logger.info("Initialized Retriever")

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        source_filter: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Natural language query
            n_results: Maximum number of results to return
            source_filter: Optional source filter ("AAD" or "StatPearls")
            min_similarity: Minimum similarity threshold (0.0-1.0)

        Returns:
            List of RetrievedDocument objects, sorted by similarity
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        logger.info(f"Retrieving documents for query: '{query[:100]}...'")

        # Generate query embedding
        query_embedding = self.embedder.encode_query(query)

        # Prepare metadata filter
        where_filter = None
        if source_filter:
            where_filter = {"source": source_filter}

        # Query vector store
        try:
            results = self.vector_store.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter
            )

            # Convert to RetrievedDocument objects
            documents = self._process_results(results, min_similarity)

            logger.info(f"Retrieved {len(documents)} documents (before similarity filter: {len(results['ids'][0])})")
            return documents

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

    def _process_results(
        self,
        results: Dict[str, Any],
        min_similarity: float
    ) -> List[RetrievedDocument]:
        """
        Process ChromaDB results into RetrievedDocument objects.

        Args:
            results: ChromaDB query results
            min_similarity: Minimum similarity threshold

        Returns:
            List of RetrievedDocument objects
        """
        documents = []

        # ChromaDB returns results in nested lists
        ids = results['ids'][0]
        distances = results['distances'][0]
        texts = results['documents'][0]
        metadatas = results['metadatas'][0]

        for doc_id, distance, text, metadata in zip(ids, distances, texts, metadatas):
            # Convert distance to similarity score (lower distance = higher similarity)
            # ChromaDB uses L2 distance, convert to cosine similarity approximation
            similarity = 1.0 / (1.0 + distance)

            # Apply similarity threshold
            if similarity < min_similarity:
                continue

            documents.append(RetrievedDocument(
                content=text,
                source=metadata.get("source", "unknown"),
                title=metadata.get("title"),
                section=metadata.get("section"),
                similarity_score=similarity,
                metadata=metadata
            ))

        # Sort by similarity (descending)
        documents.sort(key=lambda x: x.similarity_score, reverse=True)

        return documents

    def retrieve_by_metadata(
        self,
        metadata_filter: Dict[str, str],
        n_results: int = 10
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents by metadata filter (no query embedding).

        Args:
            metadata_filter: Metadata filter dict (e.g., {"source": "AAD", "title": "Melanoma"})
            n_results: Maximum number of results

        Returns:
            List of RetrievedDocument objects
        """
        logger.info(f"Retrieving documents with metadata filter: {metadata_filter}")

        try:
            # Use a dummy query embedding (all zeros) to get documents by metadata only
            dummy_embedding = [0.0] * self.embedder.get_embedding_dimension()

            results = self.vector_store.query(
                query_embeddings=[dummy_embedding],
                n_results=n_results,
                where=metadata_filter
            )

            # Process without similarity threshold
            documents = self._process_results(results, min_similarity=0.0)

            logger.info(f"Retrieved {len(documents)} documents by metadata")
            return documents

        except Exception as e:
            logger.error(f"Error during metadata retrieval: {e}")
            return []

    def format_results_for_prompt(
        self,
        documents: List[RetrievedDocument],
        include_metadata: bool = True,
        max_context_length: int = 4000
    ) -> str:
        """
        Format retrieved documents for inclusion in LLM prompt.

        Args:
            documents: List of RetrievedDocument objects
            include_metadata: Whether to include metadata in output
            max_context_length: Maximum total character length

        Returns:
            Formatted string with retrieved documents
        """
        if not documents:
            return "No relevant guidelines found."

        formatted_docs = []
        current_length = 0

        for idx, doc in enumerate(documents, start=1):
            # Format metadata
            metadata_str = ""
            if include_metadata:
                source = doc.metadata.get("source", "Unknown")
                title = doc.metadata.get("title", "Untitled")
                section = doc.metadata.get("section", "")
                metadata_str = f"[Source: {source} | {title}"
                if section:
                    metadata_str += f" - {section}"
                metadata_str += f" | Relevance: {doc.similarity:.2f}]\n"

            # Format document
            doc_text = f"\n--- Guideline {idx} ---\n"
            if metadata_str:
                doc_text += metadata_str
            doc_text += f"{doc.text}\n"

            # Check length limit
            if current_length + len(doc_text) > max_context_length:
                logger.debug(f"Reached max context length at document {idx}")
                break

            formatted_docs.append(doc_text)
            current_length += len(doc_text)

        return "\n".join(formatted_docs)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics.

        Returns:
            Dict with collection and embedder stats
        """
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "embedder": {
                "model": self.embedder.model_name,
                "dimension": self.embedder.get_embedding_dimension(),
                "device": self.embedder.device
            }
        }
