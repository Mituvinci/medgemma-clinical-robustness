"""
Vertex AI RAG Engine Retriever

Drop-in replacement for the ChromaDB Retriever.
Uses Vertex AI RAG Engine for server-side embedding and retrieval,
eliminating the need for local sentence-transformers and ChromaDB.

Requires:
  - google-cloud-aiplatform >= 1.38
  - A pre-created RAG corpus (run scripts/setup_vertex_rag.py first)
  - GOOGLE_CLOUD_PROJECT and VERTEX_RAG_CORPUS set in .env
"""

import logging
from typing import List, Optional

import vertexai
from vertexai import rag

from src.utils.schemas import RetrievedDocument

logger = logging.getLogger(__name__)


class VertexRAGRetriever:
    """Semantic search retriever using Vertex AI RAG Engine."""

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        corpus_name: str = "",
    ):
        """
        Initialize Vertex AI RAG retriever.

        Args:
            project_id: Google Cloud project ID
            location: Vertex AI region
            corpus_name: Full resource name of the RAG corpus
                e.g. projects/123/locations/us-central1/ragCorpora/456
        """
        self.project_id = project_id
        self.location = location
        self.corpus_name = corpus_name

        vertexai.init(project=project_id, location=location)
        logger.info(
            f"Initialized VertexRAGRetriever (project={project_id}, "
            f"location={location}, corpus={corpus_name})"
        )

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        source_filter: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents for a query via Vertex AI RAG Engine.

        Args:
            query: Natural language query
            n_results: Maximum number of results to return
            source_filter: Optional source filter (not used by Vertex RAG,
                kept for interface compatibility)
            min_similarity: Minimum similarity threshold (0.0-1.0)

        Returns:
            List of RetrievedDocument objects, sorted by similarity
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        if not self.corpus_name:
            logger.error("No RAG corpus configured. Run scripts/setup_vertex_rag.py first.")
            return []

        logger.info(f"Vertex RAG query: '{query[:100]}...' (n_results={n_results})")

        try:
            response = rag.retrieval_query(
                rag_resources=[
                    rag.RagResource(rag_corpus=self.corpus_name)
                ],
                text=query,
                rag_retrieval_config=rag.RagRetrievalConfig(
                    top_k=n_results
                ),
            )
        except Exception as e:
            logger.error(f"Vertex RAG retrieval failed: {e}")
            return []

        documents = self._process_response(response, min_similarity)
        logger.info(f"Retrieved {len(documents)} documents from Vertex RAG")
        return documents

    def _process_response(
        self,
        response,
        min_similarity: float,
    ) -> List[RetrievedDocument]:
        """
        Convert Vertex AI RAG response to RetrievedDocument objects.

        Args:
            response: Vertex AI retrieval_query response
            min_similarity: Minimum similarity threshold

        Returns:
            List of RetrievedDocument objects sorted by similarity
        """
        documents = []

        if not response or not hasattr(response, "contexts") or not response.contexts:
            return documents

        contexts = response.contexts
        if hasattr(contexts, "contexts"):
            context_list = contexts.contexts
        else:
            context_list = contexts if isinstance(contexts, list) else []

        for ctx in context_list:
            # Extract similarity score (Vertex returns distance; convert)
            score = 0.0
            if hasattr(ctx, "score"):
                score = float(ctx.score)
            elif hasattr(ctx, "distance"):
                score = 1.0 / (1.0 + float(ctx.distance))

            if score < min_similarity:
                continue

            # Extract text content
            text = ""
            if hasattr(ctx, "text"):
                text = ctx.text
            elif hasattr(ctx, "segment") and hasattr(ctx.segment, "text"):
                text = ctx.segment.text

            # Extract source URI for metadata
            source_uri = ""
            if hasattr(ctx, "source_uri"):
                source_uri = ctx.source_uri
            elif hasattr(ctx, "source"):
                source_uri = ctx.source

            # Derive source label and title from URI
            source_label = "Unknown"
            title = "Untitled"
            if source_uri:
                uri_lower = source_uri.lower()
                if "aad" in uri_lower:
                    source_label = "AAD"
                elif "statpearls" in uri_lower:
                    source_label = "StatPearls"
                elif "jaadcr" in uri_lower or "jaad" in uri_lower:
                    source_label = "JAADCR"

                # Use filename as title
                parts = source_uri.rstrip("/").split("/")
                if parts:
                    filename = parts[-1]
                    title = filename.replace("_", " ").replace(".txt", "").replace(".md", "")

            documents.append(
                RetrievedDocument(
                    content=text,
                    source=source_label,
                    title=title,
                    section=None,
                    similarity_score=score,
                    metadata={
                        "source": source_label,
                        "title": title,
                        "source_uri": source_uri,
                    },
                )
            )

        documents.sort(key=lambda d: d.similarity_score, reverse=True)
        return documents
