"""
RAG (Retrieval-Augmented Generation) Module

Provides complete RAG pipeline for dermatology guidelines:
- Document processing (PDF, JSON, HTML, text)
- Text chunking with sliding window
- Embedding generation (sentence-transformers)
- Vector storage (ChromaDB)
- Semantic retrieval
"""

from src.rag.vector_store import VectorStore
from src.rag.document_processor import DocumentProcessor, Document
from src.rag.chunking import TextChunker, TextChunk
from src.rag.embeddings import EmbeddingGenerator
from src.rag.ingestion import IngestionPipeline, run_ingestion_from_config
from src.rag.retriever import Retriever

__all__ = [
    "VectorStore",
    "DocumentProcessor",
    "Document",
    "TextChunker",
    "TextChunk",
    "EmbeddingGenerator",
    "IngestionPipeline",
    "run_ingestion_from_config",
    "Retriever",
]
