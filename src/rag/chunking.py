"""
Text Chunking

Splits documents into smaller chunks using sliding window approach.
Preserves sentence boundaries when possible.
Attaches metadata to each chunk for traceability.
"""

import re
from typing import List, Dict, Any
import logging

from config.config import settings

logger = logging.getLogger(__name__)


class TextChunk:
    """Represents a chunk of text with metadata."""

    def __init__(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_id: str
    ):
        self.text = text
        self.metadata = metadata
        self.chunk_id = chunk_id

    def __repr__(self):
        return f"TextChunk(id={self.chunk_id}, length={len(self.text)}, source={self.metadata.get('source')})"


class TextChunker:
    """Chunk text documents with sliding window approach."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        preserve_sentences: bool = True
    ):
        """
        Initialize text chunker.

        Args:
            chunk_size: Maximum chunk size in characters (default from config)
            chunk_overlap: Overlap between chunks in characters (default from config)
            preserve_sentences: Try to break at sentence boundaries
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.preserve_sentences = preserve_sentences

        logger.info(
            f"Initialized TextChunker with chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, preserve_sentences={self.preserve_sentences}"
        )

    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any],
        doc_id: str
    ) -> List[TextChunk]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            doc_id: Document ID for generating chunk IDs

        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []

        # If text is smaller than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            return [TextChunk(
                text=text.strip(),
                metadata=metadata.copy(),
                chunk_id=f"{doc_id}_chunk_0"
            )]

        chunks = []

        if self.preserve_sentences:
            chunks = self._chunk_by_sentences(text, metadata, doc_id)
        else:
            chunks = self._chunk_by_characters(text, metadata, doc_id)

        logger.debug(f"Created {len(chunks)} chunks from document {doc_id}")
        return chunks

    def _chunk_by_sentences(
        self,
        text: str,
        metadata: Dict[str, Any],
        doc_id: str
    ) -> List[TextChunk]:
        """
        Chunk text while preserving sentence boundaries.

        Args:
            text: Text to chunk
            metadata: Metadata dict
            doc_id: Document ID

        Returns:
            List of TextChunk objects
        """
        # Split into sentences (simple regex)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""
        chunk_index = 0

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_index

                chunks.append(TextChunk(
                    text=current_chunk.strip(),
                    metadata=chunk_metadata,
                    chunk_id=f"{doc_id}_chunk_{chunk_index}"
                ))

                # Start new chunk with overlap
                # Get last N characters for overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_chunk = overlap_text + " " + sentence
                chunk_index += 1
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        # Add final chunk if any text remains
        if current_chunk.strip():
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = chunk_index

            chunks.append(TextChunk(
                text=current_chunk.strip(),
                metadata=chunk_metadata,
                chunk_id=f"{doc_id}_chunk_{chunk_index}"
            ))

        return chunks

    def _chunk_by_characters(
        self,
        text: str,
        metadata: Dict[str, Any],
        doc_id: str
    ) -> List[TextChunk]:
        """
        Chunk text by character count (simple sliding window).

        Args:
            text: Text to chunk
            metadata: Metadata dict
            doc_id: Document ID

        Returns:
            List of TextChunk objects
        """
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            # Extract chunk
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = chunk_index

                chunks.append(TextChunk(
                    text=chunk_text,
                    metadata=chunk_metadata,
                    chunk_id=f"{doc_id}_chunk_{chunk_index}"
                ))

            # Move to next position with overlap
            start += (self.chunk_size - self.chunk_overlap)
            chunk_index += 1

        return chunks

    def chunk_documents(
        self,
        documents: List[Any],  # List of Document objects
    ) -> List[TextChunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of Document objects (from document_processor)

        Returns:
            List of TextChunk objects
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_text(
                text=doc.text,
                metadata=doc.metadata,
                doc_id=doc.doc_id
            )
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} total chunks from {len(documents)} documents")
        return all_chunks
