#!/usr/bin/env python3
"""
Add New Knowledge Base Documents to ChromaDB

Usage:
    python add_new_knowledge.py --path /path/to/new/documents
    python add_new_knowledge.py --path SourceKnowledgeBase/NewGuidelines
    python add_new_knowledge.py --file SourceKnowledgeBase/AAD/new_guideline.pdf

This script APPENDS new documents to the existing ChromaDB collection
without deleting existing data.
"""

import argparse
import sys
from pathlib import Path
from src.rag.ingestion import IngestionPipeline
from src.rag.vector_store import VectorStore

def add_new_documents(path: str, is_file: bool = False):
    """
    Add new documents to existing ChromaDB collection.

    Args:
        path: Path to directory or file
        is_file: If True, treat path as single file
    """
    path = Path(path)

    if not path.exists():
        print(f"❌ Error: Path does not exist: {path}")
        sys.exit(1)

    # Get current stats BEFORE adding
    print("=" * 80)
    print("CURRENT CHROMADB STATUS")
    print("=" * 80)

    vector_store = VectorStore()
    stats_before = vector_store.get_collection_stats()
    print(f"Current chunks in database: {stats_before['count']}")
    print()

    # Initialize pipeline
    print("=" * 80)
    print("ADDING NEW DOCUMENTS")
    print("=" * 80)

    pipeline = IngestionPipeline(vector_store=vector_store)

    if is_file:
        # Process single file
        print(f"Processing file: {path}")

        from src.rag.document_processor import DocumentProcessor
        from src.rag.chunking import TextChunker
        from src.rag.embeddings import EmbeddingGenerator

        processor = DocumentProcessor()
        chunker = TextChunker()
        embedder = EmbeddingGenerator()

        # Step 1: Process file
        documents = processor.process_file(path)
        if not documents:
            print(f"❌ No documents extracted from {path}")
            sys.exit(1)
        print(f"✓ Extracted {len(documents)} document(s)")

        # Step 2: Chunk
        chunks = chunker.chunk_documents(documents)
        print(f"✓ Created {len(chunks)} chunks")

        # Step 3: Generate embeddings
        print("Generating embeddings...")
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.encode(texts, show_progress=True)
        print(f"✓ Generated {len(embeddings)} embeddings")

        # Step 4: Add to ChromaDB (APPEND, don't reset)
        print("Adding to ChromaDB...")
        pipeline._store_chunks(chunks, embeddings)

        chunks_added = len(chunks)
    else:
        # Process directory
        print(f"Processing directory: {path}")

        # IMPORTANT: reset_collection=False means APPEND, not replace
        chunks_added = pipeline.ingest_directory(
            directory=path,
            reset_collection=False  # ← This is the key! Don't delete existing data
        )

    # Get stats AFTER adding
    print()
    print("=" * 80)
    print("UPDATE COMPLETE")
    print("=" * 80)

    stats_after = vector_store.get_collection_stats()
    print(f"Chunks before: {stats_before['count']}")
    print(f"Chunks added:  {chunks_added}")
    print(f"Total chunks:  {stats_after['count']}")
    print()

    if stats_after['count'] == stats_before['count'] + chunks_added:
        print("✅ Success! New documents added to ChromaDB")
    else:
        print("⚠️  Warning: Chunk count mismatch. Some chunks may have been skipped.")

    print()
    print("Your existing knowledge base is intact!")
    print("New documents are now searchable alongside old ones.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add new knowledge base documents to ChromaDB"
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to directory or file to add"
    )
    parser.add_argument(
        "--file",
        action="store_true",
        help="Treat path as single file (default: directory)"
    )

    args = parser.parse_args()

    add_new_documents(
        path=args.path,
        is_file=args.file
    )
