"""
Check Vertex AI RAG Corpus Status

Diagnose what's in the corpus and why retrieval returns 0 documents.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import vertexai
from vertexai.preview import rag
from config.config import settings

def main():
    print("="*70)
    print("VERTEX AI RAG CORPUS DIAGNOSTIC")
    print("="*70)

    corpus_name = settings.vertex_rag_corpus
    project_id = settings.google_cloud_project
    location = settings.vertex_rag_location

    print(f"Project: {project_id}")
    print(f"Location: {location}")
    print(f"Corpus: {corpus_name}")
    print("")

    # Initialize
    vertexai.init(project=project_id, location=location)

    # 1. Check if corpus exists
    print("Step 1: Checking if corpus exists...")
    try:
        corpus = rag.get_corpus(name=corpus_name)
        print(f"✓ Corpus exists: {corpus.display_name}")
        print(f"  Description: {corpus.description if hasattr(corpus, 'description') else 'N/A'}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return 1

    # 2. List files in corpus
    print("\nStep 2: Listing files in corpus...")
    try:
        rag_files = list(rag.list_files(corpus_name=corpus_name))
        print(f"✓ Found {len(rag_files)} files in corpus")

        if len(rag_files) == 0:
            print("\n❌ PROBLEM FOUND: Corpus is EMPTY!")
            print("   No documents imported. This is why retrieval returns 0.")
            print("\n   FIX: Import documents using:")
            print("   python scripts/add_to_vertex_rag.py --path gs://your-bucket/path/")
            return 1

        print("\nFiles in corpus:")
        for i, f in enumerate(rag_files[:10], 1):
            name = f.name if hasattr(f, 'name') else 'unknown'
            display_name = f.display_name if hasattr(f, 'display_name') else 'N/A'
            print(f"  {i}. {display_name or name}")

        if len(rag_files) > 10:
            print(f"  ... and {len(rag_files) - 10} more files")

    except Exception as e:
        print(f"✗ FAILED: {e}")
        return 1

    # 3. Test retrieval
    print("\nStep 3: Testing retrieval with sample query...")
    try:
        test_query = "skin pigmentation dermatology diagnosis"
        response = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
            text=test_query,
            rag_retrieval_config=rag.RagRetrievalConfig(top_k=3),
        )

        # Count results
        num_results = 0
        if response and hasattr(response, "contexts") and response.contexts:
            contexts = response.contexts
            if hasattr(contexts, "contexts"):
                num_results = len(contexts.contexts)
            elif isinstance(contexts, list):
                num_results = len(contexts)

        if num_results > 0:
            print(f"✓ Retrieved {num_results} documents")
            print("\nSample results:")
            contexts_list = contexts.contexts if hasattr(contexts, "contexts") else contexts
            for i, ctx in enumerate(contexts_list[:3], 1):
                text_preview = ""
                if hasattr(ctx, "text"):
                    text_preview = ctx.text[:100] + "..."
                elif hasattr(ctx, "segment") and hasattr(ctx.segment, "text"):
                    text_preview = ctx.segment.text[:100] + "..."
                print(f"  {i}. {text_preview}")

            print("\n✅ CORPUS IS WORKING! Retrieval successful.")
            print("   The 0-document issue might be query-specific.")
        else:
            print(f"❌ PROBLEM: Retrieved 0 documents even with test query")
            print(f"   Corpus has {len(rag_files)} files but retrieval returns nothing.")
            print("\n   Possible causes:")
            print("   1. Documents not properly indexed (try re-importing)")
            print("   2. Embedding/similarity threshold too high")
            print("   3. Query doesn't match document content")
            return 1

    except Exception as e:
        print(f"✗ Retrieval test failed: {e}")
        return 1

    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
