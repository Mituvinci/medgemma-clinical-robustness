"""
List ALL files in Vertex AI RAG Corpus
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import vertexai
from vertexai.preview import rag
from config.config import settings

def main():
    corpus_name = settings.vertex_rag_corpus
    project_id = settings.google_cloud_project
    location = settings.vertex_rag_location

    vertexai.init(project=project_id, location=location)

    print("="*70)
    print("ALL FILES IN VERTEX RAG CORPUS")
    print("="*70)
    print(f"Corpus: {corpus_name}\n")

    rag_files = list(rag.list_files(corpus_name=corpus_name))
    print(f"Total files: {len(rag_files)}\n")

    # Count by source
    aad_count = 0
    statpearls_count = 0
    jaadcr_count = 0
    other_count = 0

    for i, f in enumerate(rag_files, 1):
        display_name = f.display_name if hasattr(f, 'display_name') else 'N/A'
        name_lower = display_name.lower()

        if 'aad' in name_lower and 'jaad' not in name_lower:
            source = "[AAD]"
            aad_count += 1
        elif 'statpearl' in name_lower:
            source = "[StatPearls]"
            statpearls_count += 1
        elif 'jaad' in name_lower:
            source = "[JAADCR]"
            jaadcr_count += 1
        else:
            source = "[OTHER]"
            other_count += 1

        print(f"{i:2}. {source:15} {display_name}")

    print("\n" + "="*70)
    print("SUMMARY BY SOURCE:")
    print("="*70)
    print(f"AAD Guidelines:     {aad_count} files")
    print(f"StatPearls:         {statpearls_count} files")
    print(f"JAADCR Case Reports: {jaadcr_count} files")
    print(f"Other:              {other_count} files")
    print(f"TOTAL:              {len(rag_files)} files")
    print("="*70)

if __name__ == "__main__":
    sys.exit(main())
