from src.rag.retriever import Retriever

r = Retriever()
results = r.retrieve('melanoma dark lesion irregular borders', n_results=3)

for doc in results:
    print(f'Similarity: {doc.similarity_score:.2f}')
    print(f'Source: {doc.source}')
    if doc.title:
        print(f'Title: {doc.title}')
    if doc.section:
        print(f'Section: {doc.section}')
    print(f'Text: {doc.content[:300]}...')
    print()
