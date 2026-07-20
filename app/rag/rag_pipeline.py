from app.rag.vector_store import VectorStore
from app.rag.prompt import build_prompt
from app.rag.query_normalizer import normalize_query


def get_rag_prompt(question):

    # Step 1: Normalize query — extract intent + metadata filter + canonical search query
    normalized = normalize_query(question)
    search_query = normalized["search_query"]
    where_filter = normalized["where"]
    intent = normalized["intent"]

    print(f"\n[RAG] Intent: {intent}")
    print(f"[RAG] Search Query: {search_query}")
    print(f"[RAG] Metadata Filter: {where_filter}")

    # Step 2: Retrieve relevant chunks using normalized query + optional where filter
    vector_store = VectorStore()
    results = vector_store.retrieve(search_query, where=where_filter)

    documents = results["documents"][0]

    # Debug: show retrieved chunks
    for i, doc in enumerate(documents, start=1):
        print(f"\n========== Chunk {i} ==========")
        print(doc[:300])  # Print first 300 chars only to avoid log spam

    context = "\n\n".join(documents)

    print("\n========== CONTEXT ==========")
    print(context[:800])

    # Step 3: Build an intent-aware prompt
    prompt = build_prompt(context, question, intent=intent, semester=normalized["semester"])

    print("\n========== FINAL PROMPT ==========")
    print(prompt)

    return prompt