from sentence_transformers.util import semantic_search

def retrieve_context(query, contexts, embeddings, model, top_k=1):
    """Retrieve the most relevant context for a given query."""
    query_embedding = model.encode(query, convert_to_tensor=True)
    results = semantic_search(query_embedding, embeddings, top_k=top_k)
    top_contexts = [contexts[hit["corpus_id"]] for hit in results[0]]
    return top_contexts