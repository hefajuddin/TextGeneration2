from utils.embedding_utils import load_contexts
from utils.embedding_utils import create_context_embeddings
from sentence_transformers.util import semantic_search

def retrieve_context(query, top_k=1):

    """Retrieve all contexts and embedding."""
    contexts = load_contexts("data/context_data.json")
    embeddings, embedding_model = create_context_embeddings(contexts)

    """Retrieve the most relevant context for a given query."""
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    results = semantic_search(query_embedding, embeddings, top_k=top_k)
    top_contexts = [contexts[hit["corpus_id"]] for hit in results[0]]
    
    return top_contexts