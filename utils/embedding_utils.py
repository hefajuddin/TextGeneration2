import json
from sentence_transformers import SentenceTransformer

def load_contexts(file_path):
    """Load contexts from JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return [item["text"] for item in data]

def create_context_embeddings(contexts, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for the contexts."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(contexts, convert_to_tensor=True)
    return embeddings, model