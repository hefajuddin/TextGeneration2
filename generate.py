from utils.embedding_utils import load_contexts, create_context_embeddings
from models.retriever import retrieve_context
from models.generator import load_generation_model, generate_text

def main():
    # Step 1: Load and Embed Contexts
    contexts = load_contexts("data/context_data.json")
    embeddings, embedding_model = create_context_embeddings(contexts)

    # Step 2: Load Generation Model
    gen_model, tokenizer = load_generation_model("gpt2")

    # Step 3: User Query

    while True:
        query=input("Ask me: ")
        if query !="":
            break

    # query = "Explain modular programming."
    top_contexts = retrieve_context(query, contexts, embeddings, embedding_model)
    # print(f"Retrieved Context: {top_contexts[0]}")

    # Step 4: Generate Text
    prompt = f"Based on the following context: {top_contexts[0]}\nAnswer: "
    generated_text = generate_text(prompt, gen_model, tokenizer)
    print(f"Generated Text: {generated_text}")

if __name__ == "__main__":
    main()