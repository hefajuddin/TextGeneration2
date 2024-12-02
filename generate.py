# from utils.embedding_utils import load_contexts, create_context_embeddings
# from models.retriever import retrieve_context
# from models.generator import load_generation_model, generate_text

# def main():
#     # Step 1: Load and Embed Contexts
#     contexts = load_contexts("data/context_data.json")
#     embeddings, embedding_model = create_context_embeddings(contexts)

#     # Step 2: Load Generation Model
#     gen_model, tokenizer = load_generation_model("gpt2")

#     # Step 3: User Query

#     while True:
#         query=input("Ask me: ")
#         if query !="":
#             break

#     # query = "Explain modular programming."
#     top_contexts = retrieve_context(query, contexts, embeddings, embedding_model)
#     # print(f"Retrieved Context: {top_contexts[0]}")

#     # Step 4: Generate Text
#     prompt = f"Based on the following context: {top_contexts[0]}\nAnswer: "
#     generated_text = generate_text(prompt, gen_model, tokenizer)
#     print(f"Generated Text: {generated_text}")

# if __name__ == "__main__":
#     main()



from utils.embedding_utils import load_contexts, create_context_embeddings
from models.retriever import retrieve_context
from models.generator import load_generation_model, generate_text

def main():
    # Step 1: Load and Embed Contexts
    # print("Loading context data and creating embeddings...")
    contexts = load_contexts("data/context_data.json")
    embeddings, embedding_model = create_context_embeddings(contexts)

    # Step 2: Load Generation Model
    print("Loading the generation model...")
    gen_model, tokenizer = load_generation_model("gpt2")

    # Step 3: Interactive Q&A Loop
    while True:
        query=input("Ask me: ")
        if query !="":
            break

    # Step 4: Retrieve Relevant Contexts
    top_contexts = retrieve_context(query, contexts, embeddings, embedding_model, top_k=1)
    # print("Top Retrieved Contexts:")
    # for i, context in enumerate(top_contexts, 1):
    #     print(f"Context {i}: {context}")

    # Step 5: Combine Contexts for Prompt
    combined_context = " ".join(top_contexts)
    # prompt = f"Based on the following contexts: {combined_context}\nQuestion: {query}\nAnswer: "
    # Enforce that the answer is based strictly on the retrieved context
    prompt = (
        f"Use the following context to answer the question:\n"
        f"Context: {combined_context}\n"
        f"Question: {query}\n"
        f"Answer based strictly on the context:"
    )

    # Step 6: Generate the Answer
    # generated_answer = generate_text(prompt, gen_model, tokenizer)
    # print(f"\nAnswer: {generated_answer}")
    generated_answer = generate_text(prompt, gen_model, tokenizer)  # Limit output length
    print(f"{generated_answer}")

if __name__ == "__main__":
    main()
