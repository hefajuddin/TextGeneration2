from models.retriever import retrieve_context
from models.generator import load_generation_model, generate_text, post_process_answer
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("message")

    # Step 2: Retrieve Relevant Contexts
    top_contexts = retrieve_context(query)

    # Step 3: Combine Contexts for Prompt
    combined_context = " ".join(top_contexts)
    prompt = (
        f"Answer the following question based on the given context. Use your own words to provide a brief and accurate response:\n"
        f"Context: {combined_context}\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    # Step 4: Load Generator
    gen_model, tokenizer = load_generation_model("gpt2")

    # Step 5 Generating Answer
    generated_answer = generate_text(prompt, gen_model, tokenizer, max_new_tokens=50)

    # Extract only the answer portion
    answer = post_process_answer(prompt, generated_answer)
    return jsonify({"response": answer})

if __name__ == "__main__":
    print("\033[92m" + "Server is running successfully on http://127.0.0.1:5004" + "\033[0m")
    app.run(debug=True, port=5004)

