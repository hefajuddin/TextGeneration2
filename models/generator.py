from transformers import AutoTokenizer, AutoModelForCausalLM

def load_generation_model(model_name="gpt2"):
    """Load a pre-trained text generation model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# def generate_text(prompt, model, tokenizer, max_length=1024):
#     """Generate text based on a prompt."""
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(
#         **inputs,
#         max_length=max_length,
#         num_beams=5,
#         early_stopping=True,
#         no_repeat_ngram_size=2
#     )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_text(prompt, model, tokenizer, max_new_tokens=50):
    """Generates text using the model and tokenizer."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,  # Restricts only the output tokens
        temperature=0.7,  # Controls randomness
        top_p=0.9,  # Nucleus sampling
        top_k=50,  # Limits token selection
        do_sample=True,  # Enables sampling instead of greedy decoding
        pad_token_id=tokenizer.eos_token_id,  # Avoid padding issues
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text



# def generate_text(prompt, model, tokenizer, max_length=150):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(
#         inputs["input_ids"],
#         max_length=max_length,
#         temperature=0.3,  # Lower creativity
#         num_return_sequences=1,
#     )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)