from transformers import AutoTokenizer, AutoModelForCausalLM

def load_generation_model(model_name="gpt2"):
    """Load a pre-trained text generation model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_length=1024):
    """Generate text based on a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)