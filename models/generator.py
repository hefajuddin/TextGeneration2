from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_generation_model(model_name="gpt2"):
    """Load a pre-trained text generation model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_text(prompt, model, tokenizer, max_new_tokens=50):

    """Generates text using the model and tokenizer."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True)

    # Create attention mask
    attention_mask = torch.ones_like(input_ids)  # Mark all tokens as "attended"

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

def post_process_answer(prompt, generated_text):
    """Extracts the answer from the model's generated text."""
    # Remove the prompt from the generated text
    answer = generated_text[len(prompt):].strip()
    # Optionally clean up and truncate the answer to avoid context repetition
    answer = answer.split("\n")[0]  # Take only the first line of the generated output

    return answer
