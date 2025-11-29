import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_llm():
    """Identical to your original settings."""
    attn_implementation = "sdpa"
    model_id = "google/gemma-3-1b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation=attn_implementation
    )
    return tokenizer, model


def generate_answer(tokenizer, model, prompt, temperature=0.7, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        temperature=temperature,
        do_sample=True,
        max_new_tokens=max_new_tokens
    )

    text = tokenizer.decode(outputs[0])
    return text
