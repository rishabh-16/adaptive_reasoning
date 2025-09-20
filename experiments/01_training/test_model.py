import os
from transformers import AutoModelForCausalLM
import torch

# Note: model_name and tokenizer should be defined in a previous cell
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Set cache directory
cache_dir = "/home/rishabhtiwari/hf_cache"
os.makedirs(cache_dir, exist_ok=True)

# Download and load the model
print("Loading model (this may take a while for first download)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Model loaded successfully!")
print(f"Model device: {next(model.parameters()).device}")

# Sample math question
math_question = """Solve this step by step:
A train travels 240 miles in 3 hours. If it maintains the same speed, how long will it take to travel 400 miles?"""

# You'll need to import tokenizer and define it properly in your main script
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

Prepare the input
messages = [
    {"role": "user", "content": math_question}
]

Tokenize the input
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(input_text)
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

print(f"\nInput question: {math_question}")
print(f"\nGenerating response...")

Generate response
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

Decode the response
response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
print(f"\nModel response:\n{response}")
