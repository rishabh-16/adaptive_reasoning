import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
from tqdm import tqdm
import os

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def generate_reasoning_trajectory(model, tokenizer, prompt, max_length=2048):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Load the larger Qwen model
    model_name = "Qwen/Qwen-72B"  # or another large Qwen model
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Load your dataset (modify this according to your dataset)
    dataset = load_dataset("your_dataset_name", split="train")
    
    # Create output directory
    os.makedirs("trajectories", exist_ok=True)
    
    # Generate trajectories
    trajectories = []
    for item in tqdm(dataset):
        prompt = item["prompt"]  # Modify according to your dataset structure
        
        # Generate reasoning trajectory
        trajectory = generate_reasoning_trajectory(model, tokenizer, prompt)
        
        # Store the trajectory
        trajectories.append({
            "prompt": prompt,
            "trajectory": trajectory
        })
    
    # Save trajectories
    with open("trajectories/reasoning_trajectories.json", "w") as f:
        json.dump(trajectories, f, indent=2)

if __name__ == "__main__":
    main() 