import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from datasets import Dataset
import json
from trl import SFTTrainer
import wandb
import os
from config import *

def load_trajectories(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def prepare_dataset(trajectories):
    # Convert trajectories to dataset format
    data = {
        "text": [f"Prompt: {item['prompt']}\nTrajectory: {item['trajectory']}" for item in trajectories]
    }
    return Dataset.from_dict(data)

def main():
    # Initialize wandb
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config={
            "model_name": SMALL_MODEL,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH
        }
    )
    
    # Load the smaller Qwen model
    tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SMALL_MODEL,
        torch_dtype=torch.float16,  # Use FP16
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load and prepare dataset
    trajectories = load_trajectories(f"{TRAJECTORIES_DIR}/reasoning_trajectories.json")
    dataset = prepare_dataset(trajectories)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CHECKPOINTS_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        fp16=True,  # Enable FP16 training
        optim="adamw_torch",  # Use standard AdamW optimizer
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        max_grad_norm=1.0  # Gradient clipping
    )
    
    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=MAX_LENGTH,
        packing=True  # Enable packing for more efficient training
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

if __name__ == "__main__":
    main() 