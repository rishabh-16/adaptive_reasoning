#!/bin/bash

# Model configurations
export LARGE_MODEL="Qwen/Qwen-72B"
export SMALL_MODEL="Qwen/Qwen-7B"

# Training configurations
export NUM_EPOCHS=3
export BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=8
export LEARNING_RATE=1e-5
export MAX_LENGTH=2048

# Generation configurations
export GENERATION_TEMPERATURE=0.7
export GENERATION_TOP_P=0.9

# Environment configurations
export WANDB_PROJECT="qwen-sft-training"
export WANDB_ENTITY="your_username"  # Replace with your W&B username

# Directory configurations
export TRAJECTORIES_DIR="trajectories"
export CHECKPOINTS_DIR="qwen-sft-checkpoints"
export FINAL_MODEL_DIR="qwen-sft-final"
export LOGS_DIR="logs" 