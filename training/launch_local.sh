#!/bin/bash

# Create logs directory
mkdir -p logs

# Set environment variables
export WANDB_API_KEY="your_wandb_api_key"  # Replace with your W&B API key

# Create output directories
mkdir -p trajectories
mkdir -p qwen-sft-checkpoints
mkdir -p qwen-sft-final

# Function to check if a command succeeded
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
        exit 1
    fi
}

# Function to display GPU usage
show_gpu_usage() {
    echo "Current GPU Usage:"
    nvidia-smi
}

# Generate trajectories
echo "Starting trajectory generation..."
show_gpu_usage
python generate_trajectories.py 2>&1 | tee logs/trajectory_generation.log
check_status "Trajectory generation"

# Fine-tune model
echo "Starting model fine-tuning..."
show_gpu_usage
python train_sft.py 2>&1 | tee logs/fine_tuning.log
check_status "Model fine-tuning"

echo "Training pipeline completed successfully!" 