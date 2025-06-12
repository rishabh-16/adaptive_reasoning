#!/bin/bash

# SLURM configuration
#SBATCH --job-name=qwen_training
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=your_partition  # Replace with your partition name

# Create logs directory
mkdir -p logs

# Load required modules (modify according to your cluster setup)
module load cuda/11.8
module load anaconda3

# Activate conda environment
source activate your_env_name  # Replace with your environment name

# Set environment variables
export WANDB_API_KEY="your_wandb_api_key"  # Replace with your W&B API key
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

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

# Generate trajectories
echo "Starting trajectory generation..."
python generate_trajectories.py
check_status "Trajectory generation"

# Fine-tune model
echo "Starting model fine-tuning..."
python train_sft.py
check_status "Model fine-tuning"

echo "Training pipeline completed successfully!" 