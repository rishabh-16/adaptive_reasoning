#!/bin/bash
#SBATCH --job-name=01_training
#SBATCH --gpus=8
#SBATCH --account=genai_interns 
#SBATCH --qos=genai_interns
#SBATCH --output=experiments/logs/01_training/%x_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB

echo "Running 01_training"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Assigned GPUs:"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
echo "--------------------------------"

export MASTER_ADDR=$(hostname)
# Dynamic port selection to avoid conflicts
export MASTER_PORT=$((29500 + $SLURM_JOB_ID % 1000))
export WORLD_SIZE=$SLURM_GPUS_ON_NODE
export RANK=0
export LOCAL_RANK=0
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings
export OMP_NUM_THREADS=16  # Set OpenMP threads to match cpus-per-task
export ALLOW_EXTRA_ARGS=1


cd /home/rishabhtiwari/repos/01_META_REASONING_MOE/LLaMA-Factory/
llamafactory-cli train ../train_configs/OpenThinker3_debug.yaml --output_dir ../experiments/01_training/saves/OpenThinker3-30B
