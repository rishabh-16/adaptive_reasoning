#!/bin/bash
#SBATCH --job-name=01_training
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --account=compact-models 
#SBATCH --qos=h200_compact-models_high
#SBATCH --output=experiments/logs/01_training/%x_%j.out
#SBATCH --mem=400GB
#SBATCH --time=24:00:00

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
export OMP_NUM_THREADS=32  # Set OpenMP threads to match cpus-per-task
export ALLOW_EXTRA_ARGS=1
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}  # Set Triton cache to local tmp to avoid NFS issues

# Activate conda environment
source ~/.bashrc
conda activate 01_training

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_SHOW_CPP_STACKTRACES=1
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# Define output directory variable
output_dir="/checkpoint/compact-models/rishabhtiwari/adaptive_reasoning/experiments/01_training/saved_models/OpenThinker3-30B-qwen3-$SLURM_JOB_ID"

cd /home/rishabhtiwari/adaptive_reasoning/LLaMA-Factory/

# Build the command with optional num_experts_per_tok override
cmd="llamafactory-cli train ../train_configs/OpenThinker3_qwen3.yaml --output_dir \"$output_dir\""
if [ -n "$NUM_EXPERTS_PER_TOK" ]; then
    echo "Overriding num_experts_per_tok to: $NUM_EXPERTS_PER_TOK"
    cmd="$cmd num_experts_per_tok=$NUM_EXPERTS_PER_TOK"
fi

echo "Running command: $cmd"
eval $cmd

# Check if output directory exists and is empty, then remove it
if [ -d "$output_dir" ] && [ -z "$(ls -A "$output_dir")" ]; then
    echo "Output directory $output_dir is empty, removing it..."
    rmdir "$output_dir"
fi
