#!/bin/bash
#SBATCH --job-name=01_training_multi
#SBATCH --gpus=8
#SBATCH --account=genai_interns 
#SBATCH --qos=genai_interns
#SBATCH --output=experiments/01_training/logs/%x_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --distribution=block

echo "Running 01_training on multiple nodes"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total number of tasks: $SLURM_NTASKS"
echo "Node list: $SLURM_NODELIST"
echo "My node ID: $SLURM_NODEID"
echo "My task ID: $SLURM_PROCID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Assigned GPUs:"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
echo "--------------------------------"

# Set up distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Dynamic port selection to avoid conflicts
export MASTER_PORT=$((29500 + $SLURM_JOB_ID % 1000))
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings
export OMP_NUM_THREADS=8  # Set OpenMP threads to match cpus-per-task
export ALLOW_EXTRA_ARGS=1

# Additional environment variables for better multi-node performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# Optional: Set NCCL network interface if needed
# export NCCL_SOCKET_IFNAME=eth0

echo "Distributed training setup:"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "LOCAL_RANK: $LOCAL_RANK"
echo "--------------------------------"

cd /fsx-project/rishabhtiwari/01_META_REASONING_MOE/Llama-Factory/

# Run the training command with multi-node configuration
llamafactory-cli train ../train_configs/OpenThinker3_multi_node.yaml --output_dir ../experiments/01_training/saves/OpenThinker3-30B-multi