#!/bin/bash
#SBATCH --job-name=01_training_multi
#SBATCH --exclusive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --account=compact-models 
#SBATCH --qos=h200_capabilities_shared
#SBATCH --output=experiments/01_training/logs/%x_%j.out
#SBATCH --mem=400GB
#SBATCH --time=72:00:00

echo "Nodes: $SLURM_JOB_NODELIST"
echo "Total tasks: $SLURM_NTASKS"


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
# export WORLD_SIZE=$SLURM_NTASKS
# export RANK=$SLURM_PROCID
# export LOCAL_RANK=$SLURM_LOCALID
# export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings

# llama-factory expects these:
export FORCE_TORCHRUN=1
# export OMP_NUM_THREADS=1  # Set OpenMP threads to match cpus-per-task
export ALLOW_EXTRA_ARGS=1
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

# Additional environment variables for better multi-node performance
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/tmp/triton_cache  # Set Triton cache to local tmp to avoid NFS issues


# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_P2P_DISABLE=0
# export NCCL_BLOCKING_WAIT=1
# 

# Optional: Set NCCL network interface if needed
# export NCCL_SOCKET_IFNAME=eth0

echo "Distributed training setup:"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
# echo "WORLD_SIZE: $WORLD_SIZE"
# echo "RANK: $RANK"
# echo "LOCAL_RANK: $LOCAL_RANK"
# echo "--------------------------------"

# Define output directory variable
# output_dir="/checkpoint/compact-models/rishabhtiwari/adaptive_reasoning/experiments/01_training/saved_models/OpenThinker3-30B-base-$SLURM_JOB_ID"

cd /home/rishabhtiwari/adaptive_reasoning/LLaMA-Factory/

# Run the training command with multi-node configuration
# srun --export=ALL echo "I am node $SLURM_NODEID of $SLURM_NNODES"
# srun --export=ALL bash -c 'echo "I am node $SLURM_NODEID of $SLURM_NNODES"'

srun --export=ALL bash -c '
  output_dir="/checkpoint/compact-models/rishabhtiwari/adaptive_reasoning/experiments/01_training/saved_models/OpenThinker3-qwen3-$SLURM_JOB_ID"
  echo "Node rank: $SLURM_NODEID of $SLURM_NNODES"
  echo "Experts per token: $NUM_EXPERTS_PER_TOK"
  cmd="FORCE_TORCHRUN=1 \
  NNODES=$SLURM_NNODES \
  NODE_RANK=$SLURM_NODEID \
  MASTER_ADDR=$MASTER_ADDR \
  MASTER_PORT=$MASTER_PORT \
  llamafactory-cli train ../train_configs/OpenThinker3_qwen3.yaml output_dir=$output_dir"
  
  if [ -n "$NUM_EXPERTS_PER_TOK" ]; then
      echo "Overriding num_experts_per_tok to: $NUM_EXPERTS_PER_TOK"
      cmd="$cmd num_experts_per_tok=$NUM_EXPERTS_PER_TOK"
  fi
  echo "Running command: $cmd"
  eval $cmd
'

# # Check if output directory exists and is empty, then remove it (only on master node)
# if [ "$SLURM_NODEID" -eq 0 ]; then
#     if [ -d "$output_dir" ] && [ -z "$(ls -A "$output_dir")" ]; then
#         echo "Output directory $output_dir is empty, removing it..."
#         rmdir "$output_dir"
#     fi
# fi
