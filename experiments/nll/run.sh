#!/bin/bash
#SBATCH --job-name=nll
#SBATCH --array=0-3
#SBATCH --gpus=2

TOP_K_VALUES=(1 2 4 8)
TOP_K=${TOP_K_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Running SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID with top_k=$TOP_K"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Assigned GPUs:"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
echo "--------------------------------"


TOKENIZERS_PARALLELISM=false python3 -u qwen3_math_evaluation/sh/nll.py --num_experts_per_tok $TOP_K --gt True
