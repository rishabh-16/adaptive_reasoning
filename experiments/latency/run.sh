#!/bin/bash
#SBATCH --job-name=latency
#SBATCH --array=0-11
#SBATCH --gpus=1

TOP_K_VALUES=(1 2 4 6 8 10 12 14 16 20 22 24)
OUTPUT_TOKENS_VALUES=(2048)

# Calculate total number of combinations
# TOTAL_COMBINATIONS=$((${#TOP_K_VALUES[@]} * ${#OUTPUT_TOKENS_VALUES[@]}))

# Get current combination from array task ID
# TOP_K_INDEX=$((SLURM_ARRAY_TASK_ID / ${#OUTPUT_TOKENS_VALUES[@]}))
# OUTPUT_TOKENS_INDEX=$((SLURM_ARRAY_TASK_ID % ${#OUTPUT_TOKENS_VALUES[@]}))

TOP_K=${TOP_K_VALUES[$SLURM_ARRAY_TASK_ID]}
OUTPUT_TOKENS=${OUTPUT_TOKENS_VALUES[0]}

echo "Running SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID with top_k=$TOP_K and output_tokens=$OUTPUT_TOKENS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Assigned GPUs:"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
echo "--------------------------------"

TOKENIZERS_PARALLELISM=false python3 -u qwen3_math_evaluation/benchmark.py \
    --top_k $TOP_K \
    --output_tokens $OUTPUT_TOKENS
