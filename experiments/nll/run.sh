#!/bin/bash
#SBATCH --job-name=first
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=$1

TOP_K="1 2 4 6 8 10 12 14 16 20 22 24"

# Run evaluations in parallel using srun
for k in $TOP_K; do
    echo "Running evaluation with top_k=$k"
    
    srun --ntasks=1 --gpus-per-task=$1 \
         bash -c "TOKENIZERS_PARALLELISM=false \
         python3 -u qwen3_math_evaluation/sh/nll.py --num_experts_per_tok $k --gt True" &
done

wait