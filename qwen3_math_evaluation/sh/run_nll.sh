# Evaluate Qwen2.5-Math-Instruct

TOP_K="10 12 14 16 20 22 24"

# Distribute jobs across 8 GPUs
gpu=0
for k in $TOP_K; do
    echo "Running evaluation with top_k=$k on GPU $gpu"

    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    TOKENIZERS_PARALLELISM=false \
    python3 -u sh/nll.py --num_experts_per_tok $k --gt True
    
    # Increment GPU and wrap around to 0 after 7
    # gpu=$(( (gpu + 1) % 8 ))
    
    # # Wait for a GPU to free up if we've launched 8 jobs
    # if [ $((gpu % 8)) -eq 0 ]; then
    #     wait
    # fi
done

# Wait for any remaining jobs to complete
# wait