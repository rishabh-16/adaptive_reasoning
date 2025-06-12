TOP_K="10 12 14 16 20 22 24"

# Distribute jobs across 8 GPUs

for k in $TOP_K; do
    echo "Running evaluation with top_k=$k"

    export CUDA_VISIBLE_DEVICES="0"
    TOKENIZERS_PARALLELISM=false \
    python3 -u benchmark.py --top_k $k

    # Increment GPU and wrap around to 0 after 7
    # gpu=$(( (gpu + 1) % 8 ))
    
    # # Wait for a GPU to free up if we've launched 8 jobs
    # if [ $((gpu % 8)) -eq 0 ]; then
    #     wait
    # fi
done