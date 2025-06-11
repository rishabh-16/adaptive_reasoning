# Evaluate Qwen2.5-Math-Instruct
PROMPT_TYPE="qwen25-math-cot"
HF_BASE_PATH="/fsx-project/rishabhtiwari/hf_cache"
ENABLE_THINKING=false
TOP_K="8 1 2 4 6 10 12 14 16 20 22 24"

# Qwen2.5-Math-1.5B-Instruct
gpu=0
for k in $TOP_K; do
    echo "Running evaluation with top_k=$k on GPU $gpu"

    export CUDA_VISIBLE_DEVICES="$gpu"
    MODEL_NAME_OR_PATH="${HF_BASE_PATH}/Qwen--Qwen3-30B-A3B"
    bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $ENABLE_THINKING $k

    # Increment GPU ID and wrap around to 0 after 3
    # gpu=$(((gpu + 1) % 4))

    # If we've launched 4 jobs, wait for them to complete before continuing
    # if [ $gpu -eq 0 ]; then
    #     wait
    # fi
done

# Wait for any remaining jobs
# wait


# # Qwen2.5-Math-7B-Instruct
# export CUDA_VISIBLE_DEVICES="0"
# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# # Qwen2.5-Math-72B-Instruct
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-72B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH


# # Evaluate Qwen2-Math-Instruct
# PROMPT_TYPE="qwen-boxed"

# # Qwen2-Math-1.5B-Instruct
# export CUDA_VISIBLE_DEVICES="0"
# MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-1.5B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# # Qwen2-Math-7B-Instruct
# export CUDA_VISIBLE_DEVICES="0"
# MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-7B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# # Qwen2-Math-72B-Instruct
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# MODEL_NAME_OR_PATH="Qwen/Qwen2-Math-72B-Instruct"
# bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
