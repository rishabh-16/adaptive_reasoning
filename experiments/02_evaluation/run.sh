#!/bin/bash
#SBATCH --job-name=02_evaluation
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --array=0-53
#SBATCH --gpus-per-node=8
#SBATCH --account=compact-models 
#SBATCH --qos=h200_compact-models_high
#SBATCH --output=experiments/02_evaluation/logs/%x_%j.out
#SBATCH --mem=400GB
#SBATCH --time=24:00:00


source /home/rishabhtiwari/adaptive_reasoning/experiments/hyperparam_utils.sh

TOP_K_VALUES=(8 12 16)
THINKING_BUDGET_VALUES=(4096 8192 16384)
MODEL_NAME_OR_PATH_VALUES=("Qwen--Qwen3-30B-A3B" "Qwen--Qwen3-30B-A3B-Base" "Qwen--Qwen3-30B-A3B-Instruct-2507")
DATA_NAME_VALUES=("aime25" "math500")

# TOP_K_VALUES=(8 16)
# THINKING_BUDGET_VALUES=(4096)
# MODEL_NAME_OR_PATH_VALUES=("Qwen--Qwen3-30B-A3B-Instruct-2507")
# DATA_NAME_VALUES=("aime25")

HYPERPARAM_NAMES=(TOP_K_VALUES THINKING_BUDGET_VALUES MODEL_NAME_OR_PATH_VALUES DATA_NAME_VALUES)

declare -a HYPERPARAM_INDICES
compute_hyperparam_indices "$SLURM_ARRAY_TASK_ID" HYPERPARAM_NAMES HYPERPARAM_INDICES

TOP_K=${TOP_K_VALUES[${HYPERPARAM_INDICES[0]}]}
THINKING_BUDGET=${THINKING_BUDGET_VALUES[${HYPERPARAM_INDICES[1]}]}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH_VALUES[${HYPERPARAM_INDICES[2]}]}
DATA_NAME=${DATA_NAME_VALUES[${HYPERPARAM_INDICES[3]}]}

MODEL_FULL_PATH="/home/rishabhtiwari/hf_cache/${MODEL_NAME_OR_PATH}"

# Set MAX_TOKENS_PER_CALL based on THINKING_BUDGET
MAX_TOKENS_PER_CALL=$((THINKING_BUDGET + 4096))

# Set THINKING_BUDGET to -1 if model name doesn't contain 'base' or 'instruct'
if [[ "$MODEL_NAME_OR_PATH" == *"base"* || "$MODEL_NAME_OR_PATH" == *"Base"* || "$MODEL_NAME_OR_PATH" == *"instruct"* || "$MODEL_NAME_OR_PATH" == *"Instruct"* ]]; then
    THINKING_BUDGET=-1
fi

echo "top_k=$TOP_K and max_tokens_per_call=$MAX_TOKENS_PER_CALL and thinking_budget=$THINKING_BUDGET and model_name_or_path=$MODEL_NAME_OR_PATH and data_name=$DATA_NAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Assigned GPUs:"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
echo "--------------------------------"


OUTPUT_DIR="/home/rishabhtiwari/adaptive_reasoning/experiments/02_evaluation/outputs/${MODEL_NAME_OR_PATH}_max_tokens_per_call${MAX_TOKENS_PER_CALL}_thinking_budget${THINKING_BUDGET}"
mkdir -p "${OUTPUT_DIR}"
DIR_TO_CHECK="/home/rishabhtiwari/adaptive_reasoning/experiments/02_evaluation/outputs/${MODEL_NAME_OR_PATH}_max_tokens_per_call${MAX_TOKENS_PER_CALL}_thinking_budget${THINKING_BUDGET}/${DATA_NAME}/test_qwen25-math-cot_-1_seed0_t0.0_top_k${TOP_K}_enable_thinkingTrue_s0_e-1_qwen25-math-cot_metrics.json"
if [ -f "${DIR_TO_CHECK}" ]; then
    echo "File ${DIR_TO_CHECK} already exists. Skipping execution."
    exit 0
fi

SPLIT="test"
PROMPT_TYPE="qwen25-math-cot"
NUM_TEST_SAMPLE=-1


cd /home/rishabhtiwari/adaptive_reasoning/qwen3_math_evaluation
TORCHDYNAMO_VERBOSE=1 \
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_FULL_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --thinking_budget ${THINKING_BUDGET} \
    --top_k ${TOP_K} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --max_tokens_per_call ${MAX_TOKENS_PER_CALL} \
    --use_vllm \
    --save_outputs \
    --overwrite
