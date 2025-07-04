#!/bin/bash
#SBATCH --job-name=03_evaluation
#SBATCH --nodes=1
#SBATCH --array=0-13
#SBATCH --gpus=1
#SBATCH --account=genai_interns 
#SBATCH --qos=genai_interns
#SBATCH --output=experiments/03_evaluation/logs/%x_%j.out

source /home/rishabhtiwari/repos/01_META_REASONING_MOE/experiments/hyperparam_utils.sh

TOP_K_VALUES=(1 2 4 8 16 24 32)
# TOP_K_VALUES=(20 24 28 32)
CHECKPOINT_NUMBER_VALUES=(2900)
MAX_TOKENS_PER_CALL_VALUES=(32000)
ENABLE_THINKING="true"

HYPERPARAM_NAMES=(TOP_K_VALUES CHECKPOINT_NUMBER_VALUES MAX_TOKENS_PER_CALL_VALUES)

declare -a HYPERPARAM_INDICES

compute_hyperparam_indices "$SLURM_ARRAY_TASK_ID" HYPERPARAM_NAMES HYPERPARAM_INDICES

TOP_K=${TOP_K_VALUES[${HYPERPARAM_INDICES[0]}]}
CHECKPOINT_NUMBER=${CHECKPOINT_NUMBER_VALUES[${HYPERPARAM_INDICES[1]}]}
MAX_TOKENS_PER_CALL=${MAX_TOKENS_PER_CALL_VALUES[${HYPERPARAM_INDICES[2]}]}

echo "Running SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID with top_k=$TOP_K and checkpoint=$CHECKPOINT_NUMBER and max_tokens_per_call=$MAX_TOKENS_PER_CALL"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Assigned GPUs:"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
echo "--------------------------------"

MODEL_NAME_OR_PATH="/fsx-project/rishabhtiwari/saved_models/OpenThinker3-30B/checkpoint-${CHECKPOINT_NUMBER}/"
DATA_NAME="aime24"
OUTPUT_DIR="/home/rishabhtiwari/repos/01_META_REASONING_MOE/experiments/03_evaluation/outputs/checkpoint${CHECKPOINT_NUMBER}_max_tokens_per_call${MAX_TOKENS_PER_CALL}"
mkdir -p "${OUTPUT_DIR}"

# DIR_TO_CHECK="/home/rishabhtiwari/repos/01_META_REASONING_MOE/experiments/02_evaluation/outputs/checkpoint${CHECKPOINT_NUMBER}_max_tokens_per_call${MAX_TOKENS_PER_CALL}/math500/test_qwen25-math-cot_-1_seed0_t0.0_top_k${TOP_K}_enable_thinkingFalse_s0_e-1_qwen25-math-cot_metrics.json"
# if [ -f "${DIR_TO_CHECK}" ]; then
#     echo "File ${DIR_TO_CHECK} already exists. Skipping execution."
#     exit 0
# fi

SPLIT="test"
PROMPT_TYPE="qwen25-math-cot"
NUM_TEST_SAMPLE=-1

if [ "${ENABLE_THINKING}" = "true" ]; then
    ENABLE_THINKING_FLAG="--enable_thinking true"
else
    ENABLE_THINKING_FLAG=""
fi

echo "Running with ENABLE_THINKING=${ENABLE_THINKING_FLAG}"

cd /home/rishabhtiwari/repos/01_META_REASONING_MOE/qwen3_math_evaluation
TORCHDYNAMO_VERBOSE=1 \
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    ${ENABLE_THINKING_FLAG} \
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
