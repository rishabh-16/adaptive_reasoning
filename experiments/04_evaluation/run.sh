#!/bin/bash
#SBATCH --job-name=04_evaluation
#SBATCH --nodes=1
#SBATCH --array=0-1
#SBATCH --gpus=1
#SBATCH --account=genai_interns 
#SBATCH --qos=genai_interns
#SBATCH --output=experiments/04_evaluation/logs/%x_%j.out

source /home/rishabhtiwari/adaptive_reasoning/experiments/hyperparam_utils.sh

TOP_K_VALUES=(8)
CHECKPOINT_NUMBER_VALUES=(700)
MAX_TOKENS_PER_CALL_VALUES=(32000)
THINKING_BUDGET_VALUES=(16384)
ENABLE_THINKING="true"

HYPERPARAM_NAMES=(TOP_K_VALUES CHECKPOINT_NUMBER_VALUES MAX_TOKENS_PER_CALL_VALUES THINKING_BUDGET_VALUES)

declare -a HYPERPARAM_INDICES
SLURM_ARRAY_TASK_ID=0
compute_hyperparam_indices "$SLURM_ARRAY_TASK_ID" HYPERPARAM_NAMES HYPERPARAM_INDICES

TOP_K=${TOP_K_VALUES[${HYPERPARAM_INDICES[0]}]}
CHECKPOINT_NUMBER=${CHECKPOINT_NUMBER_VALUES[${HYPERPARAM_INDICES[1]}]}
MAX_TOKENS_PER_CALL=${MAX_TOKENS_PER_CALL_VALUES[${HYPERPARAM_INDICES[2]}]}
THINKING_BUDGET=${THINKING_BUDGET_VALUES[${HYPERPARAM_INDICES[3]}]}

echo "top_k=$TOP_K and checkpoint=$CHECKPOINT_NUMBER and max_tokens_per_call=$MAX_TOKENS_PER_CALL and thinking_budget=$THINKING_BUDGET"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Assigned GPUs:"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
echo "--------------------------------"

MODEL_NAME_OR_PATH="/home/rishabhtiwari/adaptive_reasoning/experiments/01_training/saved_models/OpenThinker3-30B-multi/checkpoint-${CHECKPOINT_NUMBER}/"
# MODEL_NAME_OR_PATH="/fsx-project/rishabhtiwari/hf_cache/Qwen--Qwen3-30B-A3B"
DATA_NAME="math500"
OUTPUT_DIR="/home/rishabhtiwari/adaptive_reasoning/experiments/04_evaluation/outputs/checkpoint${CHECKPOINT_NUMBER}_max_tokens_per_call${MAX_TOKENS_PER_CALL}_thinking_budget${THINKING_BUDGET}"
mkdir -p "${OUTPUT_DIR}"
DIR_TO_CHECK="/home/rishabhtiwari/adaptive_reasoning/experiments/04_evaluation/outputs/checkpoint${CHECKPOINT_NUMBER}_max_tokens_per_call${MAX_TOKENS_PER_CALL}_thinking_budget${THINKING_BUDGET}/${DATA_NAME}/test_qwen25-math-cot_-1_seed0_t0.0_top_k${TOP_K}_enable_thinkingTrue_s0_e-1_qwen25-math-cot_metrics.json"
if [ -f "${DIR_TO_CHECK}" ]; then
    echo "File ${DIR_TO_CHECK} already exists. Skipping execution."
    exit 0
fi

SPLIT="test"
PROMPT_TYPE="qwen25-math-cot"
NUM_TEST_SAMPLE=-1

if [ "${ENABLE_THINKING}" = "true" ]; then
    ENABLE_THINKING_FLAG="--enable_thinking true"
else
    ENABLE_THINKING_FLAG=""
fi

echo "Running with ENABLE_THINKING=${ENABLE_THINKING_FLAG}"

cd /home/rishabhtiwari/adaptive_reasoning/qwen3_math_evaluation
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
