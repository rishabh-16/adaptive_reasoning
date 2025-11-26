#!/bin/bash
#SBATCH --job-name=03_evaluation
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --array=0-11
#SBATCH --gpus-per-node=8
#SBATCH --account=transformer2
#SBATCH --qos=h200_comm_shared
#SBATCH --output=experiments/03_evaluation/logs/%x_%j.out
#SBATCH --mem=400GB
#SBATCH --time=24:00:00

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_SHOW_CPP_STACKTRACES=1
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1

source /home/rishabhtiwari/adaptive_reasoning/experiments/hyperparam_utils.sh

TOP_K_VALUES=(4 8 12 14 16 20)
THINKING_BUDGET_VALUES=(-1)
# MODEL_NAME_OR_PATH_VALUES=("Qwen3-Qwen3-30B-A3B-Base")
# MODEL_NAME_OR_PATH_VALUES=("qwen3-1695599" "qwen3-1695805" "qwen3-1676884" "qwen3-1695806" "qwen3-1677711" "qwen3-1695807" "qwen3-1677706")
MODEL_NAME_OR_PATH_VALUES=("qwen3-2111685" "qwen3-2111879")
DATA_NAME_VALUES=("aime25")
CHECKPOINT_NUMBER_VALUES=(276)
SEEDS=(0)
N_SAMPLING=8
TEMPERATURE=0.7
#(8 12 16 20)
#("base-1235229" "base-1235291")
# 1398948
# (4096 8192 16384 28672)
# CHECKPOINT_NUMBER_VALUES=(1104)

HYPERPARAM_NAMES=(TOP_K_VALUES THINKING_BUDGET_VALUES MODEL_NAME_OR_PATH_VALUES DATA_NAME_VALUES CHECKPOINT_NUMBER_VALUES SEEDS)

declare -a HYPERPARAM_INDICES
# SLURM_ARRAY_TASK_ID=0
compute_hyperparam_indices "$SLURM_ARRAY_TASK_ID" HYPERPARAM_NAMES HYPERPARAM_INDICES

TOP_K=${TOP_K_VALUES[${HYPERPARAM_INDICES[0]}]}
THINKING_BUDGET=${THINKING_BUDGET_VALUES[${HYPERPARAM_INDICES[1]}]}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH_VALUES[${HYPERPARAM_INDICES[2]}]}
DATA_NAME=${DATA_NAME_VALUES[${HYPERPARAM_INDICES[3]}]}
CHECKPOINT_NUMBER=${CHECKPOINT_NUMBER_VALUES[${HYPERPARAM_INDICES[4]}]}
SEED=${SEEDS[${HYPERPARAM_INDICES[5]}]}

# MODEL_FULL_PATH="/checkpoint/compact-models/rishabhtiwari/adaptive_reasoning/experiments/01_training/saved_models/OpenThinker3-${MODEL_NAME_OR_PATH}/checkpoint-${CHECKPOINT_NUMBER}/"
MODEL_FULL_PATH="/checkpoint/transformer2/rishabhtiwari/adaptive_reasoning/experiments/01_training/saved_models/OpenThinker3-${MODEL_NAME_OR_PATH}"
# MODEL_FULL_PATH="/home/rishabhtiwari/hf_cache/Qwen--Qwen3-30B-A3B-Base"
# MAX_TOKENS_PER_CALL=$((THINKING_BUDGET + 2048))
MAX_TOKENS_PER_CALL=16000


echo "top_k=$TOP_K and checkpoint=$CHECKPOINT_NUMBER and model_name_or_path=$MODEL_NAME_OR_PATH and data_name=$DATA_NAME and thinking_budget=$THINKING_BUDGET and seed=$SEED"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Assigned GPUs:"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
echo "--------------------------------"

OUTPUT_DIR="/home/rishabhtiwari/adaptive_reasoning/experiments/03_evaluation/short_runs/${MODEL_NAME_OR_PATH}_checkpoint${CHECKPOINT_NUMBER}_max_tokens_per_call${MAX_TOKENS_PER_CALL}_thinking_budget${THINKING_BUDGET}"
mkdir -p "${OUTPUT_DIR}"
DIR_TO_CHECK="/home/rishabhtiwari/adaptive_reasoning/experiments/03_evaluation/short_runs/${MODEL_NAME_OR_PATH}_checkpoint${CHECKPOINT_NUMBER}_max_tokens_per_call${MAX_TOKENS_PER_CALL}_thinking_budget${THINKING_BUDGET}/${DATA_NAME}/test_qwen25-math-cot_-1_seed${SEED}_t${TEMPERATURE}_top_k${TOP_K}_s0_e-1_qwen25-math-cot_metrics.json"
if [ -f "${DIR_TO_CHECK}" ]; then
    echo "File ${DIR_TO_CHECK} already exists. Skipping execution."
    exit 0
fi

SPLIT="test"
PROMPT_TYPE="qwen25-math-cot"
# PROMPT_TYPE="ling"
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
    --temperature ${TEMPERATURE} \
    --n_sampling ${N_SAMPLING} \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --max_tokens_per_call ${MAX_TOKENS_PER_CALL} \
    --use_vllm \
    --save_outputs \
    --overwrite
    # --tensor_parallel_size 4 \
    # --top_p 0.95 \
    # --temperature ${TEMPERATURE} \
    # --n_sampling ${N_SAMPLING} \
    # --max_model_len 40960 \
