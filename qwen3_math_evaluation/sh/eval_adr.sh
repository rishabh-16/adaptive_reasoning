set -ex

PROMPT_TYPE="qwen25-math-cot"
MODEL_NAME_OR_PATH="Qwen/Qwen3-8B"
ENABLE_THINKING="false"
TOP_K=100
OUTPUT_DIR="adapt_reason/math_eval"
SPLIT="train"
NUM_TEST_SAMPLE=-1

# DATA_NAME="gsm8k,math,svamp,asdiv,mawps,carp_en,tabmwp,minerva_math,gaokao2023en,olympiadbench,college_math"
DATA_NAME="AI-MO/aimo-validation-aime"

QUANTIZE_MODEL=true
QUANTIZE_KV=false

if [ "${ENABLE_THINKING}" = "true" ]; then
    ENABLE_THINKING_FLAG="--enable_thinking true"
else
    ENABLE_THINKING_FLAG=""
fi

# Quantization
if [ "${QUANTIZE_MODEL}" = "true" ]; then
    QUANTIZE_MODEL_FLAG="--quantize_model true"
else
    QUANTIZE_MODEL_FLAG=""
fi

if [ "${QUANTIZE_KV}" = "true" ]; then
    QUANTIZE_KV_FLAG="--quantize_kv true"
else
    QUANTIZE_KV_FLAG=""
fi

N_SAMPLINGS_SWEEP=(1 4 16 64)
WBITS_SWEEP=(16 8 4 3 2)
ABITS_SWEEP=(16 8 4 3 2)
length=${#WBITS_SWEEP[@]}

for ((i=0; i<${length}; i++)); do
    N_SAMPLING=${N_SAMPLINGS_SWEEP[i]}
    WBITS=${WBITS_SWEEP[i]}
    ABITS=${ABITS_SWEEP[i]}
    echo "--------------------------------"
    echo "Quantization: Quantize_model: ${QUANTIZE_MODEL}, Quantize_kv: ${QUANTIZE_KV}"
    echo "N_SAMPLING: ${N_SAMPLING}, WBITS: ${WBITS}, ABITS: ${ABITS}, GROUPSIZE: 128"
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES=0 python3 -u math_eval.py \
        ${QUANTIZE_MODEL_FLAG} \
        ${QUANTIZE_KV_FLAG} \
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
        --n_sampling ${N_SAMPLING} \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --max_tokens_per_call 4096 \
        --save_outputs \
        --overwrite \
        --use_safetensors \
        --wbits ${WBITS} \
        --abits ${ABITS} \
        --groupsize 128
done