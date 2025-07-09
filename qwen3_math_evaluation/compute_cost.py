import json
import pdb
from utils import *
from quantization_code.costmodel import get_cost
from math_eval import parse_args
from model_utils import load_hf_lm_and_tokenizer
from evaluate import evaluate

def compute_cost(x, tokenizer, model, args):
    data_list = args.data_names.split(",")
    full_prompt = construct_prompt(x, data_list[0], args)
    if args.thinking_budget == 0:
        full_prompt+="<think>\n\n</think>\n\n"
    if args.apply_chat_template:
        full_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": full_prompt.strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )
    num_input_tokens = len(tokenizer.encode(full_prompt))
    num_output_tokens = list(map(lambda x: len(tokenizer.encode(x)), x["code"]))
    return get_cost(model=model,
                    num_input_tokens=num_input_tokens,
                    num_output_tokens=num_output_tokens,
                    weight_precision=args.wbits,
                    activation_precision=args.abits,
                    groupsize=args.groupsize)

def majority_vote(scores):
    if not scores:  # Handle empty list
        return None
    
    true_count = scores.count(True)
    false_count = scores.count(False)
    
    if true_count > false_count:
        return True
    elif false_count > true_count:
        return False
    else:
        # In case of a tie, return the first value
        return scores[0]


if __name__ == "__main__":
    args = parse_args()
    out_dir = "costs_qkv"
    input_dir = "outputs/adapt_reason/math_eval/AI-MO/aimo-validation-aime"
    file_path = lambda dire, wbits, abits, quantize_kv, quantize_model, n_sampling: f'{dire}/train_qwen25-math-cot_-1_seed0_t0.0_top_k100_enable_thinkingFalse_quantize_kv{quantize_kv}_quantize_model{quantize_model}_wbits{wbits}_abits{abits}_groupsize128_n_sampling{n_sampling}_s0_e-1.jsonl'
    all_data = list(load_jsonl(file_path(input_dir, args.wbits, args.abits, args.quantize_kv, args.quantize_model, args.n_sampling)))
    llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
            args=args,
        )
    total_score = 0
    total_samples = 0
    for x in all_data:
        cost, model_weight_mem_ops, model_kv_mem_ops = compute_cost(x, tokenizer, llm, args)
        x["cost"] = cost
        x["model_weight_mem_ops"] = model_weight_mem_ops
        x["model_kv_mem_ops"] = model_kv_mem_ops
        x['best_of_n_score'] = majority_vote(x['score'])
        total_score += int(x['best_of_n_score'])
        total_samples += 1
    print("================================================")
    print("file path: ", file_path(input_dir, args.wbits, args.abits, args.quantize_kv, args.quantize_model, args.n_sampling))
    print(f"Average score: {round(total_score / total_samples, 5) * 100}%")
    print(f'model_weight_mem_ops: {model_weight_mem_ops}')
    print(f'model_kv_mem_ops: {model_kv_mem_ops}')
    print(f'cost: {cost}')
    save_jsonl(all_data, file_path(out_dir, args.wbits, args.abits, args.quantize_kv, args.quantize_model, args.n_sampling))
    print("================================================")