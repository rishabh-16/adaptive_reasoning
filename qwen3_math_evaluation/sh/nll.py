import json
import os
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def load_jsonl(filename):
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f]

def calculate_nlls(responses, lengths_of_questions):
    nlls = []
    for i in range(len(responses)):
        valid_logprobs = [list(lp.values())[0].logprob for lp in responses[i].prompt_logprobs[lengths_of_questions[i]:]]
        nll = -sum(valid_logprobs) / len(valid_logprobs)
        nlls.append(nll)
    return nlls

def get_prompts(jsonl_data, tokenizer, gt=False):
    prompt_temp = (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{input}<|im_end|>\n"
        "<|im_start|>assistant\n",
        "{output}",
        "\n\n",
    )
    input_template, output_template, _ = prompt_temp

    prompts = []
    prompts_with_preds = []
    lengths_of_questions = []
    
    for example in jsonl_data:
        question = example["question"].strip()
        # prediction = example["code"][0].strip()
        if gt:
            prediction = example["solution"].strip()
        else:
            prediction = example["code"][0].strip()
        full_prompt = input_template.format(input=question) + "<think>\n\n</think>\n\n"
        
        full_prompt = full_prompt.strip(" ")
        prompts.append(full_prompt)
        
        full_prompt_with_pred = full_prompt + output_template.format(output=prediction)
        full_prompt_with_pred = full_prompt_with_pred.strip(" ")
        prompts_with_preds.append(full_prompt_with_pred)
        
        lengths_of_questions.append(len(tokenizer.encode(full_prompt)))
        
    return prompts, prompts_with_preds, lengths_of_questions

def evaluate_model(prompts_with_preds, lengths_of_questions, num_experts_per_tok):
    llm = LLM(
        model="/fsx-project/rishabhtiwari/hf_cache/Qwen--Qwen3-30B-A3B",
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        trust_remote_code=True,
        hf_overrides={"num_experts_per_tok": num_experts_per_tok},
        max_model_len=8000,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=0,
    )
    
    responses = llm.generate(prompts_with_preds, sampling_params)
    nlls = calculate_nlls(responses, lengths_of_questions)
    return np.mean(nlls)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_experts_per_tok', type=int, required=True)
    parser.add_argument('--gt', type=bool, default=False)
    args = parser.parse_args()

    filename = "/home/rishabhtiwari/repos/01_META_REASONING_MOE/RSD/external/qwen25_math_evaluation/outputs/fsx-project/rishabhtiwari/hf_cache/Qwen--Qwen3-30B-A3B/math_eval/math500/test_qwen25-math-cot_-1_seed0_t0.0_top_k8_enable_thinkingFalse_s0_e-1.jsonl"
    
    jsonl_data = load_jsonl(filename)
    tokenizer = AutoTokenizer.from_pretrained("/fsx-project/rishabhtiwari/hf_cache/Qwen--Qwen3-30B-A3B")
    
    _, prompts_with_preds, lengths_of_questions = get_prompts(jsonl_data, tokenizer, args.gt)
    
    mean_nll = evaluate_model(prompts_with_preds, lengths_of_questions, args.num_experts_per_tok)
    result = {"num_experts_per_tok": args.num_experts_per_tok, "mean_nll": float(mean_nll)}
    print(result)
    output_file = f"/home/rishabhtiwari/repos/01_META_REASONING_MOE/RSD/external/qwen25_math_evaluation/outputs/nll_results_{args.num_experts_per_tok}_gt_{args.gt}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
