import random
import os
import argparse
import time
import json
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="/fsx-project/rishabhtiwari/hf_cache/Qwen--Qwen3-30B-A3B", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--output_tokens", default=1024, type=int)
    parser.add_argument("--top_k", default=8, type=int)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--use_vllm", type=bool, default=True)
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args

def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
            hf_overrides={"num_experts_per_tok": args.top_k},
        )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )
    main(llm, tokenizer, args)

def main(llm, tokenizer, args):
    examples = ["Write a long story about a house on fire."]

    ## warmup
    for i in range(1):
        if args.use_vllm:
            outputs = llm.generate(
                examples,
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.output_tokens,
                    min_tokens=args.output_tokens,
                    n=1,
                ),
            )
        else:
            tokenized_prompts = tokenizer(examples, padding="longest", return_tensors="pt", add_special_tokens=True)
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask

            if llm.device.type == "cuda":
                batch_input_ids = batch_input_ids.cuda()
                attention_mask = attention_mask.cuda()
            
            raise NotImplementedError("Not implemented")
    
    latencies = []
    for i in range(5):
        start_time = time.time()
        outputs = llm.generate(
            examples,
            SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.output_tokens,
                min_tokens=args.output_tokens,
                n=1,
            ),
        )
        latencies.append(time.time() - start_time)
    print(f"Average latency: {sum(latencies) / len(latencies)} seconds")
    # Save average latency to JSON
    avg_latency = sum(latencies) / len(latencies)
    output = {
        "avg_latency": avg_latency,
        "model": args.model_name_or_path,
        "output_tokens": args.output_tokens,
        "num_experts_per_tok": args.top_k
    }
    
    # Create outputs directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save with timestamp
    output_file = f"{args.output_dir}/latency_benchmark_num_experts_per_tok_{args.top_k}_output_tokens_{args.output_tokens}.json"
    
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)