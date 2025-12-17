import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import datasets
import torch

@dataclass
class ReAnnotateConfig:
    input_path: str
    output_path: str
    model: str
    batch_size: int = 1
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95
    system_prompt: Optional[str] = None
    resume: bool = True
    prompt_template: Optional[str] = None
    # Parallelization config
    split_id: Optional[int] = None
    split_size: int = 1000


# DEFAULT_SYSTEM_PROMPT = "Please reason step by step."
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant.

STRICT OUTPUT FORMAT REQUIREMENTS:

1) Output MUST begin with "<reasoning_start>" on its own line.
2) Put ALL detailed reasoning, intermediate steps, and calculations ONLY between:
   <reasoning_start>
   ...
   </reasoning_end>
3) Do NOT include any reasoning outside these tags.
4) After </reasoning_end>, provide a concise step-by-step solution (short, clear, minimal).
5) Conclude with the final answer written exactly as: \\boxed{FINAL_ANSWER}
6) Do NOT add any extra text, explanations, or sections beyond what is specified.

Now solve the user’s request.
"""
DEFAULT_PROMPT_TEMPLATE = "{input}"


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines but continue processing the file
                continue
    return data


def write_jsonl(path: str, rows: Iterable[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str, rows: Iterable[Dict]):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def iter_resume_index(output_path: str) -> int:
    if not os.path.exists(output_path):
        return 0
    # Count lines to resume
    with open(output_path, "r", encoding="utf-8") as f:
        for i, _ in enumerate(f, start=1):
            pass
    return i if "i" in locals() else 0


def build_chat_messages(system_prompt: str, user_content: str) -> List[Dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def to_model_input(sample: Dict, template: str) -> str:
    return template.format(input=sample[0]["value"])


def prepare_requests(
    tokenizer: AutoTokenizer,
    samples: Sequence[Dict],
    indices: Sequence[int],
    system_prompt: str,
    template: str,
) -> Tuple[List[str], List[Tuple[int, Dict]]]:
    requests = []
    metadata: List[Tuple[int, Dict]] = []
    for sample, idx in zip(samples, indices):
        user_content = to_model_input(sample, template)
        messages = build_chat_messages(system_prompt, user_content)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        requests.append(text)
        metadata.append((idx, sample))
        if idx == 0:
            print("input example: ", text)
    return requests, metadata


def generate_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    prompts: Sequence[str],
) -> List[str]:
    outputs = llm.generate(prompts, sampling_params)
    completions = []
    for output in outputs:
        if not output.outputs:
            completions.append("")
            continue
        completions.append(output.outputs[0].text.strip())
    return completions


def _with_split_suffix(path: str, split_id: Optional[int]) -> str:
    if split_id is None:
        return path
    dirname, basename = os.path.dirname(path), os.path.basename(path)
    name, ext = os.path.splitext(basename)
    return os.path.join(dirname, f"{name}-split{split_id}{ext}")


def reannotate(cfg: ReAnnotateConfig):
    # Compute output path with split suffix if provided
    output_path = _with_split_suffix(cfg.output_path, cfg.split_id)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    data = datasets.load_dataset(cfg.input_path, cache_dir="/checkpoint/agentic-models/rishabhtiwari/hf_cache")["train"]

    # Determine split bounds
    total_global = len(data)
    split_size = max(cfg.split_size, 1)
    if cfg.split_id is not None:
        split_start = cfg.split_id * split_size
        split_end = min(split_start + split_size, total_global)
        if split_start >= total_global:
            print(f"[Info] split_id={cfg.split_id} is out of range for dataset of size {total_global}.", file=sys.stderr)
            print("Done.", file=sys.stderr)
            return
    else:
        split_start = 0
        split_end = total_global

    # Resume logic: resume within the split
    resume_count = iter_resume_index(output_path) if cfg.resume else 0
    start_idx_global = split_start + resume_count

    system_prompt = cfg.system_prompt or DEFAULT_SYSTEM_PROMPT
    prompt_template = cfg.prompt_template or DEFAULT_PROMPT_TEMPLATE

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model,
        use_fast=True,
        trust_remote_code=True,
    )
    # Use all available GPUs with tensor parallelism
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs with tensor parallelism", file=sys.stderr)
    llm = LLM(
        model=cfg.model,
        trust_remote_code=True,
        tensor_parallel_size=num_gpus,
    )

    if resume_count == 0:
        open(output_path, "w", encoding="utf-8").close()

    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
    )

    total_in_split = split_end - split_start
    batch_size = max(cfg.batch_size, 1)

    for batch_start in range(start_idx_global, split_end, batch_size):
        batch_stop = min(batch_start + batch_size, split_end)
        batch_samples = data[batch_start:batch_stop]["conversations"]
        batch_indices = list(range(batch_start, batch_start + len(batch_samples)))

        try:
            prompts, metadata = prepare_requests(
                tokenizer=tokenizer,
                samples=batch_samples,
                indices=batch_indices,
                system_prompt=system_prompt,
                template=prompt_template,
            )

            completions = generate_batch(
                llm=llm,
                sampling_params=sampling_params,
                prompts=prompts,
            )

            out_rows = []
            for completion, (idx, sample) in zip(completions, metadata):
                out_row = dict(sample)
                out_row["reannotation_model"] = cfg.model
                out_row["reannotation"] = completion
                out_row["_index"] = idx
                if idx == 0:
                    print("output example: ", completion)
                out_rows.append(out_row)

            append_jsonl(output_path, out_rows)

            processed_relative = (batch_indices[-1] + 1) - split_start
            if processed_relative % 100 == 0 or processed_relative == total_in_split:
                print(
                    f"[Progress] {processed_relative}/{total_in_split} samples processed in split {cfg.split_id if cfg.split_id is not None else 0}.",
                    file=sys.stderr,
                )

        except KeyboardInterrupt:
            print("Interrupted by user. Saving progress...", file=sys.stderr)
            break
        except Exception as e:
            print(
                f"[Warning] Batch starting at {batch_start} failed: {e}",
                file=sys.stderr,
            )
            out_rows = []
            for idx, sample in zip(batch_indices, batch_samples):
                out_row = dict(sample)
                out_row["reannotation_model"] = cfg.model
                out_row["reannotation"] = None
                out_row["error"] = str(e)
                out_row["_index"] = idx
                out_rows.append(out_row)
            append_jsonl(output_path, out_rows)

    print("Done.", file=sys.stderr)


def parse_args() -> ReAnnotateConfig:
    p = argparse.ArgumentParser(
        description="Reannotate OpenThoughts dataset using a local vLLM hosted model",
    )
    p.add_argument("--dataset", default="/home/rishabhtiwari/datasets/openthoughts3_small", help="Path to input OpenThoughts JSONL")
    p.add_argument("--output", default="/checkpoint/transformer2/rishabhtiwari/openthoughts_datagen/reannotation_metadata.jsonl", help="Path to output JSONL")
    p.add_argument(
        "--model",
        default="/home/rishabhtiwari/hf_cache/Qwen--Qwen3-235B-A22B-Instruct-2507",
        help="vLLM-compatible model identifier",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of samples to generate per vLLM call",
    )
    p.add_argument("--max-new-tokens", type=int, default=32000)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--system-prompt", type=str, default=None)
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from existing output",
    )
    p.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help="Override the default reannotation prompt template",
    )
    # Parallelization options
    p.add_argument("--split-id", type=int, default=None, help="Split id (0-indexed). If provided, processes only this split.")
    p.add_argument("--split-size", type=int, default=10, help="Number of inputs per split. Output file will be suffixed with -split{split_id}.")

    args = p.parse_args()

    return ReAnnotateConfig(
        input_path=args.dataset,
        output_path=args.output,
        model=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=args.system_prompt,
        resume=not args.no_resume,
        prompt_template=args.prompt_template,
        split_id=args.split_id,
        split_size=args.split_size,
    )


if __name__ == "__main__":
    cfg = parse_args()
    reannotate(cfg)
