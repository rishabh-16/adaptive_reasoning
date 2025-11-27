from vllm import LLM
from data_loader import load_data
from openai import OpenAI
from parser import *
from transformers import AutoTokenizer
import argparse



def main():
    available_gpus = [4]
    draft_client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8030/v1",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
    examples = load_data("aime24", "test")
    example = examples[0]
    example["question"] = parse_question(example, "aime24")
    gt_cot, gt_ans = parse_ground_truth(example, "aime24")
    # prompt = construct_prompt(example, "qwen3-4b-thinking-2507")
    # prompt = tokenizer.apply_chat_template(example["question"], tokenize=False)
    # print(prompt)
    # exit()
    prompt = "Given the following question, give 8 different plans to solve it: " + example["question"] + "\n\n" + "The format of the plans should be: \n\n" + "Plan 1: \n\n" + "Plan 2: \n\n" + "Plan 3: \n\n" + "Plan 4: \n\n" + "Plan 5: \n\n" + "Plan 6: \n\n" + "Plan 7: \n\n" + "Plan 8: \n\n"
    response = draft_client.chat.completions.create(
        model="Qwen/Qwen3-4B-Thinking-2507",
        messages=[{"role": "user", "content": prompt}],
    )
    print(response.choices[0].message.content)
    
    # Save the response to a file
    output_file = "response_output.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(response.choices[0].message.content)
    print(f"Response saved to {output_file}")

if __name__ == "__main__":
    main()

