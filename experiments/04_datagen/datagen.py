import json
import datasets
def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line)['reannotation'])
            except json.JSONDecodeError:
                print(f"[Warning] Skipping malformed line in {path}.", file=sys.stderr)
    return data

def update_row_fn(data_reannotated):
    def update_row(example, idx):
        example['conversations'][1]['value'] = data_reannotated[idx]
        return example
    return update_row

def main():
    dataset = datasets.load_from_disk("/home/rishabhtiwari/datasets/openthoughts3_small")
    data_reannotated = load_jsonl('/checkpoint/transformer2/rishabhtiwari/reannotation_collated.jsonl')
    dataset['train'] = dataset['train'].map(update_row_fn(data_reannotated), with_indices=True)
    dataset.save_to_disk("/home/rishabhtiwari/datasets/openthoughts3_small_instruct")
if __name__ == "__main__":
    main()
