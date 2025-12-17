import argparse
import glob
import json
import os
import sys
from collections import OrderedDict
from typing import Dict, Iterable, Iterator, List, Optional, Sequence


def load_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[Warning] Skipping malformed line in {path}.", file=sys.stderr)


def write_jsonl(path: str, rows: Iterable[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _with_split_suffix(path: str, split_id: int) -> str:
    dirname, basename = os.path.dirname(path), os.path.basename(path)
    name, ext = os.path.splitext(basename)
    return os.path.join(dirname, f"{name}-split{split_id}{ext}")


def discover_split_ids(base_path: str) -> List[int]:
    dirname, basename = os.path.dirname(base_path), os.path.basename(base_path)
    name, ext = os.path.splitext(basename)
    pattern = os.path.join(dirname, f"{name}-split*{ext}")
    split_paths = glob.glob(pattern)
    split_ids: List[int] = []
    for path in split_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        suffix = stem.split("-split")[-1]
        try:
            split_ids.append(int(suffix))
        except ValueError:
            print(
                f"[Warning] Unable to parse split id from file name {path}. Skipping.",
                file=sys.stderr,
            )
    return sorted(split_ids)


def parse_split_ids(split_arg: Optional[str]) -> Optional[List[int]]:
    if split_arg is None:
        return None
    split_ids: List[int] = []
    for chunk in split_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            start, end = int(start_str), int(end_str)
            if end < start:
                raise ValueError(f"Invalid range '{chunk}'. End must be >= start.")
            split_ids.extend(range(start, end + 1))
        else:
            split_ids.append(int(chunk))
    seen = set()
    deduped: List[int] = []
    for sid in split_ids:
        if sid not in seen:
            deduped.append(sid)
            seen.add(sid)
    return sorted(deduped)


def combine_splits(
    base_path: str,
    output_path: str,
    split_ids: Optional[Sequence[int]] = None,
    allow_missing: bool = False,
) -> Dict[str, int]:
    if split_ids is None:
        split_ids = discover_split_ids(base_path)
        if not split_ids:
            raise FileNotFoundError(
                "No split files found. Provide --split-ids or verify that the split files exist."
            )

    stats = {
        "splits_found": 0,
        "records_read": 0,
        "records_written": 0,
        "duplicates": 0,
        "missing_splits": 0,
    }

    by_index: "OrderedDict[int, Dict]" = OrderedDict()

    for split_id in split_ids:
        split_path = _with_split_suffix(base_path, split_id)
        if not os.path.exists(split_path):
            stats["missing_splits"] += 1
            message = f"[Warning] Split {split_id} not found at {split_path}."
            if not allow_missing:
                raise FileNotFoundError(message)
            print(message + " Skipping due to --allow-missing.", file=sys.stderr)
            continue

        stats["splits_found"] += 1
        for row in load_jsonl(split_path):
            stats["records_read"] += 1
            idx = row.get("_index")
            if idx is None:
                idx = len(by_index)
                print(
                    f"[Warning] Row without '_index' encountered in split {split_id}. Assigning sequential index {idx}.",
                    file=sys.stderr,
                )
            if idx in by_index:
                stats["duplicates"] += 1
                print(
                    f"[Warning] Duplicate index {idx} encountered. Overwriting previous entry.",
                    file=sys.stderr,
                )
            by_index[idx] = row

    sorted_items = sorted(by_index.items(), key=lambda item: item[0])
    sorted_rows = [row for _, row in sorted_items]
    stats["records_written"] = len(sorted_rows)

    if not sorted_rows:
        print("[Info] No records collected. Output will be empty.", file=sys.stderr)

    write_jsonl(output_path, sorted_rows)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine split JSONL outputs produced by experiments/04_datagen/main.py",
    )
    parser.add_argument(
        "--base-output",
        required=True,
        help="Base output path used in main.py (e.g. /path/to/reannotation.jsonl).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination JSONL path for the merged dataset. Defaults to --base-output.",
    )
    parser.add_argument(
        "--split-ids",
        default=None,
        help="Comma-separated list of split ids or ranges (e.g. '0-9,12,14'). If omitted, all discovered splits are used.",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=None,
        help="Optional number of splits (0-indexed). Equivalent to specifying --split-ids as range(0, num_splits).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing split files instead of treating them as an error.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output is None:
        output_path = args.base_output
    else:
        output_path = args.output

    split_ids: Optional[Sequence[int]] = None
    if args.num_splits is not None and args.split_ids is not None:
        raise ValueError("Use either --num-splits or --split-ids, not both.")
    if args.num_splits is not None:
        if args.num_splits < 0:
            raise ValueError("--num-splits must be non-negative.")
        split_ids = list(range(args.num_splits))
    elif args.split_ids is not None:
        split_ids = parse_split_ids(args.split_ids)

    stats = combine_splits(
        base_path=args.base_output,
        output_path=output_path,
        split_ids=split_ids,
        allow_missing=args.allow_missing,
    )

    print(
        json.dumps(
            {
                "output_path": output_path,
                **stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
