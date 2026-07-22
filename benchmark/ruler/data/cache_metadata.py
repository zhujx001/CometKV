import argparse
import json
from pathlib import Path


METADATA_FILE = "metadata.json"


def build_cache_metadata(
    benchmark,
    task,
    subset,
    num_samples,
    max_seq_length,
    tokenizer_path,
    tokenizer_type,
    model_template_type,
):
    return {
        "benchmark": benchmark,
        "task": task,
        "subset": subset,
        "num_samples": num_samples,
        "max_seq_length": max_seq_length,
        "tokenizer_path": tokenizer_path,
        "tokenizer_type": tokenizer_type,
        "model_template_type": model_template_type,
    }


def load_cache_metadata(task_dir):
    metadata_path = Path(task_dir) / METADATA_FILE
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return None

    if not isinstance(metadata, dict):
        return None
    return metadata


def cache_is_valid(task_dir, expected, subset="validation"):
    task_dir = Path(task_dir)
    data_file = task_dir / f"{subset}.jsonl"
    if not data_file.is_file():
        return False

    metadata = load_cache_metadata(task_dir)
    if metadata is None:
        return False

    return metadata == expected


def write_cache_metadata(task_dir, metadata):
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = task_dir / METADATA_FILE
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--subset", type=str, default="validation")
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--max_seq_length", type=int, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--tokenizer_type", type=str, required=True)
    parser.add_argument("--model_template_type", type=str, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    expected = build_cache_metadata(
        benchmark=args.benchmark,
        task=args.task,
        subset=args.subset,
        num_samples=args.num_samples,
        max_seq_length=args.max_seq_length,
        tokenizer_path=args.tokenizer_path,
        tokenizer_type=args.tokenizer_type,
        model_template_type=args.model_template_type,
    )

    if cache_is_valid(args.task_dir, expected, subset=args.subset):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
