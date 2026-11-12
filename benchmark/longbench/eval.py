import os
import json
import argparse
import numpy as np
from pathlib import Path

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

SUPPORTED_MODELS = [
    "llama-3-8b-1048k",
    "qwen2.5-7b",
    "llama-3.1-8b",
    "qwen2.5-72b",
    "mistral-7b-instruct-v0.2",
]

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

sub_categories = {
    "Single-Document QA": ["qasper", "multifieldqa_en", "narrativeqa"],
    "Multi-Document QA": ["hotpotqa", "2wikimqa", "musique", "dureader"],
    "Summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "Few-shot learning": ["trec", "lsht", "samsum", "triviaqa"],
    "Synthetic tasks": ["passage_retrieval_en", "passage_count", "passage_retrieval_zh"],
    "Code Completion": ["repobench-p", "lcc"]
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, 
                        choices=SUPPORTED_MODELS)
    parser.add_argument("--attn_type", type=str, default="Full_Flash_Attn",
                        choices=["Full_Flash_Attn", "CometKV", "CometKV_GPU"],
                        help="Attention method")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--task', type=str, default=None, help="Only evaluate one dataset and update its score in result.json")
    parser.add_argument("--pred_dir", type=Path, default=None,
                        help="Directory containing {task}.jsonl prediction files.")
    parser.add_argument("--output_prefix", type=str, default="",
                        help="Optional prefix for result.json and summary.json.")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def collect_scores(path, use_longbench_e=False, task=None):
    scores = dict()
    all_files = sorted(os.listdir(path))
    print("Evaluating on:", all_files)
    target_filename = f"{task}.jsonl" if task else None
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        if target_filename is not None and filename != target_filename:
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('.')[0]
        file_path = os.path.join(path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        if use_longbench_e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    return scores


def merge_result_file(result_path, new_scores):
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as handle:
            merged_scores = json.load(handle)
    else:
        merged_scores = {}

    merged_scores.update(new_scores)

    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(merged_scores, handle, ensure_ascii=False, indent=4)

    return merged_scores


FIRST_LINE_DATASETS = {"trec", "triviaqa", "samsum", "lsht"}


def score_sample(dataset, prediction, answers, all_classes):
    if dataset in FIRST_LINE_DATASETS:
        prediction = prediction.lstrip("\n").split("\n")[0]
    metric_fn = dataset2metric[dataset]
    all_classes = all_classes or []
    return max(metric_fn(prediction, answer, all_classes=all_classes) for answer in answers)


def evaluate_single_file(filepath):
    dataset = filepath.stem
    if dataset not in dataset2metric:
        raise ValueError(f"Unknown dataset: {dataset} (from {filepath})")

    predictions, answers_list, lengths = [], [], []
    all_classes = None
    with filepath.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            predictions.append(record["pred"])
            answers_list.append(record.get("answers", []))
            all_classes = record.get("all_classes")
            if "length" in record:
                lengths.append(record["length"])

    scores = [
        score_sample(dataset, prediction, answers, all_classes)
        for prediction, answers in zip(predictions, answers_list)
    ]
    result = {
        "dataset": dataset,
        "num_samples": len(predictions),
        "score": round(100.0 * sum(scores) / len(scores), 2) if scores else 0.0,
    }

    if lengths:
        buckets = {"0-4k": [], "4-8k": [], "8k+": []}
        for score, length in zip(scores, lengths):
            if length < 4000:
                buckets["0-4k"].append(score)
            elif length < 8000:
                buckets["4-8k"].append(score)
            else:
                buckets["8k+"].append(score)
        result["by_length"] = {
            key: round(100.0 * np.mean(value), 2) if value else 0.0
            for key, value in buckets.items()
        }

    return result


def load_existing_result(result_path):
    if not result_path.exists():
        return {}
    try:
        with result_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def write_summary(summary_path, pred_dir, scores):
    flat_scores = {
        dataset: info["score"]
        for dataset, info in scores.items()
        if isinstance(info, dict) and "score" in info
    }
    dataset_rows = [
        {"dataset": dataset, **info}
        for dataset, info in scores.items()
        if isinstance(info, dict)
    ]
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "pred_dir": str(pred_dir),
                "num_datasets": len(dataset_rows),
                "global_average": round(sum(flat_scores.values()) / len(flat_scores), 2) if flat_scores else 0.0,
                "datasets": dataset_rows,
            },
            handle,
            ensure_ascii=False,
            indent=4,
        )


def evaluate_prediction_dir(pred_dir, task=None, output_prefix=""):
    pred_dir = Path(pred_dir).resolve()
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {pred_dir}")

    if task:
        jsonl_files = [pred_dir / f"{task}.jsonl"]
        if not jsonl_files[0].exists():
            raise FileNotFoundError(f"Prediction file not found: {jsonl_files[0]}")
    else:
        jsonl_files = sorted(pred_dir.glob("*.jsonl"))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in {pred_dir}")

    scored = {}
    for filepath in jsonl_files:
        info = evaluate_single_file(filepath)
        scored[info["dataset"]] = info
        print(f"  {info['dataset']:<25s}  samples={info['num_samples']:>4d}  score={info['score']:>7.2f}")

    prefix = f"{output_prefix}_" if output_prefix else ""
    result_path = pred_dir / f"{prefix}result.json"
    summary_path = pred_dir / f"{prefix}summary.json"
    if task:
        merged = load_existing_result(result_path)
        merged.update(scored)
    else:
        merged = scored

    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=4)
    write_summary(summary_path, pred_dir, merged)
    print(f"Wrote {result_path}")
    print(f"Wrote {summary_path}")


if __name__ == '__main__':
    args = parse_args()
    if args.pred_dir is not None:
        evaluate_prediction_dir(args.pred_dir, task=args.task, output_prefix=args.output_prefix)
    else:
        if args.model is None:
            raise ValueError("--model is required when --pred_dir is not supplied.")
        pred_root = "results/pred_e" if args.e else "results/pred"
        evaluate_prediction_dir(
            Path(pred_root) / args.model / args.attn_type,
            task=args.task,
            output_prefix=args.output_prefix,
        )
