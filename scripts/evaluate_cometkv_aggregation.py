#!/usr/bin/env python3
"""Paired task-quality ablation for block q_sum / mean_prob and full attention.

Prepare a deterministic subset once, then run shards on separate GPUs. The default
sample_frac=0 isolates top-k; --sample-size enables an independent tail quota. Scores
use the repository's LongBench metrics and GSM8K numeric exact match. This is a
small diagnostic subset, not a complete benchmark evaluation.
"""

import argparse
import contextlib
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "library/cometkv")]

from benchmark.longbench.metrics import qa_f1_score, retrieval_score, rouge_score
from benchmark.longbench.pred import build_chat, prepare_prompt, dataset2prompt, dataset2maxlen
from config import generate_config
from model_hub import load_model, load_tokenizer


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare(args):
    tokenizer = load_tokenizer(args.model)
    records = []
    sources = {}
    for task in args.tasks:
        path = args.longbench_root / f"{task}.jsonl"
        rows = read_rows(path)
        indices = sorted(random.Random(args.seed).sample(range(len(rows)), min(args.samples, len(rows))))
        sources[task] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "indices": indices}
        for index in indices:
            row = rows[index]
            cap = dataset2maxlen[task]
            prompt = prepare_prompt(tokenizer, dataset2prompt[task], row, task, args.model,
                                    args.max_length, cap)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            assert len(ids) + cap <= args.max_length
            records.append(dict(id=f"{task}:{index}", task=task, source_index=index,
                                source_id=row.get("_id"), input_ids=ids, max_new_tokens=cap,
                                answers=row["answers"], original_length=row.get("length")))
    if args.gsm_samples:
        path = args.output / "gsm8k_test.jsonl"
        rows = read_rows(path)
        indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.gsm_samples))
        sources["gsm8k"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "indices": indices}
        for index in indices:
            row = rows[index]
            prompt = build_chat(tokenizer, row["question"] +
                                "\nSolve step by step. End your response with #### followed by the final numeric answer.",
                                args.model)
            records.append(dict(id=f"gsm8k:{index}", task="gsm8k", source_index=index,
                                input_ids=tokenizer.encode(prompt, add_special_tokens=False),
                                max_new_tokens=1024, answers=[row["answer"].split("####")[-1].strip()] ))
    # Interleave tasks so each GPU shard contains all tasks.
    records.sort(key=lambda row: (row["source_index"], row["task"]))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "inputs.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = dict(model=args.model, subset_seed=args.seed, max_length=args.max_length,
                    sources=sources, tasks={})
    for task in sorted({r["task"] for r in records}):
        lengths = [len(r["input_ids"]) for r in records if r["task"] == task]
        metadata["tasks"][task] = dict(n=len(lengths), min_prompt=min(lengths), max_prompt=max(lengths),
                                     mean_prompt=statistics.mean(lengths))
    (args.output / "inputs_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["tasks"], indent=2), flush=True)


def score(task, prediction, answers):
    if task == "gsm8k":
        marked = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", prediction)
        numeric = marked or re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", prediction)
        extracted = numeric[-1].replace(",", "") if numeric else ""
        try:
            value = float(extracted)
            correct = value == float(answers[0].replace(",", ""))
        except ValueError:
            correct = False
        return float(correct), extracted
    metric = (retrieval_score if task == "passage_retrieval_en" else
              rouge_score if task in ("gov_report", "qmsum", "multi_news") else qa_f1_score)
    return max(metric(prediction, answer) for answer in answers), None


def release_cache(llm):
    torch.cuda.synchronize()
    for name in list(vars(llm)):
        if name.startswith("_cg_"):
            delattr(llm, name)
    llm.use_cuda_graph = False
    llm.kv_cache = None
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def run(args):
    controlled = ("COMETKV_STATS_MODE", "COMETKV_QUERY_AGG", "COMETKV_SAMPLE_FRAC",
                  "COMETKV_SAMPLE_AUTOSCALE", "COMETKV_MEAN_UPDATE_ALPHA",
                  "COMETKV_FULL_RECOMPUTE_INTERVAL", "COMETKV_NORM_MARGIN", "COMETKV_NO_KEY_CENTER")
    for name in controlled:
        if name in os.environ:
            raise ValueError(f"Unset {name}: use this script's explicit protocol")
    rows = read_rows(args.output / "inputs.jsonl")
    if args.only_tasks:
        rows = [row for row in rows if row["task"] in args.only_tasks]
    if args.limit:
        rows = rows[:args.limit]
    rows = rows[args.shard::args.shards]
    tokenizer = load_tokenizer(args.model)
    max_length = max(len(row["input_ids"]) + row["max_new_tokens"] for row in rows)
    with contextlib.redirect_stdout(io.StringIO()):
        llm = load_model(args.model, max_length, torch.bfloat16, "cuda:0", tokenizer)
    eos = set(llm.config.eos_token_id if isinstance(llm.config.eos_token_id, list) else [llm.config.eos_token_id])
    eos.update(tokenizer.convert_tokens_to_ids(token) for token in ("<|eot_id|>", "<|end_of_text|>"))
    output = args.output / f"predictions_{args.tag}_shard{args.shard}.jsonl"
    done = {(row["id"], row["mode"]) for row in read_rows(output)} if output.exists() else set()
    protocol = dict(model=args.model, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    stats_mode="block", sample_frac=args.sample_frac, sample_size=args.sample_size,
                    retrieval_budget=0.02, head_capacity="planned_generation",
                    tail_budget="additional", tail_support="full_candidates_mask_current_head",
                    signature_seed=1234, sink=4, recent=32, min_retrieval_topk=16,
                    include_preserved_in_budget=False, greedy=True, eos=sorted(eos),
                    cuda_graph_sparse=True, cuda_graph_full=False)
    protocol_path = args.output / f"protocol_{args.tag}_shard{args.shard}.json"
    if done and (not protocol_path.exists() or json.loads(protocol_path.read_text()) != protocol):
        raise ValueError("Existing predictions use a different protocol; choose a new --tag/output.")
    protocol_path.write_text(json.dumps(protocol, indent=2))
    with output.open("a") as handle:
        for number, row in enumerate(rows):
            modes = args.modes[number % len(args.modes):] + args.modes[:number % len(args.modes)]
            for mode in modes:
                if (row["id"], mode) in done:
                    continue
                release_cache(llm)
                torch.manual_seed(2025)
                np.random.seed(2025)
                random.seed(2025)
                ids = torch.tensor([row["input_ids"]], device="cuda:0", dtype=torch.int64)
                cap = row["max_new_tokens"]
                llm.max_length = ids.size(1) + cap
                backend = "Full_Flash_Attn" if mode == "full" else "CometKV"
                with contextlib.redirect_stdout(io.StringIO()):
                    config = generate_config(
                        args.model, ids.size(1), backend, retrieval_budget=0.02,
                        sig_seed=1234, sig_chunk_size=131072,
                        cometkv_stats_mode="block", cometkv_query_aggregation="q_sum" if mode == "full" else mode,
                        cometkv_static_pattern_start=4, cometkv_static_pattern_end=32,
                        cometkv_min_retrieval_topk=16, cometkv_exclude_preserved_from_budget=True,
                    )
                    if backend == "CometKV":
                        config[backend]["sample_size"] = args.sample_size
                        config[backend]["sample_frac"] = args.sample_frac
                    started = time.perf_counter()
                    generated = llm.generate(
                        backend, ids, torch.ones_like(ids), cap, config, do_sample=False,
                        ignore_eos=False, eos_token_ids=sorted(eos), use_cuda_graph=backend == "CometKV",
                        profile_timing=True,
                    )[0]
                    elapsed = time.perf_counter() - started
                prediction = tokenizer.decode(generated, skip_special_tokens=True)
                metric, extracted = score(row["task"], prediction, row["answers"])
                cache = llm.kv_cache
                result = dict(id=row["id"], task=row["task"], mode=mode, score=metric,
                              answers=row["answers"], prediction=prediction, extracted=extracted,
                              prompt_tokens=ids.size(1), generated_tokens=len(generated),
                              hit_cap=len(generated) == cap and generated[-1] not in eos,
                              max_new_tokens=cap, wall_s=elapsed,
                              latency=llm.generate_latency_stats, token_ids=generated,
                              retrieval_capacity=(cache.selected_indices_buffer.size(1) if backend == "CometKV" else None),
                              final_topk=getattr(cache, "active_sparse_len_host", None),
                              final_tail=getattr(cache, "active_sample_len_host", None),
                              requested_head=getattr(cache, "requested_sparse_len_host", None),
                              retrieval_slots=getattr(cache, "active_retrieval_len_host", None),
                              sealed_blocks=(cache.sig_active_blocks[0] if backend == "CometKV" else None))
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"{number+1}/{len(rows)} {row['id']} {mode}: score={metric:.3f} "
                      f"tokens={len(generated)} cap={result['hit_cap']} seconds={elapsed:.2f}", flush=True)
    release_cache(llm)


def summarize(args):
    rows = []
    for path in sorted(args.output.glob(f"predictions_{args.tag}_shard*.jsonl")):
        rows.extend(read_rows(path))
    if len({(row["id"], row["mode"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate question/mode outputs: use a new tag when changing the shard layout")
    summary = {"tag": args.tag, "tasks": {}}
    for task in sorted({row["task"] for row in rows}):
        subset = [row for row in rows if row["task"] == task]
        entry = {}
        for mode in sorted({row["mode"] for row in subset}):
            group = [row for row in subset if row["mode"] == mode]
            entry[mode] = dict(n=len(group), score=100 * statistics.mean(row["score"] for row in group),
                               mean_generated_tokens=statistics.mean(row["generated_tokens"] for row in group),
                               capped=sum(row["hit_cap"] for row in group),
                               generations_over_128=sum(row["generated_tokens"] > 128 for row in group))
        left = {row["id"]: row for row in subset if row["mode"] == "q_sum"}
        right = {row["id"]: row for row in subset if row["mode"] == "mean_prob"}
        common = sorted(left.keys() & right.keys())
        if common:
            differences = np.array([right[key]["score"] - left[key]["score"] for key in common])
            rng = np.random.default_rng(2025)
            bootstrap = rng.choice(differences, (10000, len(common)), replace=True).mean(1)
            entry["paired_mean_prob_minus_q_sum"] = dict(
                n=len(common), delta_points=100 * float(differences.mean()),
                bootstrap_95_ci_points=((100 * np.quantile(bootstrap, [.025, .975])).tolist()
                                        if len(common) > 1 else None),
                wins=int((differences > 1e-8).sum()), losses=int((differences < -1e-8).sum()),
                ties=int((abs(differences) <= 1e-8).sum()),
                identical_tokens=sum(left[key]["token_ids"] == right[key]["token_ids"] for key in common),
                capacity_mismatches=sum(left[key]["retrieval_capacity"] != right[key]["retrieval_capacity"] for key in common),
            )
        summary["tasks"][task] = entry
    (args.output / f"summary_{args.tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "summarize"])
    parser.add_argument("--model", default="/data/zjx/data-old/model/Llama-3.1-8B-Instruct")
    parser.add_argument("--output", type=Path, default=ROOT / "results/aggregation_ablation")
    parser.add_argument("--longbench-root", type=Path, default=Path("/data/zjx/data/longbench/longbench-jsonl"))
    parser.add_argument("--tasks", nargs="+", default=["hotpotqa", "multifieldqa_en", "passage_retrieval_en"])
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--gsm-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20250929)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--modes", nargs="+", choices=["q_sum", "mean_prob", "full"], default=["q_sum", "mean_prob", "full"])
    parser.add_argument("--sample-frac", type=float, default=0.0)
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Additional tail draws per layer/KV head; overrides --sample-frac.")
    parser.add_argument("--tag", default="pure_topk")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--only-tasks", nargs="+")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(8)
    os.environ["COMETKV_EVENT_PROFILE"] = "0"
    {"prepare": prepare, "run": run, "summarize": summarize}[args.action](args)


if __name__ == "__main__":
    main()
