#!/usr/bin/env python3
"""Benchmark CometKV FWE latency and same-length batch throughput."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KERNEL_LIB = PROJECT_ROOT / "library" / "cometkv"

for candidate_path in (PROJECT_ROOT, KERNEL_LIB):
    if str(candidate_path) not in sys.path:
        sys.path.insert(0, str(candidate_path))

from config import add_config_args, generate_config, get_model_path
from model_hub import load_model, load_tokenizer


DEFAULT_DATA_PATH = PROJECT_ROOT / "test_data" / "fwe.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "speed_results"
CSV_FIELDS = [
    "timestamp",
    "status",
    "error_type",
    "error",
    "length_label",
    "target_context_len",
    "context_len",
    "original_context_len",
    "truncated",
    "batch_size",
    "max_new_length",
    "generated_tokens_per_sequence",
    "decode_steps_recorded",
    "ignored_decode_steps",
    "timed_decode_steps",
    "ttft_s",
    "prefill_s",
    "profile_decode_s",
    "end2end_s",
    "step_decode_s",
    "ignored_decode_s",
    "timed_decode_s",
    "raw_tpot_ms",
    "tpot_ms",
    "tokens_per_second",
    "peak_memory_gb",
    "model_path",
    "config_model_name",
    "data_path",
    "retrieval_budget",
    "sig_bits",
    "sig_chunk_size",
    "stats_mode",
    "query_aggregation",
    "head_capacity",
    "requested_head",
    "actual_head",
    "tail_samples",
    "retrieval_slots",
    "budget_plan_visible_length",
    "sample_size",
    "sample_frac",
    "sample_uniform_mix",
    "cometkv_token_cache_size",
    "cometkv_min_retrieval_topk",
    "static_pattern_start",
    "static_pattern_end",
    "exclude_preserved_from_budget",
]


def parse_token_lengths(value: str) -> list[int]:
    lengths = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        multiplier = 1
        if item.lower().endswith("k"):
            multiplier = 1024
            item = item[:-1]
        lengths.append(int(item) * multiplier)
    if not lengths:
        raise ValueError("At least one token length is required.")
    return lengths


def parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("At least one integer is required.")
    return items


def length_label(length: int) -> str:
    if length % 1024 == 0:
        return f"{length // 1024}k"
    return str(length)


def load_first_record(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        if Path(path).suffix == ".jsonl":
            first_line = handle.readline()
            if not first_line:
                raise ValueError(f"No records in {path}")
            return json.loads(first_line)
        data = json.load(handle)
    if isinstance(data, list):
        if not data:
            raise ValueError(f"No records in {path}")
        return data[0]
    if not isinstance(data, dict):
        raise TypeError(f"Unsupported data format in {path}: {type(data).__name__}")
    return data


def encode_and_truncate(tokenizer, prompt: str, target_context_len: int, device: str) -> dict[str, Any]:
    encoded = tokenizer([prompt], return_tensors="pt", padding=False)
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask
    original_context_len = int(input_ids.shape[1])
    if original_context_len < target_context_len:
        raise ValueError(
            f"Requested {target_context_len} tokens, but prompt only has "
            f"{original_context_len} tokens after tokenization."
        )
    input_ids = input_ids[:, :target_context_len].contiguous()
    attention_mask = attention_mask[:, :target_context_len].contiguous()
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "context_len": int(input_ids.shape[1]),
        "original_context_len": original_context_len,
        "truncated": original_context_len != int(input_ids.shape[1]),
    }


def make_same_length_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must be rank-2 tensors.")
    if input_ids.shape[0] != 1 or attention_mask.shape[0] != 1:
        raise ValueError("make_same_length_batch expects a single source sequence.")
    if input_ids.shape[1] != attention_mask.shape[1]:
        raise ValueError("input_ids and attention_mask must have the same sequence length.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    return input_ids.repeat(batch_size, 1).contiguous(), attention_mask.repeat(batch_size, 1).contiguous()


def compute_decode_metrics(
    ignored_decode_s: float,
    timed_decode_s: float,
    ignored_decode_steps: int,
    timed_decode_steps: int,
    batch_size: int,
) -> dict[str, float]:
    raw_decode_s = float(ignored_decode_s) + float(timed_decode_s)
    raw_steps = int(ignored_decode_steps) + int(timed_decode_steps)
    return {
        "raw_decode_s": raw_decode_s,
        "raw_tpot_ms": raw_decode_s * 1000.0 / raw_steps if raw_steps > 0 else 0.0,
        "tpot_ms": float(timed_decode_s) * 1000.0 / timed_decode_steps if timed_decode_steps > 0 else 0.0,
        "tokens_per_second": (
            float(batch_size) * timed_decode_steps / float(timed_decode_s)
            if timed_decode_steps > 0 and timed_decode_s > 0
            else 0.0
        ),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def build_cometkv_config(args: argparse.Namespace, context_len: int) -> dict[str, Any]:
    return generate_config(
        model_name=args.config_model_name,
        context_len=context_len,
        attn_type="CometKV",
        retrieval_budget=float(args.retrieval_budget),
        sig_bits=int(args.sig_bits),
        sig_seed=int(args.sig_seed),
        sig_chunk_size=int(args.sig_chunk_size),
        cometkv_min_retrieval_topk=int(args.cometkv_min_retrieval_topk),
        cometkv_token_cache_size=int(args.cometkv_token_cache_size),
        cometkv_static_pattern_start=args.cometkv_static_pattern_start,
        cometkv_static_pattern_end=args.cometkv_static_pattern_end,
        cometkv_exclude_preserved_from_budget=bool(args.cometkv_exclude_preserved_from_budget),
        cometkv_stats_mode=getattr(args, "cometkv_stats_mode", "block"),
        cometkv_query_aggregation=getattr(args, "cometkv_query_aggregation", "mean_prob"),
    )


def summarize_step_latencies(step_latencies_ms: list[float], ignore_first_steps: int, batch_size: int) -> dict[str, float]:
    ignored_steps = min(max(int(ignore_first_steps), 0), len(step_latencies_ms))
    timed_steps = max(len(step_latencies_ms) - ignored_steps, 0)
    ignored_decode_s = sum(step_latencies_ms[:ignored_steps]) / 1000.0
    timed_decode_s = sum(step_latencies_ms[ignored_steps:]) / 1000.0
    metrics = compute_decode_metrics(
        ignored_decode_s=ignored_decode_s,
        timed_decode_s=timed_decode_s,
        ignored_decode_steps=ignored_steps,
        timed_decode_steps=timed_steps,
        batch_size=batch_size,
    )
    metrics.update(
        {
            "ignored_decode_s": ignored_decode_s,
            "timed_decode_s": timed_decode_s,
            "ignored_decode_steps": ignored_steps,
            "timed_decode_steps": timed_steps,
            "step_decode_s": ignored_decode_s + timed_decode_s,
        }
    )
    return metrics


def decode_metrics_from_profile_stats(stats: dict[str, Any], batch_size: int) -> dict[str, float]:
    ignored_decode_s = float(stats.get("ignored_decode_s", 0.0))
    timed_decode_s = float(stats.get("timed_decode_s", 0.0))
    ignored_decode_steps = int(stats.get("ignored_decode_steps", 0))
    timed_decode_steps = int(stats.get("timed_decode_steps", 0))
    metrics = compute_decode_metrics(
        ignored_decode_s=ignored_decode_s,
        timed_decode_s=timed_decode_s,
        ignored_decode_steps=ignored_decode_steps,
        timed_decode_steps=timed_decode_steps,
        batch_size=batch_size,
    )
    metrics.update(
        {
            "ignored_decode_s": ignored_decode_s,
            "timed_decode_s": timed_decode_s,
            "ignored_decode_steps": ignored_decode_steps,
            "timed_decode_steps": timed_decode_steps,
            "step_decode_s": float(stats.get("raw_decode_s", metrics["raw_decode_s"])),
        }
    )
    return metrics


@contextlib.contextmanager
def maybe_silence(enabled: bool):
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def run_one(
    args: argparse.Namespace,
    tokenizer,
    prompt: str,
    target_context_len: int,
    batch_size: int,
) -> dict[str, Any]:
    timestamp = datetime.now().isoformat(timespec="seconds")
    base_result: dict[str, Any] = {
        "timestamp": timestamp,
        "status": "ok",
        "error_type": "",
        "error": "",
        "length_label": length_label(target_context_len),
        "target_context_len": int(target_context_len),
        "batch_size": int(batch_size),
        "max_new_length": int(args.max_new_length),
        "model_path": str(args.model_path),
        "config_model_name": str(args.config_model_name),
        "data_path": str(args.data_path),
        "retrieval_budget": float(args.retrieval_budget),
        "sig_bits": int(args.sig_bits),
        "sig_chunk_size": int(args.sig_chunk_size),
        "stats_mode": os.environ.get("COMETKV_STATS_MODE", getattr(args, "cometkv_stats_mode", "block")),
        "query_aggregation": os.environ.get("COMETKV_QUERY_AGG", getattr(args, "cometkv_query_aggregation", "mean_prob")),
        "cometkv_token_cache_size": int(args.cometkv_token_cache_size),
        "cometkv_min_retrieval_topk": int(args.cometkv_min_retrieval_topk),
        "static_pattern_start": args.cometkv_static_pattern_start,
        "static_pattern_end": args.cometkv_static_pattern_end,
        "exclude_preserved_from_budget": bool(args.cometkv_exclude_preserved_from_budget),
    }
    llm = None
    try:
        set_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(args.device)

        batch = encode_and_truncate(tokenizer, prompt, target_context_len, args.device)
        input_ids, attention_mask = make_same_length_batch(
            batch["input_ids"],
            batch["attention_mask"],
            batch_size,
        )
        with maybe_silence(args.quiet_model_output):
            attn_config = build_cometkv_config(args, batch["context_len"])
            llm = load_model(
                args.model_path,
                batch["context_len"] + int(args.max_new_length),
                dtype_from_name(args.dtype),
                args.device,
                tokenizer=tokenizer,
            )

        wall_start = time.time()
        with torch.no_grad(), maybe_silence(args.quiet_model_output):
            outputs = llm.generate(
                attention_type="CometKV",
                inputs_ids=input_ids,
                attention_masks=attention_mask,
                max_new_length=int(args.max_new_length),
                attn_config=attn_config,
                do_sample=False,
                ignore_eos=True,
                prefill_bsz=int(args.prefill_bsz),
                prefill_method=args.prefill_method,
                collect_step_latency=bool(args.per_step_latency),
                profile_timing=True,
                ignore_first_steps_for_tpot=int(args.ignore_first_steps),
                use_cuda_graph=bool(args.use_cuda_graph),
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize(args.device)
        wall_end = time.time()

        stats = dict(getattr(llm, "generate_latency_stats", {}))
        step_latencies_ms = list(getattr(llm, "decode_step_latencies_ms", []))
        if args.per_step_latency:
            decode_metrics = summarize_step_latencies(
                step_latencies_ms,
                ignore_first_steps=int(args.ignore_first_steps),
                batch_size=batch_size,
            )
        else:
            decode_metrics = decode_metrics_from_profile_stats(stats, batch_size=batch_size)
        generated_tokens = len(outputs[0]) if outputs else 0
        peak_gb = (
            torch.cuda.max_memory_allocated(args.device) / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        result = {
            **base_result,
            "context_len": int(batch["context_len"]),
            "original_context_len": int(batch["original_context_len"]),
            "truncated": bool(batch["truncated"]),
            "generated_tokens_per_sequence": int(generated_tokens),
            "decode_steps_recorded": int(len(step_latencies_ms)),
            "ttft_s": float(stats.get("ttft_s", 0.0)),
            "prefill_s": float(stats.get("prefill_s", 0.0)),
            "profile_decode_s": float(stats.get("decode_s", 0.0)),
            "end2end_s": float(stats.get("end2end_s", wall_end - wall_start)),
            "peak_memory_gb": float(peak_gb),
            "head_capacity": llm.kv_cache.selected_indices_buffer.size(1),
            "requested_head": llm.kv_cache.requested_sparse_len_host,
            "actual_head": llm.kv_cache.active_sparse_len_host,
            "tail_samples": llm.kv_cache.active_sample_len_host,
            "retrieval_slots": llm.kv_cache.active_retrieval_len_host,
            "budget_plan_visible_length": llm.kv_cache.budget_plan_visible_length_host,
            "sample_size": llm.kv_cache.sample_size,
            "sample_frac": llm.kv_cache.sample_frac,
            "sample_uniform_mix": llm.kv_cache.sample_uniform_mix,
            **decode_metrics,
        }
        return result
    except (torch.cuda.OutOfMemoryError, RuntimeError, ValueError) as exc:
        if torch.cuda.is_available() and not getattr(args, "fast_exit_on_error", False):
            torch.cuda.empty_cache()
        return {
            **base_result,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc).splitlines()[0],
            "traceback": traceback.format_exc(limit=8),
        }
    finally:
        del llm
        if torch.cuda.is_available() and not getattr(args, "fast_exit_on_error", False):
            torch.cuda.empty_cache()


def write_result(jsonl_handle, csv_writer: csv.DictWriter, result: dict[str, Any]) -> None:
    jsonl_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    jsonl_handle.flush()
    csv_writer.writerow({field: result.get(field, "") for field in CSV_FIELDS})


def bool_flag_args(name: str, enabled: bool, default_enabled: bool) -> list[str]:
    if enabled == default_enabled:
        return []
    return [f"--{name}" if enabled else f"--no-{name}"]


def build_worker_command(
    args: argparse.Namespace,
    target_context_len: int,
    batch_size: int,
    worker_jsonl: Path,
    worker_csv: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--fast_exit_on_error",
        "--data_path",
        str(args.data_path),
        "--model_path",
        str(args.model_path),
        "--config_model_name",
        str(args.config_model_name),
        "--device",
        str(args.device),
        "--dtype",
        str(args.dtype),
        "--lengths",
        str(target_context_len),
        "--batch_sizes",
        str(batch_size),
        "--max_new_length",
        str(args.max_new_length),
        "--ignore_first_steps",
        str(args.ignore_first_steps),
        "--prefill_bsz",
        str(args.prefill_bsz),
        "--prefill_method",
        str(args.prefill_method),
        "--seed",
        str(args.seed),
        "--output_dir",
        str(worker_jsonl.parent),
        "--jsonl_name",
        worker_jsonl.name,
        "--csv_name",
        worker_csv.name,
        "--retrieval_budget",
        str(args.retrieval_budget),
        "--sig_bits",
        str(args.sig_bits),
        "--sig_seed",
        str(args.sig_seed),
        "--sig_chunk_size",
        str(args.sig_chunk_size),
        "--cometkv_stats_mode",
        str(getattr(args, "cometkv_stats_mode", "block")),
        "--cometkv_query_aggregation",
        str(getattr(args, "cometkv_query_aggregation", "mean_prob")),
        "--cometkv_min_retrieval_topk",
        str(args.cometkv_min_retrieval_topk),
        "--cometkv_token_cache_size",
        str(args.cometkv_token_cache_size),
    ]
    if args.cometkv_static_pattern_start is not None:
        cmd.extend(["--cometkv_static_pattern_start", str(args.cometkv_static_pattern_start)])
    if args.cometkv_static_pattern_end is not None:
        cmd.extend(["--cometkv_static_pattern_end", str(args.cometkv_static_pattern_end)])
    # add_config_args defines this as a store_true/store_false flag PAIR, not a
    # BooleanOptionalAction, so bool_flag_args' "--no-..." form is unrecognized (worker
    # exit 2 whenever the paper protocol's --cometkv_include_preserved_in_budget is used).
    cmd.append(
        "--cometkv_exclude_preserved_from_budget"
        if args.cometkv_exclude_preserved_from_budget
        else "--cometkv_include_preserved_in_budget"
    )
    cmd.extend(bool_flag_args("quiet_model_output", bool(args.quiet_model_output), True))
    if args.per_step_latency:
        cmd.append("--per_step_latency")
    if args.use_cuda_graph:
        cmd.append("--use_cuda_graph")
    return cmd


def load_worker_result(worker_jsonl: Path) -> dict[str, Any]:
    with worker_jsonl.open("r", encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    if not lines:
        raise ValueError(f"Worker did not write a result: {worker_jsonl}")
    return json.loads(lines[-1])


def run_worker_subprocess(
    args: argparse.Namespace,
    target_context_len: int,
    batch_size: int,
    run_index: int,
) -> dict[str, Any]:
    worker_dir = args.output_dir / "_workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    # PID-unique scratch names: concurrent sweeps sharing an output_dir (e.g. one per GPU) must
    # not read each other's worker files — interleaved writes corrupt the JSONL (the "_fix" race).
    worker_base = (
        f"worker_{os.getpid()}_{run_index:03d}_{length_label(target_context_len)}_b{batch_size}"
    )
    worker_jsonl = worker_dir / f"{worker_base}.jsonl"
    worker_csv = worker_dir / f"{worker_base}.csv"
    cmd = build_worker_command(args, target_context_len, batch_size, worker_jsonl, worker_csv)
    env = os.environ.copy()
    env["COMETKV_EVENT_PROFILE"] = "0"
    env.setdefault("PYTHONPATH", f"{PROJECT_ROOT}:{KERNEL_LIB}")
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, text=True)
    if worker_jsonl.exists():
        try:
            return load_worker_result(worker_jsonl)
        except Exception:
            pass
    if completed.returncode != 0:
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "status": "error",
            "error_type": "WorkerProcessError",
            "error": f"worker exited with code {completed.returncode}",
            "length_label": length_label(target_context_len),
            "target_context_len": int(target_context_len),
            "batch_size": int(batch_size),
            "max_new_length": int(args.max_new_length),
            "model_path": str(args.model_path),
            "config_model_name": str(args.config_model_name),
            "data_path": str(args.data_path),
            "retrieval_budget": float(args.retrieval_budget),
            "sig_bits": int(args.sig_bits),
            "sig_chunk_size": int(args.sig_chunk_size),
            "stats_mode": os.environ.get("COMETKV_STATS_MODE", getattr(args, "cometkv_stats_mode", "block")),
            "query_aggregation": os.environ.get("COMETKV_QUERY_AGG", getattr(args, "cometkv_query_aggregation", "mean_prob")),
            "cometkv_token_cache_size": int(args.cometkv_token_cache_size),
            "cometkv_min_retrieval_topk": int(args.cometkv_min_retrieval_topk),
            "static_pattern_start": args.cometkv_static_pattern_start,
            "static_pattern_end": args.cometkv_static_pattern_end,
            "exclude_preserved_from_budget": bool(args.cometkv_exclude_preserved_from_budget),
        }
    return load_worker_result(worker_jsonl)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model_path", type=str, default=get_model_path("llama-3.1-8b"))
    parser.add_argument("--config_model_name", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--lengths", type=str, default="32k,64k,96k")
    parser.add_argument("--batch_sizes", type=str, default="1,2,4")
    parser.add_argument("--max_new_length", type=int, default=256)
    parser.add_argument("--ignore_first_steps", type=int, default=1)
    parser.add_argument("--prefill_bsz", type=int, default=1)
    parser.add_argument("--prefill_method", type=str, choices=["full", "xattn", "minfer"], default="full")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv_name", type=str, default=None)
    parser.add_argument("--jsonl_name", type=str, default=None)
    parser.add_argument(
        "--isolate_each_run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run each length/batch item in a fresh Python subprocess to release CUDA and pinned host caches.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fast_exit_on_error", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--quiet_model_output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--per_step_latency",
        action="store_true",
        help="Synchronize every decode step and compute TPOT from per-step latencies. Slower, for diagnosis.",
    )
    parser.add_argument(
        "--use_cuda_graph",
        action="store_true",
        help="Decode via CUDA-graph capture/replay (paper steady-state TPOT protocol).",
    )
    parser = add_config_args(parser)
    parser.set_defaults(
        attn_type="CometKV",
        retrieval_budget=0.02,
        sig_seed=1234,
        sig_chunk_size=131072,
        cometkv_token_cache_size=1024,
        cometkv_min_retrieval_topk=16,
    )
    args = parser.parse_args(argv)
    args.length_values = parse_token_lengths(args.lengths)
    args.batch_size_values = parse_int_list(args.batch_sizes)
    return args


def main(argv: list[str] | None = None) -> int:
    os.environ["COMETKV_EVENT_PROFILE"] = "0"
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / (args.csv_name or f"fwe_cometkv_speed_{stamp}.csv")
    jsonl_path = args.output_dir / (args.jsonl_name or f"fwe_cometkv_speed_{stamp}.jsonl")

    prompt = None
    tokenizer = None
    if args.worker or not args.isolate_each_run:
        sample = load_first_record(args.data_path)
        prompt = sample.get("input")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{args.data_path} does not contain a non-empty 'input' field.")
        tokenizer = load_tokenizer(args.model_path)
    total_runs = len(args.length_values) * len(args.batch_size_values)
    run_index = 0
    with jsonl_path.open("w", encoding="utf-8") as jsonl_handle, csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_handle:
        csv_writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()
        for target_context_len in args.length_values:
            for batch_size in args.batch_size_values:
                run_index += 1
                print(
                    f"[{run_index}/{total_runs}] length={length_label(target_context_len)} "
                    f"batch={batch_size}",
                    flush=True,
                )
                if args.isolate_each_run and not args.worker:
                    result = run_worker_subprocess(args, target_context_len, batch_size, run_index)
                else:
                    assert tokenizer is not None and prompt is not None
                    result = run_one(args, tokenizer, prompt, target_context_len, batch_size)
                write_result(jsonl_handle, csv_writer, result)
                csv_handle.flush()
                if result["status"] == "ok":
                    print(
                        f"  TTFT={result['ttft_s']:.3f}s "
                        f"TPOT={result['tpot_ms']:.2f}ms "
                        f"End2End={result['end2end_s']:.3f}s "
                        f"Tokens/s={result['tokens_per_second']:.2f} "
                        f"peak={result['peak_memory_gb']:.2f}GB",
                        flush=True,
                    )
                else:
                    print(
                        f"  ERROR {result['error_type']}: {result['error']}",
                        flush=True,
                    )
                    if args.worker and args.fast_exit_on_error:
                        os._exit(0)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSONL: {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
