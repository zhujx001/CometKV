#!/usr/bin/env python3
"""Pair saved tail ON/OFF generations and reproduce cross-layer support mismatch.

The CUDA check uses balanced draws to remove Monte Carlo variance; it is a
counterexample for the current partition, not a model-quality measurement.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paired_quality():
    off_dirs = [ROOT / "results/aggregation_ablation",
                ROOT / "results/aggregation_ablation_summary"]
    on_dir = ROOT / "results/aggregation_ablation_tail"
    off, on, sources = {}, {}, {}
    for directories, tag, records in ((off_dirs, "pure_topk", off),
                                      ([on_dir], "default_tail", on)):
        for directory in directories:
            for path in sorted(directory.glob(f"predictions_{tag}_shard*.jsonl")):
                sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                for row in read_rows(path):
                    key = (row["id"], row["mode"])
                    assert key not in records, f"Duplicate prediction: {key}"
                    records[key] = row
    assert on, "No tail predictions found"
    original_inputs = {r["id"]: r for d in off_dirs for r in read_rows(d / "inputs.jsonl")}
    for row in read_rows(on_dir / "inputs.jsonl"):
        source = original_inputs[row["id"]]
        for field in ("input_ids", "answers", "max_new_tokens", "task"):
            assert row[field] == source[field], (row["id"], field)
    groups = {}
    for key, row in sorted(on.items()):
        baseline = off[key]
        for field in ("prompt_tokens", "max_new_tokens", "answers", "retrieval_capacity"):
            assert row[field] == baseline[field], (key, field)
        assert baseline["final_tail"] == 0 and row["final_tail"] > 0
        assert row["final_topk"] + row["final_tail"] <= row["retrieval_capacity"]
        groups.setdefault((row["mode"], row["task"]), []).append({
            "id": row["id"], "off": baseline["score"] * 100, "on": row["score"] * 100,
            "tail": row["final_tail"], "capacity": row["retrieval_capacity"],
            "on_prediction": row["prediction"], "off_prediction": baseline["prediction"],
        })
    summary = {}
    rng = np.random.default_rng(20250930)
    for (mode, task), rows in sorted(groups.items()):
        before, after = np.array([[r["off"], r["on"]] for r in rows]).T
        delta = after - before
        n = len(rows)
        ci = None
        if n > 1:
            draws = rng.choice(delta, (10000, n), replace=True).mean(axis=1)
            ci = np.quantile(draws, [0.025, 0.975]).tolist()
        summary.setdefault(mode, {})[task] = dict(
            n=n, off=float(before.mean()), on=float(after.mean()),
            delta=float(delta.mean()), paired_bootstrap_95ci=ci,
            wins=int((delta > 1e-10).sum()), losses=int((delta < -1e-10).sum()),
            ties=int((abs(delta) <= 1e-10).sum()), rows=rows,
        )
    return dict(sources_sha256=sources, matched_predictions=len(on),
                unique_questions=len({k[0] for k in on}), results=summary)


def support_counterexample(device):
    import torch

    sys.path.insert(0, str(ROOT / "library/cometkv"))
    from cometkv import sampled_tail_attention_merge

    dim, group, m = 128, 4, 96
    # Four candidates with equal exact logits and scalar values [8, 4, 0, 0].
    # Owner head = {0}, next-layer head = {1}. Reuse samples from {1, 2, 3}.
    q = torch.zeros((1, group, dim), dtype=torch.bfloat16, device=device)
    keys = torch.zeros((1, m, dim), dtype=q.dtype, device=device)
    all_values = torch.zeros((4, dim), dtype=q.dtype, device=device)
    all_values[:, 0] = torch.tensor([8, 4, 0, 0], dtype=q.dtype, device=device)
    draws = torch.tensor([1, 2, 3], device=device).repeat(m // 3)
    corr = torch.full((1, m), -np.log(m / 3), dtype=torch.float32, device=device)
    lse = torch.zeros((1, group), dtype=torch.float32, device=device)

    def merged(head, ids, correction, clip):
        output = all_values[head].expand(1, group, dim).contiguous().clone()
        sampled_tail_attention_merge(q, keys, all_values[ids].unsqueeze(0).contiguous(),
                                     correction, output, lse, dim ** -0.5, clip)
        return float(output[0, 0, 0].item())

    owner = merged(0, draws, corr, 0.0)
    reused = merged(1, draws, corr, 0.0)
    reused_clipped = merged(1, draws, corr, 4.0)
    # Full-support proposal, with current head contributing zero to the tail.
    # Clip is disabled: production clipping needs explicit mask handling before
    # this design can be used, since a masked -inf must not enter its logit mean.
    full_draws = torch.arange(4, device=device).repeat(m // 4)
    full_corr = torch.full((1, m), -np.log(m / 4), dtype=torch.float32, device=device)
    full_corr[:, full_draws == 1] = -float("inf")
    corrected = merged(1, full_draws, full_corr, 0.0)
    assert owner == 3.0 and reused == 2.0 and reused_clipped == 2.0 and corrected == 3.0
    return dict(gpu=torch.cuda.get_device_name(device), torch=torch.__version__,
                dtype="bfloat16", m=m, exact_output=3.0, owner_layer_output=owner,
                reused_layer_output=reused, reused_layer_output_clip4=reused_clipped,
                full_support_masked_output_clip0=corrected,
                missing_candidate=0, double_counted_candidate=1,
                note="Balanced draws remove sampling noise; this is not a task evaluation.")


def summarize_latency(directory):
    groups = {}
    for name in ("off_1", "on_1", "on_2", "off_2"):
        path = directory / name / "metrics.jsonl"
        records = read_rows(path)
        assert len(records) == 2, path
        for row in records:
            assert row["status"] == "ok", row
            assert row["stats_mode"] == "block" and row["query_aggregation"] == "q_sum"
            assert row["retrieval_budget"] == 0.02 and row["batch_size"] == 1
            assert row["generated_tokens_per_sequence"] == 384
            assert row["timed_decode_steps"] == 382 and row["ignored_decode_steps"] == 1
            groups.setdefault(row["length_label"], {}).setdefault(name.split("_")[0], []).append(row["tpot_ms"])
    summary = {}
    for length, modes in groups.items():
        assert len(modes["off"]) == len(modes["on"]) == 2
        off, on = float(np.mean(modes["off"])), float(np.mean(modes["on"]))
        summary[length] = dict(raw_tpot_ms=modes, off_ms=off, on_ms=on,
                               delta_ms=on-off, overhead_percent=(on/off-1)*100)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/tail_analysis")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-cuda", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    quality = paired_quality()
    (args.output / "paired_quality.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n")
    for mode, tasks in quality["results"].items():
        for task, row in tasks.items():
            print(mode, task, json.dumps({k: v for k, v in row.items() if k != "rows"}))
    if not args.skip_cuda:
        support = support_counterexample(args.device)
        (args.output / "support_counterexample.json").write_text(json.dumps(support, indent=2) + "\n")
        print("support_counterexample", json.dumps(support))
    if (args.output / "latency").is_dir():
        latency = summarize_latency(args.output / "latency")
        (args.output / "latency_summary.json").write_text(json.dumps(latency, indent=2) + "\n")
        print("latency", json.dumps(latency))


if __name__ == "__main__":
    main()
