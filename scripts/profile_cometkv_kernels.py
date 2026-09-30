#!/usr/bin/env python3
"""NCU NVTX capture and CUDA-event timings for CometKV decode kernels.

Synthetic BF16 Llama-3.1-8B shapes isolate kernel costs. Model TPOT must be
measured separately. NCU uses --nvtx --nvtx-include cometkv_profile/ --profile.
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "library/cometkv")]

from cache_hub.cometkv_cache import cometkv_cache
from cometkv import sampled_tail_attention_merge


def build_merge(args):
    rows, group, dim, m = 8, 4, 128, args.samples
    k = max(16, int(args.length * 0.02))
    query = torch.randn(rows, group, dim, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(rows, m, dim, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    output = torch.randn_like(query)
    lse = torch.full((rows, group), math.log(k), device="cuda")
    heads = torch.stack([torch.randperm(args.length, device="cuda")[:k] for _ in range(rows)]).int()
    draws = torch.randint(args.length, (rows, m), device="cuda", dtype=torch.int32)
    # Include substantial current-head hits, as a score-based proposal can do.
    hits = min(m // 4, k)
    draws[:, :hits].copy_(heads[:, :hits])
    corr = torch.full((rows, m), math.log(args.length / m), device="cuda")

    def step():
        sampled_tail_attention_merge(query, keys, values, corr, output, lse,
                                     dim ** -0.5, 4.0, draws, heads, k)

    return step, dict(head=k, layers=1, fixture_head_hit_fraction=0.25)


def build_cache_stage(args):
    layers = args.layers
    cache = cometkv_cache(
        valid_start=np.zeros(1, dtype=np.int32), layer_num=layers, batch_size=1,
        max_length=args.length + 512, max_new_length=512, num_key_value_heads=8,
        num_heads=32, head_dim=128, dtype=torch.bfloat16,
        layer_mapping={str(i): "cuda:0" for i in range(layers)},
        static_pattern_start=4, static_pattern_end=32, retrieval_budget=0.02,
        sig_bits=128, sig_topk=0, sig_min_retrieval_topk=16, sig_chunk_size=131072,
        sig_seed=1234, sig_mode="random_orth", sig_token_cache_size=1024,
        prefill_bsz=1, num_gpus=1, model_size=8, sample_size=args.samples,
        query_aggregation=args.aggregation, stats_mode="block",
    )
    queries = []
    for layer in range(layers):
        query = torch.randn(1, 1, 32, 128, device="cuda", dtype=torch.bfloat16)
        keys = torch.randn(1, args.length, 8, 128, device="cuda", dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        cache.prefill_update_kv_cache(query, keys, values, layer, 0)
        queries.append(query)
    cache.prepare_cache()
    cache.use_cuda_graph = True  # Fixed noise supplied outside captures, as in runtime.
    cache._cg_cache_seqlens.fill_(36 + cache.active_sparse_len_host)
    cache._grouped_query_asym_topk(queries[0], 0)
    outputs = [torch.zeros(8, 4, 128, device="cuda", dtype=torch.bfloat16) for _ in range(layers)]
    lse = torch.full((8, 4), math.log(36 + cache.active_sparse_len_host), device="cuda")

    def step():
        for layer, query in enumerate(queries):
            if args.stage == "selector":
                cache._grouped_query_asym_topk(query, layer)
            elif args.stage == "tail":
                cache._sampled_tail_attention(query, layer, outputs[layer], lse)
            else:
                cache.sparse_attention(query, layer)

    return step, dict(head=cache.active_sparse_len_host, layers=layers,
                      note="Fixed synthetic queries; warm head cache; tail includes proposal and 8-layer UVA prefetch")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["merge", "selector", "tail", "attention"], default="merge")
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--aggregation", choices=["q_sum", "mean_prob"], default="q_sum")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--profile", action="store_true", help="One eager invocation under an NCU NVTX range")
    parser.add_argument("--torch-profile", type=Path, help="Export one CUDA graph replay as a CUPTI timeline")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(1234)
    for name in ("COMETKV_SAMPLE_SIZE", "COMETKV_SAMPLE_FRAC", "COMETKV_QUERY_AGG", "COMETKV_STATS_MODE"):
        if name in os.environ:
            raise ValueError(f"Unset {name}; this script controls that setting explicitly")
    step, metadata = build_merge(args) if args.stage == "merge" else build_cache_stage(args)
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(10):
            step()
    torch.cuda.current_stream().wait_stream(warm)
    torch.cuda.synchronize()
    if args.profile:
        with torch.cuda.nvtx.range("cometkv_profile"):
            step()
            torch.cuda.synchronize()
        return
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(30):
        graph.replay()
    torch.cuda.synchronize()
    if args.torch_profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            graph.replay()
            torch.cuda.synchronize()
        args.torch_profile.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(args.torch_profile))
        print(profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    times = []
    for _ in range(args.repeats):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(args.iterations):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / args.iterations)
    result = dict(stage=args.stage, length=args.length, samples=args.samples,
                  aggregation=args.aggregation, gpu=torch.cuda.get_device_name(),
                  score_impl=os.environ.get("COMETKV_SCORE_IMPL", "auto"),
                  topk_impl=os.environ.get("COMETKV_TOPK_IMPL", "auto"),
                  median_us=statistics.median(times), timings_us=times, **metadata)
    print(json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    del step, graph
    gc.collect()


if __name__ == "__main__":
    main()
