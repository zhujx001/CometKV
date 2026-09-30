#!/usr/bin/env python3
"""Time projection, signature scoring and top-k on synthetic Llama-3.1-8B shapes.

CUDA graph replay excludes initialization, eviction, KV gather and attention.
Use the FWE model benchmark separately for end-to-end decode measurements.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "library" / "cometkv")]

from cache_hub.cometkv_cache import cometkv_cache


@torch.inference_mode()
def measure(prompt, stats_mode, aggregation, repeats, iterations):
    torch.manual_seed(1234)
    device = "cuda:0"
    group, kv_heads, dim = 4, 8, 128
    topk = max(16, int(prompt * 0.02))
    cache = cometkv_cache(
        valid_start=np.zeros(1, dtype=np.int32), layer_num=1, batch_size=1,
        max_length=prompt + 512, max_new_length=512, num_key_value_heads=kv_heads,
        num_heads=kv_heads * group, head_dim=dim, dtype=torch.bfloat16,
        layer_mapping={"0": device}, static_pattern_start=4, static_pattern_end=32,
        retrieval_budget=0.02, sig_topk=topk, sig_min_retrieval_topk=16,
        sig_bits=128, sig_chunk_size=131072, sig_seed=1234, sig_mode="random_orth",
        sig_token_cache_size=16, prefill_bsz=1, num_gpus=1, model_size=8,
        stats_mode=stats_mode, query_aggregation=aggregation, sample_frac=0,
    )
    keys = torch.randn(1, prompt, kv_heads, dim, device=device, dtype=torch.bfloat16)
    queries = torch.randn(1, 1, kv_heads * group, dim, device=device, dtype=torch.bfloat16)
    cache.prefill_update_kv_cache(queries, keys, keys, 0, 0)
    cache.prepare_cache()
    for start, count in ((0, 96), (96, 128)):
        cache.decode_hot_keys[0][:, :, :count].normal_(mean=0.25)
        cache.decode_hot_values[0][:, :, :count].normal_()
        cache._update_lockstep_evicted_retrieval(0, start, count)
    cache._update_retrieval_plan(cache.prompt_lengths + 257)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            cache._grouped_query_asym_topk(queries, 0)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cache._grouped_query_asym_topk(queries, 0)
    for _ in range(50):
        graph.replay()
    torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        start_event, end_event = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start_event.record()
        for _ in range(iterations):
            graph.replay()
        end_event.record()
        end_event.synchronize()
        timings.append(start_event.elapsed_time(end_event) * 1000 / iterations)
    workspace = sum(t.numel() * t.element_size() for state in cache._grouped_score_buffers.values()
                    for t in state.values())
    result = dict(
        prompt_length=prompt, indexed_length=prompt + 224,
        group_size=group, kv_heads=kv_heads, head_dim=dim, batch_size=1,
        stats_mode=stats_mode, query_aggregation=aggregation, topk=cache.active_sparse_len_host,
        median_us=statistics.median(timings), samples_us=timings,
        additional_grouped_workspace_bytes=workspace,
    )
    del graph, cache, keys, queries
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768, 65536])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path, default=ROOT / "results/block_gqa_validation/selector.json")
    args = parser.parse_args()
    for name in ("COMETKV_STATS_MODE", "COMETKV_QUERY_AGG", "COMETKV_SAMPLE_FRAC"):
        if name in os.environ:
            raise ValueError(f"Unset {name}: this benchmark controls that setting explicitly")
    results = []
    for length in args.lengths:
        for stats, aggregation in (("frozen", "q_sum"), ("block", "q_sum"), ("block", "mean_prob")):
            result = measure(length, stats, aggregation, args.repeats, args.iterations)
            results.append(result)
            print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        protocol="CUDA graph selector only; 224 sealed generated tokens; fixed top-k; random Q/K",
        results=results,
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
