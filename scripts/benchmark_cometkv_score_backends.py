#!/usr/bin/env python3
"""Compare packed-index scalar/LUT CUDA and experimental Tensor Core scoring.

Tensor Core variants unpack only a tile; no expanded index is stored in HBM.
This synthetic benchmark includes block compensation and GQA normalization.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "library/cometkv")]
from cometkv import grouped_signature_score_into


@triton.jit
def tensor_score(Q, QT, K, LO, STEP, BIAS, LOGITS, CHUNK_LSE,
                 N: tl.constexpr, G: tl.constexpr, BLOCKS: tl.constexpr,
                 START: tl.constexpr, END: tl.constexpr, PROMPT: tl.constexpr,
                 RS: tl.constexpr, BS: tl.constexpr, NORMALIZE: tl.constexpr,
                 BF16: tl.constexpr, BN: tl.constexpr, CHUNKS: tl.constexpr):
    row, chunk = tl.program_id(0), tl.program_id(1)
    token = END - 1 - (chunk * BN + tl.arange(0, BN))
    valid = token >= START
    bit = tl.arange(0, 128)
    head = tl.arange(0, 16)
    packed = tl.load(K + (row * N + token[:, None]) * 16 + bit[None, :] // 8,
                     valid[:, None] & (bit[None, :] < 120), other=0)
    binary = ((packed >> (bit[None, :] % 8)) & 1).to(tl.float32)
    query = tl.load(Q + (row * G + head[None, :]) * 128 + bit[:, None],
                    (head[None, :] < G) & (bit[:, None] < 120), other=0)
    if BF16:
        dot = tl.dot(binary.to(tl.bfloat16), query.to(tl.bfloat16))
    else:
        dot = tl.dot(binary, query, input_precision="tf32x3")
    epoch = tl.where(token < PROMPT, 0, 1 + (token - PROMPT + 32) // 128)
    lo = tl.load(LO + row * BLOCKS + epoch, valid, other=0)
    step = tl.load(STEP + row * BLOCKS + epoch, valid, other=0)
    code = tl.load(K + (row * N + token) * 16 + 15, valid, other=0).to(tl.float32)
    norm = tl.exp(lo + code * step)
    total = tl.load(QT + row * G + head, head < G, other=0)
    bias = tl.load(BIAS + (row * G + head[None, :]) * BLOCKS + epoch[:, None],
                   valid[:, None] & (head[None, :] < G), other=0)
    value = (2.0 * dot - total[None, :]) * norm[:, None] * RS + bias * BS
    value = tl.where(valid[:, None], value, -float("inf"))
    tl.store(LOGITS + (row * G + head[None, :]) * N + token[:, None],
             value, valid[:, None] & (head[None, :] < G))
    if NORMALIZE:
        maximum = tl.max(value, 0)
        z = tl.sum(tl.exp(value - maximum[None, :]), 0)
        tl.store(CHUNK_LSE + (row * G + head) * CHUNKS + chunk,
                 maximum + tl.log(z), head < G)


@triton.jit
def tensor_lse(CHUNK_LSE, LSE, CHUNKS: tl.constexpr, BLOCK: tl.constexpr):
    row_head = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(CHUNK_LSE + row_head * CHUNKS + c, c < CHUNKS, other=-float("inf"))
    mx = tl.max(x, 0)
    tl.store(LSE + row_head, mx + tl.log(tl.sum(tl.exp(x - mx), 0)))


@triton.jit
def tensor_merge(LOGITS, LSE, SCORES, N: tl.constexpr, G: tl.constexpr,
                 START: tl.constexpr, END: tl.constexpr, GH: tl.constexpr, BN: tl.constexpr):
    row, chunk = tl.program_id(0), tl.program_id(1)
    token = START + chunk * BN + tl.arange(0, BN)
    h = tl.arange(0, GH)
    z = tl.load(LSE + row * G + h, h < G, other=0)
    x = tl.load(LOGITS + (row * G + h[:, None]) * N + token[None, :],
                (h[:, None] < G) & (token[None, :] < END), other=-float("inf")) - z[:, None]
    mx = tl.max(x, 0)
    mx = tl.where(token < END, mx, 0)
    score = mx + tl.log(tl.sum(tl.exp(x - mx[None, :]), 0)) - math.log(G)
    tl.store(SCORES + row * N + token, score, token < END)


def graph_timing(step):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(30):
        graph.replay()
    timings = []
    for _ in range(7):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(200):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / 200)
    return statistics.median(timings), timings


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--group", type=int, choices=[1, 4], default=1)
    parser.add_argument("--backend", choices=["all", "scalar", "lookup", "tc_bf16", "tc_tf32x3"], default="all")
    parser.add_argument("--tile", type=int, choices=[32, 64, 128, 256], default=32)
    parser.add_argument("--warps", type=int, choices=[4, 8], default=4)
    parser.add_argument("--stages", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.profile and args.backend == "all":
        parser.error("--profile requires one --backend")
    torch.manual_seed(1234)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, n, g, blocks = 8, args.length, args.group, 3
    start, end, prompt = 4, n - 32, n - 256
    q = torch.randn(rows, g, 128, device="cuda")
    qt = q[..., :120].sum(-1).contiguous()
    keys = torch.randint(256, (rows, n, 16), device="cuda", dtype=torch.uint8)
    starts = torch.tensor([[start, 0]] * rows, device="cuda", dtype=torch.int32)
    ends = torch.tensor([[end, 0]] * rows, device="cuda", dtype=torch.int32)
    lo = torch.randn(rows, blocks, device="cuda") * .1 + 2
    norm_step = torch.full_like(lo, .002)
    bias = torch.randn(rows, g, blocks, device="cuda")
    rs, bs = (.01, .1) if g > 1 else (1., 8.)
    scores = torch.full((rows, n), -float("inf"), device="cuda")
    logits = torch.full((rows, g, n), -float("inf"), device="cuda")
    chunk_lse = torch.empty((rows, g, triton.cdiv(n, 256)), device="cuda")
    head_lse = torch.empty((rows, g), device="cuda")
    tc_chunks = triton.cdiv(end - start, args.tile)
    tc_lse = torch.empty((rows, g, tc_chunks), device="cuda")
    common = (q, qt, keys, starts, ends, lo, norm_step, bias, logits, chunk_lse,
              head_lse, scores, 120, end - start, prompt, end, 128, 32, rs, bs, g > 1)
    grouped_signature_score_into(*common)
    reference = scores.clone()
    k = int(n * .02)
    expected_indices = reference.topk(k, dim=1).indices
    results = []
    backends = ["scalar", "lookup", "tc_bf16", "tc_tf32x3"] if args.backend == "all" else [args.backend]
    for backend in backends:
        def step():
            if backend in ("scalar", "lookup"):
                grouped_signature_score_into(*common, backend == "lookup")
            else:
                tensor_score[(rows, tc_chunks)](
                    q, qt, keys, lo, norm_step, bias, logits if g > 1 else scores, tc_lse,
                    n, g, blocks, start, end, prompt, rs, bs, g > 1, backend == "tc_bf16", args.tile, tc_chunks,
                    num_warps=args.warps, num_stages=args.stages)
                if g > 1:
                    tensor_lse[(rows * g,)](tc_lse, head_lse, tc_chunks, triton.next_power_of_2(tc_chunks))
                    tensor_merge[(rows, triton.cdiv(end - start, 256))](
                        logits, head_lse, scores, n, g, start, end, triton.next_power_of_2(g), 256)
        try:
            step()
        except triton.runtime.errors.OutOfResources as error:
            if args.profile:
                raise
            result = dict(backend=backend, length=n, group=g, tile=args.tile,
                          warps=args.warps, stages=args.stages, median_us=None,
                          status="resource_limit", error=str(error))
            results.append(result)
            print(json.dumps(result), flush=True)
            continue
        if args.profile:
            torch.cuda.synchronize()
            with torch.cuda.nvtx.range("cometkv_score_profile"):
                step()
                torch.cuda.synchronize()
            return
        us, timings = graph_timing(step)
        error = (scores[:, start:end] - reference[:, start:end]).abs()
        actual_indices = scores.topk(k, dim=1).indices
        matched = (actual_indices[:, :, None] == expected_indices[:, None, :]).any(-1).float().mean()
        result = dict(backend=backend, length=n, group=g, median_us=us, timings_us=timings,
                      tile=args.tile, warps=args.warps, stages=args.stages,
                      max_abs_error=error.max().item(), mean_abs_error=error.mean().item(),
                      topk_overlap=matched.item(), dtype="FP32 query projection, packed uint8 signatures",
                      gpu=torch.cuda.get_device_name(), tail_budget_affected=False)
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
