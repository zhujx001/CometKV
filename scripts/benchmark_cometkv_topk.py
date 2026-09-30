#!/usr/bin/env python3
"""Time exact top-k selection with the runtime's int32 output contract."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'library/cometkv')]
from cometkv import exact_topk_indices_into
from benchmark_cometkv_score_backends import graph_timing


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length', type=int, default=32768)
    parser.add_argument('--rows', type=int, default=8)
    parser.add_argument('--k', type=int)
    parser.add_argument('--distribution', choices=['random', 'logprob', 'equal', 'narrow'], default='random')
    parser.add_argument('--backend', choices=['all', 'torch', 'radix'], default='all')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.profile and args.backend == 'all':
        parser.error('--profile requires one backend')
    torch.set_num_threads(8)
    torch.manual_seed(345)
    n, rows = args.length, args.rows
    k, start, end = args.k or max(16, int(.02 * n)), 4, n - 32
    scores = torch.randn((rows, n), device='cuda')
    if args.distribution == 'logprob':
        scores = torch.logsumexp(torch.randn(rows, 4, n, device='cuda').log_softmax(-1), dim=1) - 1.3862943611198906
    elif args.distribution == 'equal':
        scores.zero_()
    elif args.distribution == 'narrow':
        scores.mul_(1e-3).add_(-8.)
    scores[:, :start] = scores[:, end:] = -float('inf')
    indices = torch.full((rows, k + 13), -1, device='cuda', dtype=torch.int32)
    histogram = torch.empty((rows, (n + 1023) // 1024, 288), device='cuda', dtype=torch.int32)
    candidates = torch.empty_like(scores, dtype=torch.int32)
    state = torch.empty((rows, 7), device='cuda', dtype=torch.int32)
    expected_values = scores.topk(k, dim=1).values
    results = []
    for backend in (['torch', 'radix'] if args.backend == 'all' else [args.backend]):
        def step():
            if backend == 'torch':
                indices[:, :k].copy_(scores.topk(k, dim=1, sorted=False).indices.to(torch.int32))
            else:
                exact_topk_indices_into(scores, indices, histogram, candidates, state, k, start, end)
        step()
        if args.profile:
            for _ in range(10):
                step()
            torch.cuda.synchronize()
            with torch.cuda.nvtx.range('cometkv_topk_profile'):
                step()
                torch.cuda.synchronize()
            return
        us, timings = graph_timing(step)
        actual_values = scores.gather(1, indices[:, :k].long()).sort(dim=1, descending=True).values
        assert torch.equal(actual_values, expected_values)
        assert (indices[:, k:] == -1).all()
        for row in indices[:, :k]:
            assert row.unique().numel() == k
        result = dict(backend=backend, length=n, rows=rows, k=k, distribution=args.distribution,
                      median_us=us, timings_us=timings, exact_values=True, gpu=torch.cuda.get_device_name())
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
