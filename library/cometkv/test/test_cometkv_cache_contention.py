"""Cache eviction must never change KV rows already selected for attention."""
import pytest
import torch

from cometkv import concat_static_recent_lookup_gather_uva_kv_update_cache as gather


if not torch.cuda.is_available():
    pytest.skip("CometKV contention tests require CUDA.", allow_module_level=True)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("policy,dim,dtype", [
    ("score", 128, torch.bfloat16),
    ("lru", 128, torch.bfloat16),
    ("legacy", 128, torch.bfloat16),
    ("legacy", 64, torch.float16),
])
def test_eviction_preserves_exact_attention_kv(graph, policy, dim, dtype):
    rows, cache_size, ways, distinct = 16, 64, 4, 32
    static_len, recent_len = 4, 3
    # Both legacy and W-way mapping send these IDs to the same cache set.
    tokens = torch.arange(distinct) * cache_size
    request_cpu = torch.cat([tokens, tokens[:1], torch.tensor([-1])])[None].repeat(rows, 1)
    width = request_cpu.shape[1]
    total_tokens = distinct * cache_size
    generator = torch.Generator().manual_seed(829)
    cpu_kv = torch.randn(rows, total_tokens, 2, dim, generator=generator, dtype=dtype).pin_memory()
    requested = request_cpu.to(device="cuda", dtype=torch.int32)
    expected = cpu_kv[torch.arange(rows)[:, None], request_cpu.clamp_min(0)].clone()
    expected[:, -1].zero_()
    static_k = torch.randn(rows, static_len, dim, device="cuda", dtype=dtype)
    static_v = torch.randn_like(static_k)
    recent_k = torch.randn(rows, recent_len, dim, device="cuda", dtype=dtype)
    recent_v = torch.randn_like(recent_k)
    total_width = static_len + recent_len + width
    out_k = torch.full((rows, total_width + 5, 1, dim), 77, device="cuda", dtype=dtype)
    out_v = torch.full_like(out_k, 77)
    ids = torch.full((rows, cache_size), -1, device="cuda", dtype=torch.int32)
    locks = torch.zeros_like(ids)
    stamps = torch.zeros_like(ids) if policy != "legacy" else None
    keys = torch.zeros(rows, cache_size, dim, device="cuda", dtype=dtype)
    values = torch.zeros_like(keys)
    hits = torch.zeros(rows, width, device="cuda", dtype=torch.int32)
    step = torch.zeros(1, device="cuda", dtype=torch.int32)
    priorities = (-torch.arange(total_tokens, device="cuda", dtype=torch.float32))[None].repeat(rows, 1)

    def launch():
        gather(static_k, static_v, recent_k, recent_v, requested, cpu_kv,
               out_k, out_v, hits, ids, locks, keys, values,
               static_len, recent_len, width, stamps, step, ways, None,
               priorities if policy == "score" else None, None)

    if graph:
        launch()
        torch.cuda.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            launch()
        run = captured.replay
    else:
        run = launch
    expected_k = torch.cat([static_k.cpu(), recent_k.cpu(), expected[:, :, 0]], dim=1)
    expected_v = torch.cat([static_v.cpu(), recent_v.cpu(), expected[:, :, 1]], dim=1)
    for iteration in range(64):
        step.fill_(iteration + 1)
        run()
        torch.cuda.synchronize()
        assert torch.equal(out_k[:, :total_width, 0].cpu(), expected_k), (policy, graph, iteration, "K")
        assert torch.equal(out_v[:, :total_width, 0].cpu(), expected_v), (policy, graph, iteration, "V")
        assert bool((out_k[:, total_width:] == 77).all())
        assert bool((out_v[:, total_width:] == 77).all())
        assert not bool(locks.any())
