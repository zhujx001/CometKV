import pytest
import torch

from cometkv import concat_static_recent_lookup_gather_uva_kv_update_cache


if not torch.cuda.is_available():
    pytest.skip("CometKV score-policy cache tests require a CUDA GPU.", allow_module_level=True)


DTYPE = torch.bfloat16


def _stack_cpu_kv(cpu_keys, cpu_values):
    return torch.stack((cpu_keys, cpu_values), dim=2).contiguous().pin_memory()


def _gather(cpu_kv, request_token_ids, cache_token_ids, cache_locks, cache_keys, cache_values,
            *, stamps, step, ways, prio_buckets=None, prio_scores=None, cache_prio=None, dim=128):
    rows = cache_token_ids.size(0)
    request_count = request_token_ids.size(1)
    static_keys = torch.zeros((rows, 1, dim), dtype=DTYPE, device="cuda")
    static_values = torch.zeros_like(static_keys)
    recent_keys = torch.zeros((rows, 1, dim), dtype=DTYPE, device="cuda")
    recent_values = torch.zeros_like(recent_keys)
    out_keys = torch.zeros((rows, request_count, 1, dim), dtype=DTYPE, device="cuda")
    out_values = torch.zeros_like(out_keys)
    hit_mask = torch.zeros((rows, request_count), dtype=torch.int32, device="cuda")
    concat_static_recent_lookup_gather_uva_kv_update_cache(
        static_keys, static_values, recent_keys, recent_values,
        request_token_ids, cpu_kv, out_keys, out_values, hit_mask,
        cache_token_ids, cache_locks, cache_keys, cache_values,
        0, 0, request_count,
        stamps, step, ways,
        prio_buckets, prio_scores, cache_prio,
    )
    torch.cuda.synchronize()
    return out_keys, out_values, hit_mask


def _fresh_cache(rows, cache_size, dim=128):
    ids = torch.full((rows, cache_size), -1, dtype=torch.int32, device="cuda")
    locks = torch.zeros((rows, cache_size), dtype=torch.int32, device="cuda")
    keys = torch.zeros((rows, cache_size, dim), dtype=DTYPE, device="cuda")
    values = torch.zeros_like(keys)
    stamps = torch.zeros((rows, cache_size), dtype=torch.int32, device="cuda")
    return ids, locks, keys, values, stamps


def _setup(rows=1, cache_size=4, total_tokens=32, dim=128):
    cpu_keys = torch.randn((rows, total_tokens, dim), dtype=DTYPE, pin_memory=True)
    cpu_values = torch.randn((rows, total_tokens, dim), dtype=DTYPE, pin_memory=True)
    cpu_kv = _stack_cpu_kv(cpu_keys, cpu_values)
    cache = _fresh_cache(rows, cache_size, dim)
    step = torch.zeros((1,), dtype=torch.int32, device="cuda")
    return cpu_kv, cache, step


def _fill_single_set(cpu_kv, cache, step, ways, prio_buckets=None, prio_scores=None,
                     cache_prio=None):
    # Insert tokens 0..ways-1 at steps 0..ways-1 (cache_size == ways -> one set, all tokens map in).
    ids, locks, keys, values, stamps = cache
    for t in range(ways):
        step.fill_(t)
        req = torch.tensor([[t]], dtype=torch.int32, device="cuda")
        _gather(cpu_kv, req, ids, locks, keys, values, stamps=stamps, step=step, ways=ways,
                prio_buckets=prio_buckets, prio_scores=prio_scores, cache_prio=cache_prio)
    assert set(ids[0].tolist()) == set(range(ways))


def test_score_evicts_worst_bucket_way_not_lru():
    # Buckets (SMALLER = better): token 1 has the worst bucket, token 0 is the LRU way.
    # Score policy must evict token 1; plain LRU would have evicted token 0.
    ways, total_tokens = 4, 32
    cpu_kv, cache, step = _setup(cache_size=ways, total_tokens=total_tokens)
    ids, locks, keys, values, stamps = cache
    prio = torch.zeros((1, total_tokens), dtype=torch.int16, device="cuda")
    prio[0, 0], prio[0, 1], prio[0, 2], prio[0, 3] = 5, 100, 7, 9
    prio[0, 4] = 3
    _fill_single_set(cpu_kv, cache, step, ways, prio_buckets=prio)

    step.fill_(10)
    req = torch.tensor([[4]], dtype=torch.int32, device="cuda")
    _gather(cpu_kv, req, ids, locks, keys, values, stamps=stamps, step=step, ways=ways,
            prio_buckets=prio)
    assert set(ids[0].tolist()) == {0, 2, 3, 4}, ids[0].tolist()


def test_score_evicts_min_fp32_score_way():
    # fp32 scores (LARGER = better): token 2 has the lowest score -> evicted.
    ways, total_tokens = 4, 32
    cpu_kv, cache, step = _setup(cache_size=ways, total_tokens=total_tokens)
    ids, locks, keys, values, stamps = cache
    prio = torch.full((1, total_tokens), 10.0, dtype=torch.float32, device="cuda")
    prio[0, 2] = -3.0
    _fill_single_set(cpu_kv, cache, step, ways, prio_scores=prio)

    step.fill_(10)
    req = torch.tensor([[4]], dtype=torch.int32, device="cuda")
    _gather(cpu_kv, req, ids, locks, keys, values, stamps=stamps, step=step, ways=ways,
            prio_scores=prio)
    assert set(ids[0].tolist()) == {0, 1, 3, 4}, ids[0].tolist()


def test_score_tie_breaks_toward_older_stamp():
    # Tokens 1 and 2 share the worst bucket; token 1 has the older stamp -> evicted.
    ways, total_tokens = 4, 32
    cpu_kv, cache, step = _setup(cache_size=ways, total_tokens=total_tokens)
    ids, locks, keys, values, stamps = cache
    prio = torch.zeros((1, total_tokens), dtype=torch.int16, device="cuda")
    prio[0, 1] = 50
    prio[0, 2] = 50
    _fill_single_set(cpu_kv, cache, step, ways, prio_buckets=prio)  # stamps: t0=0 t1=1 t2=2 t3=3

    step.fill_(10)
    req = torch.tensor([[4]], dtype=torch.int32, device="cuda")
    _gather(cpu_kv, req, ids, locks, keys, values, stamps=stamps, step=step, ways=ways,
            prio_buckets=prio)
    assert set(ids[0].tolist()) == {0, 2, 3, 4}, ids[0].tolist()


def test_score_never_evicts_way_stamped_this_step():
    # Token 1 has the worst bucket but was HIT at the current step (stamp == step): the victim
    # must be the next-worst way (token 3) instead — demand-hot slots are protected.
    ways, total_tokens = 4, 32
    cpu_kv, cache, step = _setup(cache_size=ways, total_tokens=total_tokens)
    ids, locks, keys, values, stamps = cache
    prio = torch.zeros((1, total_tokens), dtype=torch.int16, device="cuda")
    prio[0, 1] = 100
    prio[0, 3] = 60
    _fill_single_set(cpu_kv, cache, step, ways, prio_buckets=prio)

    step.fill_(10)
    _, _, hm = _gather(cpu_kv, torch.tensor([[1]], dtype=torch.int32, device="cuda"),
                       ids, locks, keys, values, stamps=stamps, step=step, ways=ways,
                       prio_buckets=prio)
    assert int(hm[0, 0].item()) == 1  # hit refreshes token 1's stamp to 10

    _gather(cpu_kv, torch.tensor([[4]], dtype=torch.int32, device="cuda"),
            ids, locks, keys, values, stamps=stamps, step=step, ways=ways,
            prio_buckets=prio)
    assert set(ids[0].tolist()) == {0, 1, 2, 4}, ids[0].tolist()


def test_score_prefers_empty_way_over_eviction():
    ways, total_tokens = 4, 32
    cpu_kv, cache, step = _setup(cache_size=ways, total_tokens=total_tokens)
    ids, locks, keys, values, stamps = cache
    prio = torch.zeros((1, total_tokens), dtype=torch.int16, device="cuda")
    prio[0, 0] = 100  # worst bucket, but an empty way exists -> no eviction
    for t in range(2):
        step.fill_(t)
        _gather(cpu_kv, torch.tensor([[t]], dtype=torch.int32, device="cuda"),
                ids, locks, keys, values, stamps=stamps, step=step, ways=ways, prio_buckets=prio)

    step.fill_(5)
    _gather(cpu_kv, torch.tensor([[9]], dtype=torch.int32, device="cuda"),
            ids, locks, keys, values, stamps=stamps, step=step, ways=ways, prio_buckets=prio)
    survivors = set(x for x in ids[0].tolist() if x >= 0)
    assert survivors == {0, 1, 9}, ids[0].tolist()


def test_score_hit_returns_exact_kv():
    # The score policy only changes victim choice; hits still return the exact cached KV.
    ways, total_tokens = 4, 32
    cpu_kv, cache, step = _setup(cache_size=ways, total_tokens=total_tokens)
    ids, locks, keys, values, stamps = cache
    prio = torch.zeros((1, total_tokens), dtype=torch.int16, device="cuda")
    req = torch.tensor([[7]], dtype=torch.int32, device="cuda")
    _gather(cpu_kv, req, ids, locks, keys, values, stamps=stamps, step=step, ways=ways,
            prio_buckets=prio)
    ok, ov, hm = _gather(cpu_kv, req, ids, locks, keys, values, stamps=stamps, step=step,
                         ways=ways, prio_buckets=prio)
    assert int(hm[0, 0].item()) == 1
    assert torch.equal(ok[0, 0, 0], cpu_kv[0, 7, 0].to("cuda"))
    assert torch.equal(ov[0, 0, 0], cpu_kv[0, 7, 1].to("cuda"))


def test_multi_set_mapping_respects_prio_rows():
    # rows=2 with different prio rows: the same insert evicts different ways per row.
    ways, total_tokens, rows = 4, 32, 2
    cpu_kv, _, step = _setup(rows=rows, cache_size=ways, total_tokens=total_tokens)
    cache = _fresh_cache(rows, ways)
    ids, locks, keys, values, stamps = cache
    prio = torch.zeros((rows, total_tokens), dtype=torch.int16, device="cuda")
    prio[0, 1] = 100  # row 0: token 1 worst
    prio[1, 3] = 100  # row 1: token 3 worst
    for t in range(ways):
        step.fill_(t)
        req = torch.tensor([[t], [t]], dtype=torch.int32, device="cuda")
        _gather(cpu_kv, req, ids, locks, keys, values, stamps=stamps, step=step, ways=ways,
                prio_buckets=prio)

    step.fill_(10)
    req = torch.tensor([[4], [4]], dtype=torch.int32, device="cuda")
    _gather(cpu_kv, req, ids, locks, keys, values, stamps=stamps, step=step, ways=ways,
            prio_buckets=prio)
    assert set(ids[0].tolist()) == {0, 2, 3, 4}, ids[0].tolist()
    assert set(ids[1].tolist()) == {0, 1, 2, 4}, ids[1].tolist()


def test_demand_gather_maintains_cache_prio():
    ways, total_tokens = 4, 32
    cpu_kv, cache, step = _setup(cache_size=ways, total_tokens=total_tokens)
    ids, locks, keys, values, stamps = cache
    prio = torch.zeros((1, total_tokens), dtype=torch.int16, device="cuda")
    cache_prio = torch.full((1, ways), -1, dtype=torch.int16, device="cuda")
    req = torch.tensor([[7]], dtype=torch.int32, device="cuda")

    prio[0, 7] = 33
    _gather(cpu_kv, req, ids, locks, keys, values, stamps=stamps, step=step,
            ways=ways, prio_buckets=prio, cache_prio=cache_prio)
    slot = ids[0].tolist().index(7)
    assert int(cache_prio[0, slot].item()) == 33

    prio[0, 7] = 21
    step.fill_(1)
    _, _, hm = _gather(cpu_kv, req, ids, locks, keys, values, stamps=stamps, step=step,
                       ways=ways, prio_buckets=prio, cache_prio=cache_prio)
    assert int(hm[0, 0].item()) == 1
    assert int(cache_prio[0, slot].item()) == 21
