import pytest
import torch

from cometkv import concat_static_recent_lookup_gather_uva_kv_update_cache


if not torch.cuda.is_available():
    pytest.skip("CometKV LRU cache tests require a CUDA GPU.", allow_module_level=True)


DTYPE = torch.bfloat16


def _stack_cpu_kv(cpu_keys, cpu_values):
    return torch.stack((cpu_keys, cpu_values), dim=2).contiguous().pin_memory()


def _gather(cpu_kv, request_token_ids, cache_token_ids, cache_locks, cache_keys, cache_values,
            *, static_capacity=1, recent_capacity=1, dim=128, stamps=None, step=None, ways=2):
    # Sparse-only gather (static_len=recent_len=0) so every request goes through the cache path.
    rows = cache_token_ids.size(0)
    request_count = request_token_ids.size(1)
    static_keys = torch.zeros((rows, static_capacity, dim), dtype=DTYPE, device="cuda")
    static_values = torch.zeros_like(static_keys)
    recent_keys = torch.zeros((rows, recent_capacity, dim), dtype=DTYPE, device="cuda")
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
    )
    torch.cuda.synchronize()
    return out_keys, out_values, hit_mask


def _fresh_cache(rows, cache_size, dim, lru):
    ids = torch.full((rows, cache_size), -1, dtype=torch.int32, device="cuda")
    locks = torch.zeros((rows, cache_size), dtype=torch.int32, device="cuda")
    keys = torch.zeros((rows, cache_size, dim), dtype=DTYPE, device="cuda")
    values = torch.zeros_like(keys)
    stamps = torch.zeros((rows, cache_size), dtype=torch.int32, device="cuda") if lru else None
    return ids, locks, keys, values, stamps


def test_lru_evicts_least_recently_stamped_way():
    # ways=4, single set (cache_size == ways). Insert 5 distinct tokens that all map to the same
    # set across successive steps; the 5th insert must evict the least-recently-used way (token 0),
    # leaving tokens {1,2,3,4}.
    rows, cache_size, dim, ways = 1, 4, 128, 4
    total_tokens = 32
    cpu_keys = torch.randn((rows, total_tokens, dim), dtype=DTYPE, pin_memory=True)
    cpu_values = torch.randn((rows, total_tokens, dim), dtype=DTYPE, pin_memory=True)
    cpu_kv = _stack_cpu_kv(cpu_keys, cpu_values)
    ids, locks, keys, values, stamps = _fresh_cache(rows, cache_size, dim, lru=True)
    step = torch.zeros((1,), dtype=torch.int32, device="cuda")

    for t in range(4):  # fill all 4 ways, increasing stamp each step
        step.fill_(t)
        req = torch.tensor([[t]], dtype=torch.int32, device="cuda")
        _gather(cpu_kv, req, ids, locks, keys, values, stamps=stamps, step=step, ways=ways)
    assert set(ids[0].tolist()) == {0, 1, 2, 3}

    step.fill_(10)
    req = torch.tensor([[4]], dtype=torch.int32, device="cuda")
    _gather(cpu_kv, req, ids, locks, keys, values, stamps=stamps, step=step, ways=ways)
    # token 0 (stamp 0, oldest) evicted; 1,2,3 kept; 4 inserted
    assert set(ids[0].tolist()) == {1, 2, 3, 4}, ids[0].tolist()


def test_lru_hit_refreshes_stamp_and_protects_from_eviction():
    # Prime tokens 0..3, then re-hit token 0 with a newer stamp so token 1 becomes the LRU victim.
    rows, cache_size, dim, ways = 1, 4, 128, 4
    total_tokens = 32
    cpu_keys = torch.randn((rows, total_tokens, dim), dtype=DTYPE, pin_memory=True)
    cpu_values = torch.randn((rows, total_tokens, dim), dtype=DTYPE, pin_memory=True)
    cpu_kv = _stack_cpu_kv(cpu_keys, cpu_values)
    ids, locks, keys, values, stamps = _fresh_cache(rows, cache_size, dim, lru=True)
    step = torch.zeros((1,), dtype=torch.int32, device="cuda")

    for t in range(4):
        step.fill_(t)
        _gather(cpu_kv, torch.tensor([[t]], dtype=torch.int32, device="cuda"),
                ids, locks, keys, values, stamps=stamps, step=step, ways=ways)

    # Re-hit token 0 at a newer stamp: it must be a HIT and its stamp refreshed above token 1's.
    step.fill_(20)
    _, _, hit_mask = _gather(cpu_kv, torch.tensor([[0]], dtype=torch.int32, device="cuda"),
                             ids, locks, keys, values, stamps=stamps, step=step, ways=ways)
    assert int(hit_mask[0, 0].item()) == 1

    # Now insert token 4: LRU victim is token 1 (stamp 1), NOT token 0 (refreshed to 20).
    step.fill_(30)
    _gather(cpu_kv, torch.tensor([[4]], dtype=torch.int32, device="cuda"),
            ids, locks, keys, values, stamps=stamps, step=step, ways=ways)
    survivors = set(ids[0].tolist())
    assert 0 in survivors and 4 in survivors and 1 not in survivors, survivors


def test_hit_returns_cached_kv_and_reports_hit_mask():
    rows, cache_size, dim, ways = 1, 4, 128, 4
    total_tokens = 32
    cpu_keys = torch.randn((rows, total_tokens, dim), dtype=DTYPE, pin_memory=True)
    cpu_values = torch.randn((rows, total_tokens, dim), dtype=DTYPE, pin_memory=True)
    cpu_kv = _stack_cpu_kv(cpu_keys, cpu_values)
    ids, locks, keys, values, stamps = _fresh_cache(rows, cache_size, dim, lru=True)
    step = torch.zeros((1,), dtype=torch.int32, device="cuda")

    req = torch.tensor([[7]], dtype=torch.int32, device="cuda")
    _, _, hm_miss = _gather(cpu_kv, req, ids, locks, keys, values, stamps=stamps, step=step, ways=ways)
    assert int(hm_miss[0, 0].item()) == 0  # first touch: miss
    ok, ov, hm_hit = _gather(cpu_kv, req, ids, locks, keys, values, stamps=stamps, step=step, ways=ways)
    assert int(hm_hit[0, 0].item()) == 1  # second: hit
    assert torch.allclose(ok[0, 0, 0], cpu_keys[0, 7].to("cuda"))
    assert torch.allclose(ov[0, 0, 0], cpu_values[0, 7].to("cuda"))


def test_ways2_no_stamps_matches_legacy_2way():
    # stamps=None + ways=2 must reproduce the legacy 2-way path exactly (same final cache state and
    # outputs) as calling the launcher without the LRU trailing args at all.
    rows, cache_size, dim = 2, 8, 128
    total_tokens = 40
    cpu_keys = torch.randn((rows, total_tokens, dim), dtype=DTYPE, pin_memory=True)
    cpu_values = torch.randn((rows, total_tokens, dim), dtype=DTYPE, pin_memory=True)
    cpu_kv = _stack_cpu_kv(cpu_keys, cpu_values)
    # Tokens chosen to map to DISTINCT 2-way sets (distinct token % (cache_size/2)) so the final
    # cache state is deterministic — otherwise concurrent blocks racing over one set make the
    # comparison scheduling-dependent (a test artifact, not a kernel difference).
    reqs = torch.tensor([[1, 2, 3, 0], [1, 2, 3, 0]], dtype=torch.int32, device="cuda")

    def run(pass_lru_args):
        ids, locks, keys, values, _ = _fresh_cache(rows, cache_size, dim, lru=False)
        for _ in range(3):  # repeat to exercise inserts + hits
            if pass_lru_args:
                out_k, out_v, hm = _gather(cpu_kv, reqs, ids, locks, keys, values,
                                           stamps=None, step=None, ways=2)
            else:
                rows_ = rows
                sk = torch.zeros((rows_, 1, dim), dtype=DTYPE, device="cuda")
                sv = torch.zeros_like(sk); rk = torch.zeros_like(sk); rv = torch.zeros_like(sk)
                out_k = torch.zeros((rows_, reqs.size(1), 1, dim), dtype=DTYPE, device="cuda")
                out_v = torch.zeros_like(out_k)
                hm = torch.zeros((rows_, reqs.size(1)), dtype=torch.int32, device="cuda")
                concat_static_recent_lookup_gather_uva_kv_update_cache(
                    sk, sv, rk, rv, reqs, cpu_kv, out_k, out_v, hm,
                    ids, locks, keys, values, 0, 0, reqs.size(1))
                torch.cuda.synchronize()
        return ids.clone(), out_k, out_v, hm

    ids_a, ka, va, hma = run(True)
    ids_b, kb, vb, hmb = run(False)
    assert torch.equal(ids_a, ids_b)
    assert torch.equal(ka, kb) and torch.equal(va, vb)
    assert torch.equal(hma, hmb)
