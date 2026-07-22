import math

import numpy as np
import pytest
import torch

try:
    from cache_hub.exact_topk_cache import exact_topk_cache
    from flash_attn import flash_attn_with_kvcache
    _HAS_GPU = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    _HAS_GPU = False

pytestmark = pytest.mark.skipif(not _HAS_GPU, reason="requires CUDA + flash_attn")

BSZ, LAYERS, KV_HEAD, HEADS, DIM = 2, 2, 2, 8, 64


def make_cache(total, budget=0.1, max_length=None, **kw):
    torch.manual_seed(7)
    cache = exact_topk_cache(
        valid_start=np.zeros(BSZ, dtype=np.int64),
        layer_num=LAYERS,
        batch_size=BSZ,
        max_length=max_length or (total + 16),
        num_key_value_heads=KV_HEAD,
        num_heads=HEADS,
        head_dim=DIM,
        dtype=torch.bfloat16,
        layer_mapping={str(i): "cuda:0" for i in range(LAYERS)},
        prefill_bsz=1,
        num_gpus=1,
        model_size=8,
        retrieval_budget=budget,
        **kw,
    )
    for ldx in range(LAYERS):
        cache.key_cache[ldx][:, :total].normal_()
        cache.value_cache[ldx][:, :total].normal_()
    cache.context = total  # as if prefill committed `total` tokens
    return cache


def ref_scores(cache, q, layer_idx, total):
    group = HEADS // KV_HEAD
    q_mean = (q.view(BSZ, KV_HEAD, group, DIM).to(torch.float32).mean(dim=2)
              .to(torch.bfloat16).view(BSZ, KV_HEAD, DIM, 1))
    keys = cache.key_cache[layer_idx][:, :total]
    return torch.matmul(keys.permute(0, 2, 1, 3), q_mean).view(BSZ, KV_HEAD, total).float()


def test_fixed_topk_frozen_from_prompt_length():
    cache = make_cache(total=500, budget=0.1)
    cache._ensure_plan()
    assert cache.fixed_topk == 50
    cache.context = 900  # decode advanced; k must not grow
    cache._ensure_plan()
    assert cache.fixed_topk == 50


def test_full_budget_matches_dense_flash_attention():
    total = 256
    cache = make_cache(total=total, budget=1.0)
    cache.fixed_topk = total  # cover every token -> must equal dense attention
    q = torch.randn(BSZ, 1, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    # last layer: cache.context == total exactly (post-increment convention)
    out = cache.sparse_attention(q, LAYERS - 1)
    ref = flash_attn_with_kvcache(
        q=q,
        k_cache=cache.key_cache[LAYERS - 1][:, :total].contiguous(),
        v_cache=cache.value_cache[LAYERS - 1][:, :total].contiguous(),
    )
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_selection_uses_group_mean_query_per_kv_head():
    total = 300
    cache = make_cache(total=total, budget=0.1)
    cache._ensure_plan()
    k = cache.fixed_topk
    q = torch.randn(BSZ, 1, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    scores = ref_scores(cache, q, 0, total + 1)  # layer 0: total = context + 1
    expected = torch.topk(scores, k, dim=-1).indices

    seen = {}

    orig_gather = torch.gather

    def spy(inp, dim, index):
        seen["index"] = index
        return orig_gather(inp, dim, index)

    torch.gather = spy
    try:
        cache.sparse_attention(q, 0)
    finally:
        torch.gather = orig_gather

    got = seen["index"][:, :, :, 0].permute(0, 2, 1)  # (bsz, kv_head, k)
    for b in range(BSZ):
        for h in range(KV_HEAD):
            assert set(got[b, h].tolist()) == set(expected[b, h].tolist())


def test_force_sink_and_recent_are_always_selected():
    total = 300
    cache = make_cache(total=total, budget=0.2, force_sink=4, force_recent=32)
    cache._ensure_plan()
    q = torch.randn(BSZ, 1, HEADS, DIM, dtype=torch.bfloat16, device="cuda")

    seen = {}
    orig_gather = torch.gather

    def spy(inp, dim, index):
        seen["index"] = index
        return orig_gather(inp, dim, index)

    torch.gather = spy
    try:
        cache.sparse_attention(q, LAYERS - 1)  # total tokens = cache.context = 300
    finally:
        torch.gather = orig_gather

    got = seen["index"][:, :, :, 0].permute(0, 2, 1)
    must_have = set(range(4)) | set(range(total - 32, total))
    for b in range(BSZ):
        for h in range(KV_HEAD):
            assert must_have.issubset(set(got[b, h].tolist()))


def test_attention_math_matches_fp32_reference():
    total = 200
    cache = make_cache(total=total, budget=0.25)
    cache._ensure_plan()
    k = cache.fixed_topk
    q = torch.randn(BSZ, 1, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    out = cache.sparse_attention(q, LAYERS - 1)

    group = HEADS // KV_HEAD
    scores = ref_scores(cache, q, LAYERS - 1, total)
    sel = torch.topk(scores, k, dim=-1).indices  # (bsz, kv_head, k)
    ref = torch.empty_like(out, dtype=torch.float32)
    for b in range(BSZ):
        for h in range(HEADS):
            kv = h // group
            ks = cache.key_cache[LAYERS - 1][b, sel[b, kv], kv].float()
            vs = cache.value_cache[LAYERS - 1][b, sel[b, kv], kv].float()
            att = torch.softmax((ks @ q[b, 0, h].float()) / math.sqrt(DIM), dim=0)
            ref[b, 0, h] = att @ vs
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)


def full_attention_reference(cache, q, layer_idx, total):
    """Dense fp32 attention over ALL cached tokens — the quantity the hybrid estimator targets."""
    group = HEADS // KV_HEAD
    ref = torch.empty(BSZ, 1, HEADS, DIM, dtype=torch.float32, device="cuda")
    for b in range(BSZ):
        for h in range(HEADS):
            kv = h // group
            ks = cache.key_cache[layer_idx][b, :total, kv].float()
            vs = cache.value_cache[layer_idx][b, :total, kv].float()
            att = torch.softmax((ks @ q[b, 0, h].float()) / math.sqrt(DIM), dim=0)
            ref[b, 0, h] = att @ vs
    return ref


def test_sample_frac_zero_is_bit_identical_to_pure_topk():
    total = 300
    q = torch.randn(BSZ, 1, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    cache_a = make_cache(total=total, budget=0.1)                    # legacy default
    cache_b = make_cache(total=total, budget=0.1, sample_frac=0.0)   # explicit zero
    assert torch.equal(cache_a.sparse_attention(q, LAYERS - 1),
                       cache_b.sparse_attention(q, LAYERS - 1))


def test_hybrid_estimator_reduces_full_attention_error_vs_pure_topk():
    # The self-normalized IS estimate over many reseeded draws must approach FULL attention
    # much closer than the truncate-and-renormalize top-k output does. Data is shaped so the
    # tail carries real mass (small logit spread => diffuse attention, the fwe regime).
    total, budget = 512, 0.05
    torch.manual_seed(11)
    q = (0.3 * torch.randn(BSZ, 1, HEADS, DIM)).to(torch.bfloat16).cuda()

    cache_topk = make_cache(total=total, budget=budget)
    full_ref = full_attention_reference(cache_topk, q, LAYERS - 1, total)
    out_topk = cache_topk.sparse_attention(q, LAYERS - 1).float()
    err_topk = (out_topk - full_ref).norm() / full_ref.norm()

    n_trials, acc = 64, None
    for trial in range(n_trials):
        cache_h = make_cache(total=total, budget=budget, sample_frac=0.5,
                             sample_tau=1.0, sample_seed=1000 + trial)
        out_h = cache_h.sparse_attention(q, LAYERS - 1).float()
        acc = out_h if acc is None else acc + out_h
    err_hybrid_mean = (acc / n_trials - full_ref).norm() / full_ref.norm()

    # Averaged over draws the sampling noise cancels; the residual is estimator bias, which
    # must be well below the top-k truncation bias for the design to make sense.
    assert err_hybrid_mean < 0.5 * err_topk, (
        f"hybrid mean-of-{n_trials} error {err_hybrid_mean:.4f} not < half of "
        f"pure top-k truncation error {err_topk:.4f}"
    )


def test_hybrid_single_draw_output_is_sane_and_seeded():
    total = 400
    q = torch.randn(BSZ, 1, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    cache_a = make_cache(total=total, budget=0.1, sample_frac=0.5, sample_seed=42)
    cache_b = make_cache(total=total, budget=0.1, sample_frac=0.5, sample_seed=42)
    out_a = cache_a.sparse_attention(q, LAYERS - 1)
    out_b = cache_b.sparse_attention(q, LAYERS - 1)
    assert torch.isfinite(out_a.float()).all()
    assert out_a.shape == (BSZ, 1, HEADS, DIM)
    assert torch.equal(out_a, out_b), "same seed must reproduce the same draws/output"
    # And the head (exact) part keeps it anchored: error vs full attention should not blow
    # past the pure-topk error by more than the sampling-variance margin.
    full_ref = full_attention_reference(cache_a, q, LAYERS - 1, total)
    cache_t = make_cache(total=total, budget=0.1)
    err_topk = (cache_t.sparse_attention(q, LAYERS - 1).float() - full_ref).norm() / full_ref.norm()
    err_h = (out_a.float() - full_ref).norm() / full_ref.norm()
    assert err_h < 3.0 * err_topk + 0.05
