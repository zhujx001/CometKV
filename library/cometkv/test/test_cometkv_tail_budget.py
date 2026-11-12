"""Growing head budgets, tail budget accounting (paper: tail inside rho; legacy: additional
tail), the sampling context threshold, token-cache sizing, and tail sampling regressions."""

import inspect
import math
import os

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CometKV tail/budget tests require CUDA", allow_module_level=True)

from cache_hub.cometkv_cache import cometkv_cache
from cometkv import sampled_tail_attention_merge


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith("COMETKV_SAMPLE_") or name in (
            "COMETKV_MAX_RETRIEVAL_TOPK", "COMETKV_QUERY_AGG", "COMETKV_STATS_MODE",
            "COMETKV_TAIL_IN_BUDGET", "COMETKV_TOKEN_CACHE_SLOTS", "COMETKV_TOKEN_CACHE_ROUND",
        ):
            monkeypatch.delenv(name)
    monkeypatch.setenv("COMETKV_TOKEN_CACHE_MULT", "1")
    torch.manual_seed(734)


def make_cache(prompt=128, generation=514, layers=1, **overrides):
    # Defaults pin the legacy accounting (a fixed m on top of the head, no context threshold) so
    # the short-prompt regressions below keep exercising the tail; paper-mode tests override them.
    parameters = dict(
        tail_in_budget=False, sample_min_context=0, sample_mode="size",
        valid_start=np.zeros(1, dtype=np.int32), layer_num=layers, batch_size=1,
        max_length=prompt + generation, max_new_length=generation,
        num_key_value_heads=2, num_heads=8, head_dim=128, dtype=DTYPE,
        layer_mapping={str(layer): DEVICE for layer in range(layers)},
        static_pattern_start=4, static_pattern_end=32, retrieval_budget=0.02,
        sig_bits=128, sig_topk=0, sig_min_retrieval_topk=16,
        sig_chunk_size=16384, sig_seed=1234, sig_mode="random_orth",
        sig_token_cache_size=16, prefill_bsz=1, num_gpus=1, model_size=8,
        query_aggregation="q_sum", sample_size=64,
    )
    parameters.update(overrides)
    cache = cometkv_cache(**parameters)
    for layer in range(layers):
        keys = torch.randn(1, prompt, 2, 128, device=DEVICE, dtype=DTYPE)
        values = torch.randn_like(keys)
        queries = torch.randn(1, prompt, 8, 128, device=DEVICE, dtype=DTYPE)
        cache.prefill_update_kv_cache(queries, keys, values, layer, 0)
    cache.prepare_cache()
    torch.cuda.synchronize()
    return cache


@pytest.mark.parametrize("prompt,expected", [(1024, 184), (8192, 327)])
@pytest.mark.parametrize("include_preserved", [False, True])
def test_long_generation_head_matches_length_budget(prompt, expected, include_preserved):
    cache = make_cache(prompt=prompt, generation=8194,
                       exclude_preserved_from_budget=not include_preserved)
    assert cache.selected_indices_buffer.size(1) >= expected - (37 if include_preserved else 0)
    for generated in (0, 129, 257, 8193):
        visible = prompt + generated
        preserved = 36 if generated == 0 else 37  # slide boundary: sink 4 + local 33
        target = max(16, int(0.02 * visible) - (preserved if include_preserved else 0))
        cache._update_retrieval_plan(cache.prompt_lengths + generated)
        assert cache.active_sparse_len_host == target
        assert cache.requested_sparse_len_host == target
        assert (cache.topk_buffer == target).all()
        assert cache.active_sample_len_host == 64
        assert cache.active_retrieval_len_host == target + 64


@pytest.mark.parametrize("size", [0, 32, 64, 128, 256])
def test_explicit_tail_size_does_not_reduce_head(size):
    cache = make_cache(prompt=1024, generation=1026, sample_size=size)
    for generated, head in ((0, 20), (1025, 40)):
        cache._update_retrieval_plan(cache.prompt_lengths + generated)
        assert cache.active_sparse_len_host == head
        assert cache.active_sample_len_host == size
        assert cache.active_retrieval_len_host == head + size


@pytest.mark.parametrize("fixed_k", [0, 50])
def test_explicit_head_cap_applies_to_capacity_and_actual_k(fixed_k):
    cache = make_cache(prompt=1024, generation=8194, sig_topk=fixed_k,
                       sig_max_retrieval_topk=32)
    assert cache.selected_indices_buffer.size(1) == 32
    cache._update_retrieval_plan(cache.prompt_lengths + 8193)
    assert cache.active_sparse_len_host == 32
    assert cache.active_sample_len_host == 64


def test_no_sampling_when_head_already_covers_candidates():
    cache = make_cache(sig_topk=1024)
    assert cache.active_sparse_len_host == cache.active_candidates_host
    assert cache.active_sample_len_host == 0


def test_unplanned_capacity_overflow_is_not_silently_clipped():
    cache = make_cache(prompt=1024, generation=8194)
    cache.selected_indices_buffer = cache.selected_indices_buffer[:, :21].contiguous()
    with pytest.raises(RuntimeError, match="exceeds preallocated"):
        cache._update_retrieval_plan(cache.prompt_lengths + 8193)


def test_paper_accounting_is_the_constructor_default():
    parameters = inspect.signature(cometkv_cache.__init__).parameters
    expected = dict(tail_in_budget=True, sample_min_context=32768, sample_mode="fraction",
                    sample_frac=0.25, sample_size=256, sample_max_m=256, sample_min_m=64,
                    token_cache_mult=2.0, token_cache_round=8)
    assert {name: parameters[name].default for name in expected} == expected


@pytest.mark.parametrize("budget,min_m,expected_m", [
    (0.02, 64, 0),     # K_t = 163: floor(K_t / 4) = 40 < MIN_M 64, head-only
    (0.02, 32, 40),    # MIN_M 32: m = floor(K_t / 4)
    (0.1, 64, 204),    # K_t = 819
    (0.2, 64, 256),    # K_t = 1638: floor(K_t / 4) = 409, capped at MAX_M 256
])
def test_fraction_mode_samples_a_quarter_of_the_budget_capped_at_max_m(budget, min_m, expected_m):
    cache = make_cache(prompt=8192, generation=2, retrieval_budget=budget, sample_mode="fraction",
                       sample_frac=0.25, sample_min_m=min_m, tail_in_budget=True, sample_min_context=0)
    budget_k = int(budget * 8192)
    assert cache.active_sample_len_host == expected_m
    assert cache.active_sparse_len_host == budget_k - expected_m
    assert cache.active_retrieval_len_host == budget_k


def test_fraction_mode_with_additional_accounting_keeps_the_full_head():
    cache = make_cache(prompt=8192, generation=2, retrieval_budget=0.1, sample_mode="fraction",
                       sample_frac=0.25, tail_in_budget=False, sample_min_context=0)
    assert (cache.active_sparse_len_host, cache.active_sample_len_host) == (819, 204)


def test_off_mode_never_samples():
    cache = make_cache(prompt=8192, generation=2, sample_mode="off", sample_size=256,
                       tail_in_budget=True, sample_min_context=0)
    assert (cache.active_sparse_len_host, cache.active_sample_len_host) == (163, 0)


@pytest.mark.parametrize("env,match", [
    ({"COMETKV_SAMPLE_MODE": "ratio"}, "COMETKV_SAMPLE_MODE"),
    ({"COMETKV_SAMPLE_SIZE": "64"}, "COMETKV_SAMPLE_MODE"),       # SIZE alone is ambiguous now
    ({"COMETKV_SAMPLE_MODE": "fraction", "COMETKV_SAMPLE_FRAC": "1.0"}, "COMETKV_SAMPLE_FRAC"),
    ({"COMETKV_TOKEN_CACHE_ROUND": "12"}, "COMETKV_TOKEN_CACHE_ROUND"),
    ({"COMETKV_TOKEN_CACHE_SLOTS": "100"}, "COMETKV_TOKEN_CACHE_SLOTS"),
])
def test_invalid_sampling_and_cache_settings_fail_fast(monkeypatch, env, match):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=match):
        make_cache()


def test_sample_mode_environment_selects_size_with_the_given_count(monkeypatch):
    monkeypatch.setenv("COMETKV_SAMPLE_MODE", "size")
    monkeypatch.setenv("COMETKV_SAMPLE_SIZE", "48")
    cache = make_cache(prompt=8192, generation=2, sample_mode="fraction", tail_in_budget=True,
                       sample_min_context=0)
    assert (cache.sample_mode, cache.active_sample_len_host, cache.active_sparse_len_host) == ("size", 48, 115)


@pytest.mark.parametrize("size", [32, 64, 128])
def test_paper_tail_is_charged_to_the_budget(size):
    # K_t = floor(0.02 * L_t) at L_t = 8192 / 8321 / 8449; the head gives up m of them.
    cache = make_cache(prompt=8192, generation=514, sample_size=size,
                       tail_in_budget=True, sample_min_context=0)
    for generated, budget in ((0, 163), (129, 166), (257, 168)):
        cache._update_retrieval_plan(cache.prompt_lengths + generated)
        assert cache.active_sample_len_host == size
        assert cache.active_sparse_len_host == budget - size
        assert cache.requested_sparse_len_host == budget - size
        assert (cache.topk_buffer == budget - size).all()
        assert cache.active_retrieval_len_host == budget


@pytest.mark.parametrize("in_budget", [True, False])
def test_tail_starts_at_the_context_threshold(in_budget):
    cache = make_cache(prompt=8192, generation=514, sample_size=64,
                       tail_in_budget=in_budget, sample_min_context=8321)
    for generated, budget, m in ((0, 163, 0), (128, 166, 0), (129, 166, 64), (257, 168, 64)):
        cache._update_retrieval_plan(cache.prompt_lengths + generated)
        head = budget - m if in_budget else budget
        assert cache.active_sample_len_host == m
        assert cache.active_sparse_len_host == head
        assert cache.active_retrieval_len_host == head + m


def test_paper_tail_is_disabled_when_the_budget_cannot_hold_it():
    # K_t = 20 / 40 <= m = 64: a non-empty head and the tail cannot both fit in rho.
    cache = make_cache(prompt=1024, generation=1026, sample_size=64,
                       tail_in_budget=True, sample_min_context=0)
    for generated, budget in ((0, 20), (1025, 40)):
        cache._update_retrieval_plan(cache.prompt_lengths + generated)
        assert cache.active_sample_len_host == 0
        assert cache.active_sparse_len_host == budget
    # m == K_t would also leave an empty head.
    cache = make_cache(prompt=8192, generation=2, sample_size=163,
                       tail_in_budget=True, sample_min_context=0)
    assert cache.active_sample_len_host == 0
    assert cache.active_sparse_len_host == 163


@pytest.mark.parametrize("flag,expected", [("0", False), ("1", True)])
def test_tail_budget_environment_overrides_the_config(monkeypatch, flag, expected):
    monkeypatch.setenv("COMETKV_TAIL_IN_BUDGET", flag)
    monkeypatch.setenv("COMETKV_SAMPLE_MIN_CONTEXT", "4096")
    cache = make_cache(tail_in_budget=not expected, sample_min_context=0)
    assert cache.tail_in_budget is expected
    assert cache.sample_min_context == 4096


@pytest.mark.parametrize("name,value", [
    ("COMETKV_TAIL_IN_BUDGET", "yes"), ("COMETKV_SAMPLE_MIN_CONTEXT", "-1"),
])
def test_invalid_tail_budget_settings_fail_fast(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        make_cache()


@pytest.mark.parametrize("mult_env,prompt,generation,expected", [
    (None, 8192, 514, 352),   # default: K = floor(0.02 * 8706) = 174 -> 2K = 348 -> 8-way sets
    ("2", 2048, 4096, 248),   # sized for the largest K of the request: floor(0.02 * 6144) = 122
    ("1", 8192, 514, 1024),   # MULT <= 1 keeps the legacy 1024-rounded, top-k sized cache
])
def test_token_cache_holds_twice_the_request_budget(monkeypatch, mult_env, prompt, generation, expected):
    if mult_env is None:
        monkeypatch.delenv("COMETKV_TOKEN_CACHE_MULT")
    else:
        monkeypatch.setenv("COMETKV_TOKEN_CACHE_MULT", mult_env)
    cache = make_cache(prompt=prompt, generation=generation)
    assert cache.sig_token_cache_size == expected
    assert all(ids.size(1) == expected for ids in cache.token_cache_ids)


@pytest.mark.parametrize("clip", [0.0, 4.0])
def test_shared_full_support_draws_recover_different_layer_heads(clip):
    # Exact logits all zero. Balanced full-support draws remove sampling noise.
    # V=[8,4,0,0], so full attention is 3 for either head choice. Old owner-tail
    # reuse gave 2 on the second layer even with infinitely many samples.
    m, dim, group = 96, 128, 4
    q = torch.zeros((1, group, dim), dtype=DTYPE, device=DEVICE)
    keys = torch.zeros((1, m, dim), dtype=DTYPE, device=DEVICE)
    values = torch.zeros((4, dim), dtype=DTYPE, device=DEVICE)
    values[:, 0] = torch.tensor([8, 4, 0, 0], device=DEVICE, dtype=DTYPE)
    draws = torch.arange(4, device=DEVICE, dtype=torch.int32).repeat(m // 4)[None]
    corr = torch.full((1, m), -math.log(m / 4), device=DEVICE)
    lse = torch.zeros((1, group), device=DEVICE)
    for head in (0, 1):
        output = values[head].expand(1, group, dim).contiguous().clone()
        heads = torch.tensor([[head, -1, -1]], dtype=torch.int32, device=DEVICE)
        sampled_tail_attention_merge(q, keys, values[draws.long()].contiguous(), corr,
                                     output, lse, dim ** -0.5, clip, draws, heads, 1)
        assert (output[..., 0] == 3).all()
        assert (output[..., 1:] == 0).all()


@pytest.mark.parametrize("clip", [0.0, 0.5, 4.0])
def test_all_draws_in_head_preserve_main_exactly(clip):
    q = torch.randn((2, 4, 128), dtype=DTYPE, device=DEVICE)
    keys = torch.randn((2, 32, 128), dtype=DTYPE, device=DEVICE)
    values = torch.randn_like(keys)
    output = torch.randn_like(q)
    expected = output.clone()
    indices = torch.zeros((2, 32), dtype=torch.int32, device=DEVICE)
    heads = torch.zeros((2, 1), dtype=torch.int32, device=DEVICE)
    corr = torch.zeros((2, 32), device=DEVICE)
    lse = torch.randn((2, 4), device=DEVICE)
    sampled_tail_attention_merge(q, keys, values, corr, output, lse, 128 ** -0.5,
                                 clip, indices, heads, 1)
    assert torch.equal(output, expected)


@pytest.mark.parametrize("head_len,m,dim,unaligned", [
    (0, 17, 31, False), (1, 17, 128, False), (1, 256, 128, False),
    (17, 65, 64, False), (655, 256, 128, False), (1310, 256, 128, False),
    (4096, 96, 256, False), (4097, 33, 96, False), (17, 17, 128, True),
])
@pytest.mark.parametrize("clip", [0.0, 0.5])
def test_tail_merge_large_heads_and_padding_match_dense_oracle(head_len, m, dim, unaligned, clip):
    """Exercise exact membership, duplicates, inactive capacity and large-head fallback."""
    rows, group = 2, 4
    q = torch.randn((rows, group, dim), device=DEVICE, dtype=DTYPE)
    keys = torch.randn((rows, m, dim), device=DEVICE, dtype=DTYPE)
    values = torch.randn_like(keys)
    if unaligned:
        keys = torch.randn(rows * m * dim + 1, device=DEVICE, dtype=DTYPE)[1:].view(rows, m, dim)
        values = torch.randn(rows * m * dim + 1, device=DEVICE, dtype=DTYPE)[1:].view(rows, m, dim)
    output = torch.randn_like(q)
    main = output.float().clone()
    lse = torch.randn((rows, group), device=DEVICE) + 3
    # Widely spaced IDs and repeated heads exercise collisions without relying on
    # the kernel's particular hash function. Padding IDs must remain in the tail.
    heads = (torch.arange(head_len + 3, device=DEVICE, dtype=torch.int32) * 65536)
    heads = heads.expand(rows, -1).contiguous()
    draws = torch.randint(head_len + 1, (rows, m), device=DEVICE, dtype=torch.int32) * 65536 + 1
    hits = min(head_len, m // 3)
    draws[:, :hits] = heads[:, :hits]
    draws[:, -3:] = heads[:, head_len:]
    if head_len >= 17:
        heads[:, 2] = heads[:, 1]
        heads[0, 3] = -1
        heads[0, 4] = -(2 ** 31)
        draws[:, 3] = -1  # Present only in row 0; also the table's empty marker.
        draws[:, 4] = -(2 ** 31)
    corr = torch.randn((rows, m), device=DEVICE)
    corr[:, 5] = -float("inf")
    corr[:, 6] = float("nan")
    corr[:, 7] = float("inf")
    valid = torch.isfinite(corr) & ~(draws[:, :, None] == heads[:, None, :head_len]).any(-1)
    logits = q.float() @ keys.float().transpose(1, 2) / math.sqrt(dim) + corr[:, None]
    logits = logits.masked_fill(~valid[:, None], -float("inf"))
    if clip > 0:
        mean = (logits.masked_fill(~valid[:, None], 0).sum(-1, keepdim=True)
                / valid.sum(-1)[:, None, None].clamp_min(1))
        logits = torch.minimum(logits, mean + clip)
    weights = torch.cat((lse[..., None], logits), dim=-1).softmax(-1)
    expected = weights[..., :1] * main + weights[..., 1:] @ values.float()
    sampled_tail_attention_merge(q, keys, values, corr, output, lse, dim ** -0.5,
                                 clip, draws, heads, head_len)
    torch.testing.assert_close(output.float(), expected, atol=4e-3, rtol=8e-3)


def attention_oracle(cache, queries, layer, clip):
    """Dense q.k/V reference using actual sampled positions and current-layer head."""
    state = cache._sample_state[DEVICE]
    heads = cache.selected_indices_buffer[:, :cache.active_sparse_len_host].long()
    draws = state["draws"].long()
    keys = cache.cpu_key_cache[layer].to(DEVICE).float()
    values = cache.cpu_value_cache[layer].to(DEVICE).float()
    if cache.quantize_cpu_kv:
        keys *= cache.cpu_kv_k_scale[layer][:, None].to(DEVICE)
        values *= cache.cpu_kv_v_scale[layer].to(DEVICE).unsqueeze(-1)
    # Prompt recent is not in the retrieval store; read preserved KV from its source.
    main_keys = torch.cat((cache.static_keys[layer].reshape(cache.batch_groups, 36, 128).float(),
                           keys.gather(1, heads[..., None].expand(-1, -1, 128))), dim=1)
    main_values = torch.cat((cache.static_values[layer].reshape(cache.batch_groups, 36, 128).float(),
                             values.gather(1, heads[..., None].expand(-1, -1, 128))), dim=1)
    tail_keys = keys.gather(1, draws[..., None].expand(-1, -1, 128))
    tail_values = values.gather(1, draws[..., None].expand(-1, -1, 128))
    q = queries.reshape(cache.batch_groups, cache.group_size, 128).float()
    main_logits = q @ main_keys.transpose(1, 2) / math.sqrt(128)
    tail_logits = q @ tail_keys.transpose(1, 2) / math.sqrt(128) + state["corr"][:, None]
    valid = ~(draws[:, :, None] == heads[:, None, :]).any(-1)
    if clip > 0:
        cap = (tail_logits.masked_fill(~valid[:, None], 0).sum(-1, keepdim=True)
               / valid.sum(-1)[:, None, None].clamp_min(1)) + clip
        tail_logits = torch.minimum(tail_logits, cap)
    tail_logits.masked_fill_(~valid[:, None], -float("inf"))
    weights = torch.cat((main_logits, tail_logits), dim=-1).softmax(-1)
    return weights @ torch.cat((main_values, tail_values), dim=1)


@pytest.mark.parametrize("aggregation", ["q_sum", "mean_prob"])
@pytest.mark.parametrize("store,quant", [("cpu", "none"), ("gpu", "none"), ("cpu", "int8")])
def test_multilayer_tail_matches_current_head_oracle(monkeypatch, aggregation, store, quant):
    monkeypatch.setenv("COMETKV_SAMPLE_CLIP", "0.5")
    cache = make_cache(layers=2, query_aggregation=aggregation, kv_store_device=store, cpu_kv_quant=quant)
    query = torch.randn((1, 1, 8, 128), dtype=DTYPE, device=DEVICE)
    cache.sparse_attention(query, 0)
    state = cache._sample_state[DEVICE]
    draws, corr = state["draws"].clone(), state["corr"].clone()
    owner_head = cache.selected_indices_buffer[:, :cache.active_sparse_len_host].clone()
    # Tail-only proposal: the owner's head is removed from the softmax part, and the
    # uniform mix keeps it drawable for the next layer, whose head differs.
    assert torch.isneginf(state["prop"].gather(1, owner_head.long())).all()
    assert torch.isfinite(corr).all()
    query.normal_()
    output = cache.sparse_attention(query, 1)
    assert torch.equal(state["draws"], draws) and torch.equal(state["corr"], corr)
    assert not torch.equal(owner_head, cache.selected_indices_buffer[:, :cache.active_sparse_len_host])
    expected = attention_oracle(cache, query, 1, clip=0.5)
    torch.testing.assert_close(output.reshape_as(expected).float(), expected, atol=2e-2, rtol=2e-2)


def test_paper_tail_turns_on_in_a_running_cache_and_matches_oracle(monkeypatch):
    monkeypatch.setenv("COMETKV_SAMPLE_CLIP", "0.5")
    cache = make_cache(layers=2, prompt=8192, generation=2, sample_size=64,
                       tail_in_budget=True, sample_min_context=8193)
    assert cache.active_sample_len_host == 0
    assert cache.active_sparse_len_host == 163
    query = torch.randn((1, 1, 8, 128), dtype=DTYPE, device=DEVICE)
    cache.sparse_attention(query, 0)              # head-only step below the threshold
    cache.sample_min_context = 8192               # the context now reaches the threshold
    cache._update_retrieval_plan(cache.prompt_lengths)
    assert cache.active_sample_len_host == 64
    assert cache.active_sparse_len_host == 163 - 64
    cache.sparse_attention(query, 0)
    assert cache._sample_state[DEVICE]["m"] == 64
    query.normal_()
    output = cache.sparse_attention(query, 1)
    expected = attention_oracle(cache, query, 1, clip=0.5)
    torch.testing.assert_close(output.reshape_as(expected).float(), expected, atol=2e-2, rtol=2e-2)


def test_tail_only_proposal_draws_less_from_the_owner_head(monkeypatch):
    def owner_head_share(exclude):
        monkeypatch.setenv("COMETKV_SAMPLE_EXCLUDE_HEAD", exclude)
        torch.manual_seed(11)
        cache = make_cache(prompt=2048, generation=130, sample_size=256)
        torch.manual_seed(12)
        query = torch.randn((1, 1, 8, 128), dtype=DTYPE, device=DEVICE)
        cache.sparse_attention(query, 0)
        draws = cache._sample_state[DEVICE]["draws"].long()
        head = cache.selected_indices_buffer[:, :cache.active_sparse_len_host].long()
        return (draws[:, :, None] == head[:, None, :]).any(-1).float().mean().item()

    full_support, tail_only = owner_head_share("0"), owner_head_share("1")
    # Only the 10% uniform mix can still land in the head (~2% of the candidates).
    assert tail_only < 0.05
    assert tail_only < full_support


@pytest.mark.parametrize("topk_impl", ["torch", "radix"])
def test_cuda_graph_replays_new_queries_noise_and_heads(monkeypatch, topk_impl):
    monkeypatch.setenv("COMETKV_SAMPLE_CLIP", "0.5")
    monkeypatch.setenv("COMETKV_TOPK_IMPL", topk_impl)
    cache = make_cache(layers=2, sample_size=256)
    cache.use_cuda_graph = True
    cache._cg_cache_seqlens.fill_(36 + cache.active_sparse_len_host)
    queries = [torch.randn((1, 1, 8, 128), dtype=DTYPE, device=DEVICE) for _ in range(2)]
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            for layer in range(2):
                cache.sparse_attention(queries[layer], layer)
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cache.sparse_attention(queries[0], 0)
        output = cache.sparse_attention(queries[1], 1)
    state = cache._sample_state[DEVICE]
    for _ in range(3):
        queries[0].normal_()
        queries[1].normal_()
        state["noise"].uniform_()
        graph.replay()
        expected = attention_oracle(cache, queries[1], 1, clip=0.5)
        torch.testing.assert_close(output.reshape_as(expected).float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("env,expected", [
    ({"COMETKV_TOKEN_CACHE_ROUND": "1024"}, 1024),   # legacy granularity: 2 x 174 = 348 -> 1024
    ({"COMETKV_TOKEN_CACHE_ROUND": "64"}, 384),
    ({"COMETKV_TOKEN_CACHE_SLOTS": "512"}, 512),     # explicit capacity, MULT ignored
    ({"COMETKV_TOKEN_CACHE_SLOTS": "512", "COMETKV_TOKEN_CACHE_MULT": "1"}, 512),
])
def test_token_cache_rounding_and_explicit_slots(monkeypatch, env, expected):
    monkeypatch.setenv("COMETKV_TOKEN_CACHE_MULT", "2")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    cache = make_cache(prompt=8192, generation=514)
    assert cache.sig_token_cache_size == expected
    assert all(ids.size(1) == expected for ids in cache.token_cache_ids)
