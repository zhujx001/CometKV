"""Regressions for sealed-block centering and independent GQA query scoring."""

import math

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CometKV block/GQA tests require CUDA", allow_module_level=True)

from cache_hub.cometkv_cache import cometkv_cache


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    for name in ("COMETKV_STATS_MODE", "COMETKV_QUERY_AGG", "COMETKV_NO_KEY_CENTER",
                 "COMETKV_SAMPLE_AUTOSCALE", "COMETKV_SAMPLE_FRAC", "COMETKV_SAMPLE_SIZE",
                 "COMETKV_SCORE_IMPL", "COMETKV_TOPK_IMPL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COMETKV_TOKEN_CACHE_MULT", "1")
    torch.manual_seed(123)


def make_cache(prompt=64, group=4, batch=1, layers=1, **overrides):
    parameters = dict(
        valid_start=np.zeros(batch, dtype=np.int32), layer_num=layers,
        batch_size=batch, max_length=prompt + 512, max_new_length=512,
        num_key_value_heads=2, num_heads=2 * group, head_dim=128, dtype=DTYPE,
        layer_mapping={str(layer): DEVICE for layer in range(layers)},
        static_pattern_start=4, static_pattern_end=32, retrieval_budget=0.1,
        sig_bits=128, sig_topk=8, sig_min_retrieval_topk=0,
        sig_chunk_size=16384, sig_seed=1234, sig_mode="random_orth",
        sig_token_cache_size=16, prefill_bsz=batch, num_gpus=1, model_size=8,
    )
    parameters.update(overrides)
    cache = cometkv_cache(**parameters)
    prompt_keys = []
    for layer in range(layers):
        keys = torch.randn(batch, prompt, 2, 128, device=DEVICE, dtype=DTYPE)
        values = torch.randn_like(keys)
        queries = torch.randn(batch, prompt, 2 * group, 128, device=DEVICE, dtype=DTYPE)
        cache.prefill_update_kv_cache(queries, keys, values, layer, 0)
        prompt_keys.append(keys.transpose(1, 2).reshape(batch * 2, prompt, 128).float())
    cache.prepare_cache()
    torch.cuda.synchronize()
    return cache, prompt_keys


def seal(cache, start=0, count=96, layer=0, shift=12.0):
    keys = torch.randn(cache.batch_size, cache.kv_head, count, 128, device=DEVICE, dtype=DTYPE)
    keys = keys + shift
    values = torch.randn_like(keys)
    cache.decode_hot_keys[layer][:, :, :count].copy_(keys)
    cache.decode_hot_values[layer][:, :, :count].copy_(values)
    cache._update_lockstep_evicted_retrieval(layer, start, count)
    return keys.reshape(cache.batch_groups, count, 128).float()


def reconstructed_keys(cache, layer, start, end):
    """Decode virtual keys, then use ordinary dense dot products as the oracle."""
    packed = cache.signature_index[layer].reshape(cache.batch_groups, -1, cache.sig_bytes)[:, start:end]
    shifts = torch.arange(8, device=DEVICE, dtype=torch.uint8)
    bits = ((packed.unsqueeze(-1) >> shifts) & 1).flatten(-2)[..., :cache.asym_sig_bits]
    signs = bits.float() * 2 - 1
    positions = torch.arange(start, end, device=DEVICE)
    if cache.block_stats:
        epochs = torch.where(
            positions < cache.lockstep_prompt_length_host, 0,
            1 + (positions - cache.lockstep_prompt_length_host + cache._fixed_prompt_local_recent_overlap())
            // cache._fixed_prompt_local_slide_stride(),
        )
    else:
        epochs = torch.zeros_like(positions)
    lo = cache.sig_block_norm_lo[layer][:, epochs]
    step = cache.sig_block_norm_step[layer][:, epochs]
    norms = (lo + packed[..., -1].float() * step).exp()
    reconstructed = norms[..., None] * (signs @ cache.base_projection[:cache.asym_sig_bits].to(DEVICE))
    reconstructed /= cache.signature_score_scale
    return reconstructed + cache.sig_block_centers[layer][:, epochs]


def expected_scores(cache, queries, layer=0):
    start, end = cache.plan_range1_start_host, cache.plan_range1_end_host
    keys = reconstructed_keys(cache, layer, start, end)
    q = queries.reshape(cache.batch_groups, cache.group_size, 128).float()
    logits = torch.einsum("hgd,htd->hgt", q, keys) / math.sqrt(128)
    if cache.query_aggregation == "q_sum":
        return logits.sum(1) * math.sqrt(128) * cache.signature_score_scale
    return torch.logsumexp(logits.log_softmax(-1), dim=1) - math.log(cache.group_size)


@pytest.mark.parametrize("group,sig_bits", [(1, 128), (2, 64), (4, 128), (7, 128), (8, 128)])
@pytest.mark.parametrize("score_impl", ["scalar", "lookup"])
def test_grouped_scores_match_dense_virtual_keys_across_blocks(group, sig_bits, score_impl, monkeypatch):
    monkeypatch.setenv("COMETKV_SCORE_IMPL", score_impl)
    cache, _ = make_cache(group=group, batch=2, sig_bits=sig_bits)
    seal(cache)
    seal(cache, start=96, count=128, shift=-7.0)
    cache._update_retrieval_plan(cache.prompt_lengths + 257)
    queries = torch.randn(2, 1, 2 * group, 128, device=DEVICE, dtype=DTYPE)
    cache._grouped_query_asym_topk(queries, 0)
    start, end = cache.plan_range1_start_host, cache.plan_range1_end_host
    actual = cache._asym_scores_buffers[DEVICE][:, start:end]
    expected = expected_scores(cache, queries)
    torch.testing.assert_close(actual, expected, atol=4e-4, rtol=2e-4)
    indices = cache.selected_indices_buffer[:, :cache.active_sparse_len_host].long()
    oracle = expected.topk(cache.active_sparse_len_host, dim=1).indices + start
    assert torch.equal(indices.sort(1).values, oracle.sort(1).values)
    torch.testing.assert_close(actual.exp().sum(1), torch.ones(cache.batch_groups, device=DEVICE))


@pytest.mark.parametrize("center", [True, False])
def test_block_stats_freeze_old_signatures_and_cover_new_norms(monkeypatch, center):
    if not center:
        monkeypatch.setenv("COMETKV_NO_KEY_CENTER", "1")
    cache, prompt = make_cache()
    old_signatures = cache.signature_index[0][:, :, :64].clone()
    old_lo = cache.sig_norm_lo[0].clone()
    old_step = cache.sig_norm_step[0].clone()
    block = seal(cache)
    mean = block.mean(1) if center else torch.zeros_like(block[:, 0])
    residual = block - mean[:, None]
    logs = residual.norm(dim=-1).clamp_min(1e-6).log()
    if center:
        torch.testing.assert_close(cache.sig_block_centers[0][:, 1], mean - prompt[0].mean(1))
    torch.testing.assert_close(cache.sig_block_norm_lo[0][:, 1], logs.amin(1))
    torch.testing.assert_close(cache.sig_block_norm_step[0][:, 1], (logs.amax(1) - logs.amin(1)) / 255)
    snapshot = cache.sig_block_centers[0][:, 1].clone()
    seal(cache, start=96, count=128, shift=-15.0)
    assert torch.equal(old_signatures, cache.signature_index[0][:, :, :64])
    assert torch.equal(old_lo, cache.sig_norm_lo[0])
    assert torch.equal(old_step, cache.sig_norm_step[0])
    assert torch.equal(snapshot, cache.sig_block_centers[0][:, 1])


def test_opposite_queries_do_not_collapse_to_a_zero_query():
    cache, _ = make_cache(group=2, prompt=128)
    query = cache.base_projection[0].to(DEVICE).bfloat16() * 16
    queries = torch.stack((query, -query, query, -query)).reshape(1, 1, 4, 128)
    assert torch.count_nonzero(queries.reshape(2, 2, 128).sum(1)) == 0
    cache._grouped_query_asym_topk(queries, 0)
    scores = cache._asym_scores_buffers[DEVICE][:, 4:96].clone()
    assert torch.all(scores.std(1) > 0.01)
    torch.testing.assert_close(scores, expected_scores(cache, queries), atol=2e-4, rtol=2e-4)
    reversed_queries = queries.reshape(1, 1, 2, 2, 128).flip(3).reshape_as(queries)
    cache._grouped_query_asym_topk(reversed_queries, 0)
    torch.testing.assert_close(scores, cache._asym_scores_buffers[DEVICE][:, 4:96])


def test_block_compensation_q_sum_ablation_uses_the_same_score_scale():
    cache, _ = make_cache(query_aggregation="q_sum")
    seal(cache)
    cache._update_retrieval_plan(cache.prompt_lengths + 129)
    queries = torch.randn(1, 1, 8, 128, device=DEVICE, dtype=DTYPE)
    cache._grouped_query_asym_topk(queries, 0)
    start, end = cache.plan_range1_start_host, cache.plan_range1_end_host
    torch.testing.assert_close(cache._asym_scores_buffers[DEVICE][:, start:end], expected_scores(cache, queries),
                               atol=3e-3, rtol=3e-4)


@pytest.mark.parametrize("aggregation", ["q_sum", "mean_prob"])
def test_runtime_radix_selection_matches_same_scores_and_preserves_budget(aggregation):
    cache, _ = make_cache(prompt=16384, query_aggregation=aggregation, sig_topk=0,
                          retrieval_budget=.02)
    cache.exact_topk_impl = "radix"
    query = torch.randn((1, 1, 8, 128), device=DEVICE, dtype=DTYPE)
    cache._grouped_query_asym_topk(query, 0)
    k = cache.active_sparse_len_host
    scores = cache._asym_scores_buffers[DEVICE]
    selected = cache.selected_indices_buffer[:, :k].long()
    assert selected.size(1) == int(16384 * .02)
    expected = scores.argsort(dim=1, descending=True, stable=True)[:, :k].sort(1).values
    assert torch.equal(selected, expected)
    assert (cache.selected_indices_buffer[:, k:] == -1).all()


def test_radix_fallback_preserves_k_above_kernel_limit():
    cache, _ = make_cache(prompt=8192, sig_topk=4100)
    cache.exact_topk_impl = "radix"
    query = torch.randn((1, 1, 8, 128), device=DEVICE, dtype=DTYPE)
    cache._grouped_query_asym_topk(query, 0)
    assert cache.active_sparse_len_host == 4100
    actual = cache.selected_indices_buffer[:, :4100].long().sort(1).values
    expected = cache._asym_scores_buffers[DEVICE].topk(4100, dim=1).indices.sort(1).values
    assert torch.equal(actual, expected)


def test_grouped_scoring_cuda_graph_replays_new_queries_after_eviction():
    cache, _ = make_cache()
    seal(cache)
    cache._update_retrieval_plan(cache.prompt_lengths + 129)
    queries = torch.randn(1, 1, 8, 128, device=DEVICE, dtype=DTYPE)
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            cache._grouped_query_asym_topk(queries, 0)
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cache._grouped_query_asym_topk(queries, 0)
    queries.normal_()
    graph.replay()
    start, end = cache.plan_range1_start_host, cache.plan_range1_end_host
    torch.testing.assert_close(cache._asym_scores_buffers[DEVICE][:, start:end], expected_scores(cache, queries),
                               atol=4e-4, rtol=2e-4)


@pytest.mark.parametrize("quant", ["none", "int8"])
def test_full_recompute_reads_only_the_layer_that_has_finished_writing(quant):
    cache, _ = make_cache(layers=2, stats_mode="frozen", full_recompute_interval=96,
                          kv_store_device="gpu", cpu_kv_quant=quant)
    original_mean = cache.sig_key_mean[1].clone()
    if quant == "none":
        cache.cpu_key_cache[1][:, 64:160].fill_(float("nan"))
    seal(cache, layer=0, shift=0.25)
    assert torch.equal(cache.sig_key_mean[1], original_mean)
    if quant == "none":
        assert cache.cpu_key_cache[1][:, 64:160].isnan().all()
    seal(cache, layer=1, shift=0.5)
    for layer in range(2):
        keys = cache.cpu_key_cache[layer][:, :160].float()
        if quant == "int8":
            keys = keys * cache.cpu_kv_k_scale[layer][:, None]
        torch.testing.assert_close(cache.sig_key_mean[layer], keys.mean(1))
        assert cache.sig_norm_lo[layer].isfinite().all()


def test_incompatible_statistic_updates_fail_explicitly():
    with pytest.raises(ValueError, match="mean_update_alpha"):
        make_cache(mean_update_alpha=0.1)
    with pytest.raises(ValueError, match="Full recompute"):
        make_cache(full_recompute_interval=128)


def test_mean_prob_tail_proposal_uses_log_mixture_probabilities(monkeypatch):
    monkeypatch.setenv("COMETKV_SAMPLE_MIN_M", "0")
    cache, _ = make_cache(prompt=128, sample_frac=0.25, sig_topk=16)
    assert not cache.sample_autoscale
    queries = torch.randn(1, 1, 8, 128, device=DEVICE, dtype=DTYPE)
    cache.sparse_attention(queries, 0)
    state = cache._sample_state[DEVICE]
    scores = cache._asym_scores_buffers[DEVICE].clone()
    probabilities = (scores / cache.sample_tau).softmax(1)
    probabilities.mul_(1 - cache.sample_uniform_mix)
    probabilities[:, cache.plan_range1_start_host:cache.plan_range1_end_host].add_(
        cache.sample_uniform_mix / cache.active_candidates_host)
    drawn_probabilities = probabilities.gather(1, state["draws"].long())
    expected = -(cache.active_sample_len_host * drawn_probabilities).log()
    torch.testing.assert_close(state["corr"], expected, atol=2e-5, rtol=2e-5)
