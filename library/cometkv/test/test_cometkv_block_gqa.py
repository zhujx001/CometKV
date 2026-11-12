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
    # The score oracles below model the 120-bit scorer without query centering (the default);
    # tests that exercise query centering opt in explicitly.
    monkeypatch.setenv("COMETKV_QCENTER", "off")
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
        query_aggregation="mean_prob",  # These probability-oracle tests opt in explicitly.
        # Tiny prompts: keep the tail on (additional accounting, no context threshold) so the
        # proposal oracles still run; paper accounting is covered in test_cometkv_tail_budget.py.
        tail_in_budget=False, sample_min_context=0,
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
@pytest.mark.parametrize("stats_mode", ["block", "mu_frozen"])
def test_block_stats_freeze_old_signatures_and_cover_new_norms(monkeypatch, center, stats_mode):
    if not center:
        monkeypatch.setenv("COMETKV_NO_KEY_CENTER", "1")
    cache, prompt = make_cache(stats_mode=stats_mode)
    old_signatures = cache.signature_index[0][:, :, :64].clone()
    old_lo = cache.sig_norm_lo[0].clone()
    old_step = cache.sig_norm_step[0].clone()
    block = seal(cache)
    # Block-0 statistics = the 64-token prompt minus the 4 sink tokens.
    prompt_mean = prompt[0][:, 4:].mean(1)
    mean = (block.mean(1) if stats_mode == "block" else prompt_mean) if center else torch.zeros_like(block[:, 0])
    residual = block - mean[:, None]
    logs = residual.norm(dim=-1).clamp_min(1e-6).log()
    if center:
        torch.testing.assert_close(cache.sig_block_centers[0][:, 1], mean - prompt_mean)
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


@pytest.mark.parametrize("stats_mode", ["block", "mu_frozen"])
def test_block_compensation_q_sum_ablation_uses_the_same_score_scale(stats_mode):
    cache, _ = make_cache(query_aggregation="q_sum", stats_mode=stats_mode)
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


@pytest.mark.parametrize("stats_mode", ["block", "mu_frozen"])
@pytest.mark.parametrize("aggregation", ["q_sum", "mean_prob"])
def test_grouped_scoring_cuda_graph_replays_new_queries_after_eviction(stats_mode, aggregation):
    cache, _ = make_cache(stats_mode=stats_mode, query_aggregation=aggregation)
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
                               atol=3e-3 if aggregation == "q_sum" else 4e-4,
                               rtol=3e-4 if aggregation == "q_sum" else 2e-4)


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
    # Tail-only proposal: the softmax part excludes the owner layer's head.
    head = cache.selected_indices_buffer[:, :cache.active_sparse_len_host].long()
    scores.scatter_(1, head.clamp_min(0), torch.where(
        head >= 0, torch.full_like(head, float("-inf"), dtype=scores.dtype), scores.gather(1, head.clamp_min(0))))
    probabilities = (scores / cache.sample_tau).softmax(1)
    probabilities.mul_(1 - cache.sample_uniform_mix)
    probabilities[:, cache.plan_range1_start_host:cache.plan_range1_end_host].add_(
        cache.sample_uniform_mix / cache.active_candidates_host)
    drawn_probabilities = probabilities.gather(1, state["draws"].long())
    expected = -(cache.active_sample_len_host * drawn_probabilities).log()
    torch.testing.assert_close(state["corr"], expected, atol=2e-5, rtol=2e-5)


def test_symmetric_scores_equal_sign_hamming_oracle(monkeypatch):
    monkeypatch.setenv('COMETKV_SCORER', 'sym')
    monkeypatch.setenv('COMETKV_NO_KEY_CENTER', '1')
    cache, _ = make_cache(prompt=256, batch=2)
    queries = torch.randn(2, 1, 8, 128, device=DEVICE, dtype=DTYPE)
    cache._grouped_query_asym_topk(queries, 0)
    start, end = cache.plan_range1_start_host, cache.plan_range1_end_host
    packed = cache.signature_index[0].reshape(cache.batch_groups, -1, cache.sig_bytes)[:, start:end]
    shifts = torch.arange(8, device=DEVICE, dtype=torch.uint8)
    key_bits = ((packed.unsqueeze(-1) >> shifts) & 1).flatten(-2)[..., :cache.asym_sig_bits]
    grouped = queries.reshape(cache.batch_groups, cache.group_size, 128).float()
    query_bits = (grouped @ cache.base_projection.to(DEVICE).T)[..., :cache.asym_sig_bits] >= 0
    distance = (key_bits[:, None].bool() != query_bits[:, :, None]).sum(-1)
    logits = (cache.asym_sig_bits - 2 * distance).float() * cache.RSQRT_DIM / cache.signature_score_scale
    expected = torch.logsumexp(logits.log_softmax(-1), dim=1) - math.log(cache.group_size)
    actual = cache._asym_scores_buffers[DEVICE][:, start:end]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_cache_free_uva_gathers_identical_keys_and_values(monkeypatch):
    cache, _ = make_cache(prompt=256)
    queries = torch.randn(1, 1, 8, 128, device=DEVICE, dtype=DTYPE)
    cache._grouped_query_asym_topk(queries, 0)
    before = cache._build_static_fixed_concat_flash_kv_from_indices(0, cache.selected_indices_buffer)
    expected_k, expected_v = before[0].clone(), before[1].clone()
    cache.disable_token_cache = True
    after = cache._build_static_fixed_concat_flash_kv_from_indices(0, cache.selected_indices_buffer)
    torch.testing.assert_close(after[0], expected_k, rtol=0, atol=0)
    torch.testing.assert_close(after[1], expected_v, rtol=0, atol=0)
    assert torch.count_nonzero(cache.hit_mask_buffer) == 0


def test_cache_free_mode_does_not_allocate_token_cache(monkeypatch):
    monkeypatch.setenv('COMETKV_DISABLE_TOKEN_CACHE', '1')
    cache, _ = make_cache(prompt=256)
    assert cache.token_cache_ids is None
    assert cache.token_cache_keys is None
    queries = torch.randn(1, 1, 8, 128, device=DEVICE, dtype=DTYPE)
    assert cache.sparse_attention(queries, 0).isfinite().all()


def test_tail_eager_and_graph_driver_draw_same_independent_windows(monkeypatch):
    monkeypatch.setenv('COMETKV_SAMPLE_STRIDE', '1')
    caches=[]
    for _ in range(2):
        torch.manual_seed(123)
        cache, _ = make_cache(prompt=256, layers=2, sample_mode="size", sample_size=32)
        cache._sample_state_for(0, torch.device(DEVICE))
        caches.append(cache)
    queries = torch.randn(1, 1, 8, 128, device=DEVICE, dtype=DTYPE)
    for step in range(3):
        drawn=[]
        for mode, cache in enumerate(caches):
            cache.use_cuda_graph = bool(mode)
            if mode: cache.cg_set_step_inputs()
            windows=[]
            for layer in range(2):
                cache._grouped_query_asym_topk(queries, layer)
                output=torch.zeros_like(queries)
                lse=torch.zeros((1,8,1),device=DEVICE,dtype=torch.float32)
                cache._sampled_tail_attention(queries,layer,output,lse)
                windows.append(cache._sample_state[DEVICE]['draws'].clone())
            drawn.append(windows)
        for layer in range(2):
            torch.testing.assert_close(drawn[0][layer], drawn[1][layer], rtol=0, atol=0)
        assert not torch.equal(drawn[0][0], drawn[0][1])


def test_tail_update_waits_for_each_layer_to_publish_new_cpu_keys():
    cache, _ = make_cache(prompt=128, layers=2, sample_mode="size", sample_size=64)
    cache.cpu_kv_cache[1][:, 128:224].fill_(float('nan'))
    seal(cache, layer=0, shift=0.5)
    cache._update_retrieval_plan(cache.prompt_lengths + 129)
    cache._tail_index_update_step = True
    queries = torch.randn(1, 1, 8, 128, device=DEVICE, dtype=DTYPE)
    for layer in range(2):
        if layer == 1:
            seal(cache, layer=1, shift=0.5)
        cache._grouped_query_asym_topk(queries, layer)
        output = torch.zeros_like(queries)
        lse = torch.zeros((1,8,1),device=DEVICE,dtype=torch.float32)
        cache._sampled_tail_attention(queries, layer, output, lse)
        assert (cache._sample_state[DEVICE]['draws'] >= 128).any()
        # The layer-0 window must not prefetch layer 1 before its seal above.
        assert output.isfinite().all()


@pytest.mark.parametrize("qcenter", ["fp", "8bit"])
def test_query_centered_scoring_cuda_graph_replay_matches_eager(qcenter, monkeypatch):
    monkeypatch.setenv("COMETKV_QCENTER", qcenter)
    cache, _ = make_cache(query_aggregation="q_sum")
    assert cache.qcenter == qcenter
    seal(cache)   # block-sealed keys get their own q_bias entries
    cache._update_retrieval_plan(cache.prompt_lengths + 129)
    start, end = cache.plan_range1_start_host, cache.plan_range1_end_host
    qbias = cache.q_bias[0] if cache.q_bias is not None else cache._qbias_from_signatures(0)
    assert qbias[:, start:end].abs().sum() > 0
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
    torch.cuda.synchronize()
    replayed = cache._asym_scores_buffers[DEVICE][:, start:end].clone()
    replayed_sel = cache.selected_indices_buffer[:, :cache.active_sparse_len_host].clone()
    cache._grouped_query_asym_topk(queries, 0)
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed, cache._asym_scores_buffers[DEVICE][:, start:end])
    assert torch.equal(replayed_sel.sort(1).values,
                       cache.selected_indices_buffer[:, :cache.active_sparse_len_host].sort(1).values)


@pytest.mark.parametrize("score_impl", ["lookup", "scalar"])
def test_16b_signature_scores_match_the_fp32_buffer_path(monkeypatch, score_impl):
    # 16 B/token: the scorer decodes c from signature byte 14 with per-block ranges. Adding the
    # same decoded c from a per-token fp32 buffer after the kernel must give identical scores.
    monkeypatch.setenv("COMETKV_QCENTER", "8bit")
    monkeypatch.setenv("COMETKV_SCORE_IMPL", score_impl)
    cache, _ = make_cache(query_aggregation="q_sum")
    assert cache.qbias_in_signature and cache.q_bias is None
    seal(cache)
    seal(cache, start=96, count=128, shift=-7.0)
    cache._update_retrieval_plan(cache.prompt_lengths + 257)
    queries = torch.randn(1, 1, 8, 128, device=DEVICE, dtype=DTYPE)
    cache._grouped_query_asym_topk(queries, 0)
    actual = cache._asym_scores_buffers[DEVICE].clone()
    selected = cache.selected_indices_buffer[:, :cache.active_sparse_len_host].sort(1).values
    decoded = cache._qbias_from_signatures(0)
    cache.qbias_in_signature, cache.q_bias = False, [decoded]
    cache._grouped_query_asym_topk(queries, 0)
    assert torch.equal(actual, cache._asym_scores_buffers[DEVICE])
    assert torch.equal(selected, cache.selected_indices_buffer[:, :cache.active_sparse_len_host].sort(1).values)


def test_16b_sealed_block_c_decodes_within_half_a_step(monkeypatch):
    monkeypatch.setenv("COMETKV_QCENTER", "8bit")
    cache, _ = make_cache(query_aggregation="q_sum")
    block = seal(cache)                       # tokens [prompt, prompt + 96), block statistics
    start = cache.lockstep_prompt_length_host
    exact = torch.einsum("rtd,rd->rt", block - block.mean(1, keepdim=True), cache.q_mean[0])
    decoded = cache._qbias_from_signatures(0)[:, start:start + block.size(1)]
    step = (exact.amax(1, keepdim=True) - exact.amin(1, keepdim=True)) / 255.0
    assert ((decoded - exact).abs() <= step * 0.5 + 1e-3).all()
