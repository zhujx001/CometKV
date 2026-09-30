import importlib
import math
import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CometKV runtime tests require a CUDA GPU.", allow_module_level=True)

from cache_hub.cometkv_cache import cometkv_cache
from model_hub.LLM import LLM


cometkv_cache_module = importlib.import_module("cache_hub.cometkv_cache")


DTYPE = torch.bfloat16
DEVICE = "cuda:0"


class FakeEvent:
    def __init__(self):
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True


def make_cache(layer_num=2, **overrides):
    params = dict(
        valid_start=np.array([0], dtype=np.int32),
        layer_num=layer_num,
        batch_size=1,
        max_length=32,
        num_key_value_heads=2,
        num_heads=4,
        head_dim=128,
        dtype=DTYPE,
        layer_mapping={str(i): DEVICE for i in range(layer_num)},
        max_new_length=8,
        static_pattern_start=4,
        static_pattern_end=4,
        retrieval_budget=0.018,
        sig_bits=128,
        sig_topk=0,
        sig_chunk_size=16384,
        sig_seed=1234,
        sig_mode="random_orth",
        sig_token_cache_size=16,
        prefill_bsz=1,
        num_gpus=1,
        model_size=8,
    )
    params.update(overrides)
    return cometkv_cache(
        **params,
    )


def build_reference_random_orth_projection(sig_bits=128, head_dim=128, sig_seed=1234):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sig_seed)
    basis = torch.randn((head_dim, head_dim), generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(basis, mode="reduced")
    return q[:, :sig_bits].t().contiguous()


def pack_reference(vectors, projection, norm_margin=0.0):
    valid_length = vectors.shape[-2]
    leading_shape = vectors.shape[:-2]
    row_vectors = vectors.reshape(-1, valid_length, vectors.shape[-1]).contiguous()
    vectors_fp32 = row_vectors.float()
    prompt_mean = vectors_fp32.sum(dim=1, keepdim=True) / max(valid_length, 1)
    centered = vectors_fp32 - prompt_mean
    projected = torch.matmul(centered, projection.t())
    bits = (projected >= 0).to(torch.uint8)
    asym_sig_bits = projection.shape[0] - 8
    bits[..., asym_sig_bits:] = 0
    bit_weights = (1 << torch.arange(8, device=bits.device, dtype=torch.uint8)).view(*([1] * (bits.dim() - 1)), 8)
    packed = (bits.view(*bits.shape[:-1], -1, 8) * bit_weights).sum(dim=-1).to(torch.uint8)

    log_norms = centered.norm(dim=-1).clamp_min(1e-6).log()
    lo = log_norms.min(dim=-1).values
    hi = log_norms.max(dim=-1).values
    if norm_margin > 0:
        span = (hi - lo).clamp_min(1e-6)
        lo = lo - norm_margin * span
        hi = hi + norm_margin * span
    step = ((hi - lo) / 255.0).clamp_min(1e-8)
    norm_codes = torch.round((log_norms - lo.unsqueeze(-1)) / step.unsqueeze(-1))
    packed[..., -1] = norm_codes.clamp(0, 255).to(torch.uint8)
    return packed.view(*leading_shape, valid_length, packed.shape[-1]).contiguous()


class DummyLLM(LLM):
    def init_kv_cache(self, valid_start, attn_config):
        raise AssertionError("CometKV padded inputs should be rejected before cache initialization")

    def inference(self, *args, **kwargs):
        raise AssertionError("CometKV padded inputs should be rejected before inference")


def test_sync_waits_for_prefill_copy_and_releases_temp_refs():
    cache = make_cache()
    fake = FakeEvent()
    cache.prefill_copy_events = {0: [fake]}
    cache.prefill_copy_refs = {0: [object()]}

    assert not hasattr(cache, "use_pred_q")

    cache.sync(0, 0)

    assert fake.synchronized
    assert cache.prefill_copy_events[0] == []
    assert cache.prefill_copy_refs[0] == []



def test_prepare_cache_waits_for_prefill_copy_events():
    cache = make_cache(max_new_length=5)
    first = FakeEvent()
    second = FakeEvent()
    cache.prefill_copy_events = {0: [first], 1: [second]}
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)

    cache.prepare_cache()

    assert first.synchronized
    assert second.synchronized
    assert hasattr(cache, "range_starts_buffer")
    assert cache.range_starts_buffer.shape == (cache.batch_groups, 2)
    assert cache.topk_buffer.shape == (cache.batch_groups,)
    # topk is clamped to the candidate count: with prompt=12, static 4/4 and the max-decode
    # retrieval window of 8 candidates, the sig_min_retrieval_topk=16 floor is capped to 8 so no
    # empty (-1) slots are created (they would be zero-filled into the attended window -> dilution).
    assert cache.selected_indices_buffer.shape[1] == 8
    assert cache.use_static_prompt_retrieval is True
    assert cache.range_starts_buffer[0, 0].item() == 4
    assert cache.range_ends_buffer[0, 0].item() == 8


def test_cometkv_fast_only_rejects_varlen_valid_start():
    cache = make_cache(batch_size=2, valid_start=np.array([0, 3], dtype=np.int32))
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)

    with pytest.raises(AssertionError, match="lockstep"):
        cache.prepare_cache()


def test_cometkv_fast_only_rejects_padded_same_start_batch():
    cache = make_cache(batch_size=2, valid_start=np.array([3, 3], dtype=np.int32))
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)

    with pytest.raises(AssertionError, match="valid_start == 0"):
        cache.prepare_cache()


def test_cometkv_generate_rejects_padded_batch_before_cache_initialization():
    llm = DummyLLM("dummy", max_length=16, dtype=DTYPE, device_map=DEVICE)
    input_ids = torch.ones((2, 5), dtype=torch.int64, device=DEVICE)
    attention_masks = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=torch.int64,
        device=DEVICE,
    )

    with pytest.raises(ValueError, match="same-length inputs"):
        llm.generate(
            attention_type="CometKV",
            inputs_ids=input_ids,
            attention_masks=attention_masks,
            max_new_length=1,
            attn_config={"CometKV": {}},
        )


def test_cometkv_fast_path_constructor_rejects_removed_modes():
    removed_kwargs = [
        {"sig_token_cache_lru": True},
        {"sig_token_cache_protected_lru": True},
        {"sig_token_cache_protect_window": 2},
        {"sig_token_cache_layer_ratios": [1.0]},
        {"sig_union3_prefetch_ratio": 0.5},
        {"sig_cache_admission_distance_slack": 0},
        {"sig_cache_admission_topk_ratio": 0.5},
        {"sig_prefill_sync_mode": "window"},
        {"sig_prefill_async_window": 4},
        {"sig_use_shared_q": False},
        {"sig_gpu_recent_tokens": 8},
        {"sig_gpu_index": True},
        {"sig_decode_backend": "uva"},
        {"sig_use_block_rep": False},
        {"sig_retrieval_scope": "middle"},
        {"sig_flash_threshold": 256},
        {"sig_page_size": 8},
        {"spill_staging_tokens": 256},
    ]
    for kwargs in removed_kwargs:
        with pytest.raises(TypeError):
            make_cache(**kwargs)


def test_same_length_batch_prefill_writes_independent_rows_and_signatures():
    cache = make_cache(
        layer_num=1,
        batch_size=2,
        valid_start=np.array([0, 0], dtype=np.int32),
        prefill_bsz=2,
        max_new_length=5,
    )
    seq_len = 12
    query_states = torch.zeros((2, seq_len, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)
    key_states = torch.empty((2, seq_len, cache.kv_head, cache.head_dim), dtype=DTYPE, device=DEVICE)
    value_states = torch.empty_like(key_states)
    for batch_idx in range(2):
        for head_idx in range(cache.kv_head):
            for token_idx in range(seq_len):
                key_states[batch_idx, token_idx, head_idx, :].fill_(
                    batch_idx * 1000 + head_idx * 100 + token_idx
                )
                value_states[batch_idx, token_idx, head_idx, :].fill_(
                    batch_idx * 1000 + head_idx * 100 + token_idx + 10000
                )

    returned_keys, returned_values = cache.prefill_update_kv_cache(
        query_states,
        key_states,
        value_states,
        layer_idx=0,
        start_bdx=0,
    )
    cache.sync(0, 0)

    expected_key_rows = key_states.transpose(1, 2).reshape(cache.batch_groups, seq_len, cache.head_dim)
    expected_value_rows = value_states.transpose(1, 2).reshape(cache.batch_groups, seq_len, cache.head_dim)
    projection = build_reference_random_orth_projection().to(device=DEVICE)
    expected_signatures = pack_reference(expected_key_rows, projection)

    assert returned_keys.shape == (2, seq_len, cache.kv_head, cache.head_dim)
    assert returned_values.shape == (2, seq_len, cache.kv_head, cache.head_dim)
    assert torch.equal(cache.prompt_lengths, torch.full((2,), seq_len, dtype=torch.int32, device=DEVICE))
    assert torch.equal(cache.visible_lengths, torch.full((2,), seq_len, dtype=torch.int32, device=DEVICE))
    assert cache.context == seq_len
    assert torch.equal(cache.cpu_kv_cache[0][:, :seq_len, 0, :].to(DEVICE), expected_key_rows)
    assert torch.equal(cache.cpu_kv_cache[0][:, :seq_len, 1, :].to(DEVICE), expected_value_rows)
    assert torch.equal(cache.static_keys[0][:, :, :4, :], key_states[:, :4].transpose(1, 2))
    assert torch.equal(cache.static_values[0][:, :, :4, :], value_states[:, :4].transpose(1, 2))
    assert torch.equal(cache.prompt_recent_keys[0], key_states[:, 8:12].transpose(1, 2))
    assert torch.equal(cache.prompt_recent_values[0], value_states[:, 8:12].transpose(1, 2))
    assert torch.equal(cache.static_keys[0][:, :, 4:8, :], key_states[:, 8:12].transpose(1, 2))
    assert torch.equal(cache.static_values[0][:, :, 4:8, :], value_states[:, 8:12].transpose(1, 2))
    assert torch.equal(
        cache.signature_index[0][:, :, :seq_len, :].reshape(cache.batch_groups, seq_len, cache.sig_bytes),
        expected_signatures,
    )


def test_cometkv_fast_only_requires_static_prompt_retrieval():
    cache = make_cache(static_pattern_start=4, static_pattern_end=32)
    cache.prompt_lengths.fill_(2)
    cache.visible_lengths.fill_(2)
    cache.layer_visible_lengths.fill_(2)

    with pytest.raises(AssertionError, match="static prompt retrieval"):
        cache.prepare_cache()


def test_retrieval_budget_excludes_sink_and_recent_tokens_by_default():
    cache = make_cache(
        layer_num=1,
        max_length=32768,
        max_new_length=1,
        static_pattern_start=4,
        static_pattern_end=32,
        retrieval_budget=0.02,
        sig_token_cache_size=1024,
    )
    cache.prompt_lengths.fill_(32768)
    cache.visible_lengths.fill_(32768)
    cache.layer_visible_lengths.fill_(32768)

    cache.prepare_cache()

    # floor(32768 * 0.02) = 655 sparse retrieved tokens; sink/recent are direct attention.
    assert torch.equal(cache.topk_buffer, torch.full((cache.batch_groups,), 655, dtype=torch.int32, device=DEVICE))


def test_retrieval_budget_can_count_sink_and_recent_tokens():
    cache = make_cache(
        layer_num=1,
        max_length=32768,
        max_new_length=1,
        static_pattern_start=4,
        static_pattern_end=32,
        retrieval_budget=0.02,
        sig_token_cache_size=1024,
        exclude_preserved_from_budget=False,
    )
    cache.prompt_lengths.fill_(32768)
    cache.visible_lengths.fill_(32768)
    cache.layer_visible_lengths.fill_(32768)

    cache.prepare_cache()

    # floor(32768 * 0.02) - sink(4) - recent(32) = 619
    assert torch.equal(cache.topk_buffer, torch.full((cache.batch_groups,), 619, dtype=torch.int32, device=DEVICE))


def test_compute_topk_can_exclude_preserved_tokens_from_sparse_budget():
    cache = make_cache(
        layer_num=1,
        retrieval_budget=0.02,
        sig_min_retrieval_topk=0,
    )

    assert cache._compute_topk(retrieval_length=1000, preserved_length=100, visible_length=1100) == 22

    retrieval_lengths = torch.tensor([1000, 5, 0], dtype=torch.int32, device=DEVICE)
    preserved_lengths = torch.tensor([100, 100, 10], dtype=torch.int32, device=DEVICE)
    visible_lengths = torch.tensor([1100, 105, 10], dtype=torch.int32, device=DEVICE)

    assert torch.equal(
        cache._compute_topk_for_lengths(retrieval_lengths, preserved_lengths, visible_lengths),
        torch.tensor([22, 2, 0], dtype=torch.int32, device=DEVICE),
    )


def test_budget_topk_has_minimum_request_width():
    cache = make_cache(
        layer_num=1,
        max_new_length=5,
        static_pattern_start=4,
        static_pattern_end=4,
        retrieval_budget=0.001,
    )
    cache.prompt_lengths.fill_(32)
    cache.visible_lengths.fill_(32)
    cache.layer_visible_lengths.fill_(32)

    cache.prepare_cache()

    assert cache.selected_indices_buffer.shape[1] == 16
    assert torch.equal(cache.topk_buffer, torch.full((cache.batch_groups,), 16, dtype=torch.int32, device=DEVICE))


def test_budget_topk_minimum_can_be_disabled():
    cache = make_cache(
        layer_num=1,
        max_new_length=5,
        static_pattern_start=4,
        static_pattern_end=4,
        retrieval_budget=0.001,
        sig_min_retrieval_topk=0,
    )
    cache.prompt_lengths.fill_(32)
    cache.visible_lengths.fill_(32)
    cache.layer_visible_lengths.fill_(32)

    cache.prepare_cache()

    assert cache.selected_indices_buffer.shape[1] == 1
    assert torch.equal(cache.topk_buffer, torch.zeros((cache.batch_groups,), dtype=torch.int32, device=DEVICE))


def test_sig_topk_override_has_minimum_request_width():
    cache = make_cache(
        layer_num=1,
        max_new_length=5,
        static_pattern_start=4,
        static_pattern_end=4,
        sig_topk=2,
    )
    cache.prompt_lengths.fill_(32)
    cache.visible_lengths.fill_(32)
    cache.layer_visible_lengths.fill_(32)

    cache.prepare_cache()

    assert cache.selected_indices_buffer.shape[1] == 16
    assert torch.equal(cache.topk_buffer, torch.full((cache.batch_groups,), 16, dtype=torch.int32, device=DEVICE))


def test_sig_topk_minimum_can_be_disabled():
    cache = make_cache(
        layer_num=1,
        max_new_length=5,
        static_pattern_start=4,
        static_pattern_end=4,
        sig_topk=2,
        sig_min_retrieval_topk=0,
    )
    cache.prompt_lengths.fill_(32)
    cache.visible_lengths.fill_(32)
    cache.layer_visible_lengths.fill_(32)

    cache.prepare_cache()

    assert cache.selected_indices_buffer.shape[1] == 2
    assert torch.equal(cache.topk_buffer, torch.full((cache.batch_groups,), 2, dtype=torch.int32, device=DEVICE))


def test_budget_topk_still_caps_to_retrieval_length_above_minimum():
    cache = make_cache(
        layer_num=1,
        max_new_length=5,
        static_pattern_start=4,
        static_pattern_end=4,
        retrieval_budget=2.0,
    )
    cache.prompt_lengths.fill_(32)
    cache.visible_lengths.fill_(32)
    cache.layer_visible_lengths.fill_(32)

    cache.prepare_cache()

    assert cache.selected_indices_buffer.shape[1] == 28
    assert torch.equal(cache.topk_buffer, torch.full((cache.batch_groups,), 24, dtype=torch.int32, device=DEVICE))


def test_sync_releases_prefill_signature_references():
    cache = make_cache(layer_num=1)
    query_states = torch.randn((1, 12, 4, 128), dtype=DTYPE, device=DEVICE)
    key_states = torch.randn((1, 12, 2, 128), dtype=DTYPE, device=DEVICE)
    value_states = torch.randn((1, 12, 2, 128), dtype=DTYPE, device=DEVICE)

    cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx=0, start_bdx=0)
    assert len(cache.prefill_signature_refs[0]) == 1

    cache.sync(0, 0)

    assert len(cache.prefill_signature_refs[0]) == 0
    assert cache.prefill_signature_events[0] == []


def test_prefill_offload_avoids_cpu_staging_tensors(monkeypatch):
    cache = make_cache(layer_num=1)
    query_states = torch.randn((1, 12, 4, 128), dtype=DTYPE, device=DEVICE)
    key_states = torch.randn((1, 12, 2, 128), dtype=DTYPE, device=DEVICE)
    value_states = torch.randn((1, 12, 2, 128), dtype=DTYPE, device=DEVICE)
    cpu_staging_calls = []
    original_to = torch.Tensor.to

    def recording_to(self, *args, **kwargs):
        device = None
        if args:
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]
        if isinstance(device, torch.device):
            device = str(device)
        if device == "cpu":
            cpu_staging_calls.append(tuple(self.shape))
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", recording_to)

    cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx=0, start_bdx=0)

    assert cpu_staging_calls == []


def test_prefill_signature_build_avoids_custom_signature_kernel():
    cache = make_cache(layer_num=1)
    query_states = torch.randn((1, 12, 4, 128), dtype=DTYPE, device=DEVICE)
    key_states = torch.randn((1, 12, 2, 128), dtype=DTYPE, device=DEVICE)
    value_states = torch.randn((1, 12, 2, 128), dtype=DTYPE, device=DEVICE)
    assert not hasattr(cometkv_cache_module, "build_packed_signatures_into")

    cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx=0, start_bdx=0)
    cache.sync(0, 0)

    assert torch.count_nonzero(cache.signature_index[0][0, :, :12, :]).item() > 0


def test_prefill_signature_index_matches_reference_projection():
    cache = make_cache(layer_num=1)
    query_states = torch.randn((1, 12, 4, 128), dtype=DTYPE, device=DEVICE)
    key_states = torch.randn((1, 12, 2, 128), dtype=DTYPE, device=DEVICE)
    value_states = torch.randn((1, 12, 2, 128), dtype=DTYPE, device=DEVICE)

    cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx=0, start_bdx=0)
    cache.sync(0, 0)

    projection = build_reference_random_orth_projection().to(device=DEVICE)
    reference = pack_reference(key_states.transpose(1, 2), projection)

    assert torch.equal(cache.signature_index[0][0, :, :12, :], reference[0])


def test_init_determines_token_cache_size_from_retrieval_budget_without_allocating_cache():
    cache = cometkv_cache(
        valid_start=np.array([0], dtype=np.int32),
        layer_num=1,
        batch_size=1,
        max_length=98312,
        num_key_value_heads=2,
        num_heads=4,
        head_dim=128,
        dtype=DTYPE,
        layer_mapping={'0': DEVICE},
        max_new_length=8,
        static_pattern_start=4,
        static_pattern_end=4,
        retrieval_budget=0.1,
        sig_bits=128,
        sig_topk=0,
        sig_chunk_size=131072,
        sig_seed=1234,
        sig_mode='random_orth',
        sig_token_cache_size=1024,
        prefill_bsz=1,
        num_gpus=1,
        model_size=8,
    )

    assert cache.sig_token_cache_size == 10240
    assert cache.token_cache_ids is None
    assert cache.token_cache_locks is None
    assert cache.token_cache_keys is None
    assert cache.token_cache_values is None


def test_prepare_cache_allocates_fixed_token_cache_once():
    cache = cometkv_cache(
        valid_start=np.array([0], dtype=np.int32),
        layer_num=1,
        batch_size=1,
        max_length=98312,
        num_key_value_heads=2,
        num_heads=4,
        head_dim=128,
        dtype=DTYPE,
        layer_mapping={'0': DEVICE},
        max_new_length=8,
        static_pattern_start=32,
        static_pattern_end=64,
        retrieval_budget=0.018,
        sig_bits=128,
        sig_topk=0,
        sig_chunk_size=131072,
        sig_seed=1234,
        sig_mode='random_orth',
        sig_token_cache_size=1024,
        prefill_bsz=1,
        num_gpus=1,
        model_size=8,
    )
    # Init-time provisional sizing stays at the legacy ~1x-topk rule; prepare_cache() then grows
    # the cache to ~4x topk (temporal selection reuse) bounded by residual VRAM.
    assert cache.sig_token_cache_size == 2048
    assert cache.token_cache_ids is None

    cache.prompt_lengths.fill_(98166)
    cache.visible_lengths.fill_(98166)
    cache.layer_visible_lengths.fill_(98166)

    cache.prepare_cache()

    initial_cache_ids = cache.token_cache_ids[0]
    assert initial_cache_ids.shape[1] == cache.sig_token_cache_size
    assert cache.sig_token_cache_size >= 2048
    assert cache.sig_token_cache_size <= 8192  # 4x topk rounded, unless VRAM-capped below

    cache.prepare_cache()

    assert cache.token_cache_ids[0] is initial_cache_ids


def test_prepare_cache_token_cache_legacy_sizing_with_mult_one(monkeypatch):
    monkeypatch.setenv("COMETKV_TOKEN_CACHE_MULT", "1")
    cache = cometkv_cache(
        valid_start=np.array([0], dtype=np.int32),
        layer_num=1,
        batch_size=1,
        max_length=98312,
        num_key_value_heads=2,
        num_heads=4,
        head_dim=128,
        dtype=DTYPE,
        layer_mapping={'0': DEVICE},
        max_new_length=8,
        static_pattern_start=32,
        static_pattern_end=64,
        retrieval_budget=0.018,
        sig_bits=128,
        sig_topk=0,
        sig_chunk_size=131072,
        sig_seed=1234,
        sig_mode='random_orth',
        sig_token_cache_size=1024,
        prefill_bsz=1,
        num_gpus=1,
        model_size=8,
    )
    cache.prompt_lengths.fill_(98166)
    cache.visible_lengths.fill_(98166)
    cache.layer_visible_lengths.fill_(98166)
    cache.prepare_cache()
    assert cache.token_cache_ids[0].shape[1] == 2048


def test_default_direct_cache_uses_fused_pinned_kv_layout():
    cache = make_cache(layer_num=1)

    assert cache.cpu_kv_cache is not None
    assert cache.cpu_kv_cache[0].shape == (cache.batch_groups, cache.retrieval_capacity, 2, cache.head_dim)
    assert cache.cpu_kv_cache[0].is_pinned()
    assert cache.cpu_kv_cache[0].is_contiguous()
    assert cache.cpu_key_cache[0].data_ptr() == cache.cpu_kv_cache[0][:, :, 0, :].data_ptr()
    assert cache.cpu_value_cache[0].data_ptr() == cache.cpu_kv_cache[0][:, :, 1, :].data_ptr()


def test_cometkv_fast_only_cpu_kv_cache_covers_generated_tokens():
    cache = make_cache(layer_num=1, max_length=256, max_new_length=130, sig_topk=4)

    assert cache.cpu_kv_cache[0].size(1) == cache.max_length


def test_prefill_update_writes_interleaved_cpu_kv_without_fused_staging():
    cache = make_cache(layer_num=1)
    query_states = torch.randn((1, 12, 4, 128), dtype=DTYPE, device=DEVICE)
    key_states = torch.randn((1, 12, 2, 128), dtype=DTYPE, device=DEVICE)
    value_states = torch.randn((1, 12, 2, 128), dtype=DTYPE, device=DEVICE)

    cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx=0, start_bdx=0)

    key_ref, value_ref = cache.prefill_copy_refs[0][0]
    assert torch.is_tensor(key_ref)
    assert torch.is_tensor(value_ref)
    assert key_ref.shape == (cache.batch_groups, 12, cache.head_dim)
    assert value_ref.shape == (cache.batch_groups, 12, cache.head_dim)
    assert key_ref.is_cuda
    assert value_ref.is_cuda
    assert key_ref.is_contiguous()
    assert value_ref.is_contiguous()

    cache.sync(0, 0)
    expected_keys = key_states.transpose(1, 2).reshape(cache.batch_groups, 12, cache.head_dim).cpu()
    expected_values = value_states.transpose(1, 2).reshape(cache.batch_groups, 12, cache.head_dim).cpu()
    assert torch.equal(cache.cpu_kv_cache[0][:, :12, 0, :], expected_keys)
    assert torch.equal(cache.cpu_kv_cache[0][:, :12, 1, :], expected_values)


def test_int8_quantize_kv_writes_interleaved_cpu_kv_and_value_scale():
    cache = make_cache(layer_num=1, cpu_kv_quant="int8")
    cache.prefill_signature_chunk_size = 5
    token_start = 3
    n_tokens = 12
    key_rows = torch.randn((cache.batch_groups, n_tokens, cache.head_dim), dtype=DTYPE, device=DEVICE)
    value_rows = torch.randn_like(key_rows)

    cache._freeze_k_per_channel_scale(0, 0, key_rows)
    refs = cache._quantize_kv_into_cpu(0, 0, token_start, key_rows, value_rows)
    torch.cuda.synchronize()

    k_scale = cache.cpu_kv_k_scale[0].unsqueeze(1)
    expected_k = torch.clamp(torch.round(key_rows.float() / k_scale), -127, 127).to(torch.int8).cpu()
    v_scale = value_rows.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    expected_v = torch.clamp(torch.round(value_rows.float() / v_scale), -127, 127).to(torch.int8).cpu()
    expected_v_scale = v_scale.squeeze(-1).cpu()

    assert len(refs) == 3
    assert all(torch.is_tensor(item) for ref in refs for item in ref)
    assert torch.equal(cache.cpu_kv_cache[0][:, token_start:token_start + n_tokens, 0, :], expected_k)
    assert torch.equal(cache.cpu_kv_cache[0][:, token_start:token_start + n_tokens, 1, :], expected_v)
    assert torch.equal(cache.cpu_kv_v_scale[0][:, token_start:token_start + n_tokens], expected_v_scale)


def test_fast_only_rejects_non_static_default_direct_gather_path():
    cache = make_cache(layer_num=1, sig_topk=2)
    cache.prompt_lengths.fill_(6)
    cache.visible_lengths.fill_(6)
    cache.layer_visible_lengths.fill_(6)

    with pytest.raises(AssertionError, match="static prompt retrieval"):
        cache.prepare_cache()


def test_decode_region_split_uses_global_latest_recent_window():
    cache = make_cache(layer_num=1)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(15)

    regions = cache.get_decode_regions(0)

    assert regions["sink_range"] == (0, 4)
    assert regions["recent_range"] == (11, 15)
    assert regions["retrieval_ranges"] == [(4, 11)]


def test_decode_region_split_keeps_middle_as_one_retrieval_range():
    cache = make_cache(layer_num=1, max_length=40, static_pattern_end=3)
    cache.prompt_lengths.fill_(30)
    cache.visible_lengths.fill_(35)

    regions = cache.get_decode_regions(0)

    assert regions["sink_range"] == (0, 4)
    assert regions["recent_range"] == (32, 35)
    assert regions["retrieval_ranges"] == [(4, 32)]



def test_decode_update_makes_current_token_visible_before_global_commit():
    cache = make_cache(layer_num=2)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    cache.context = 12

    key_states = torch.randn((1, 1, 2, 128), dtype=DTYPE, device=DEVICE)
    value_states = torch.randn((1, 1, 2, 128), dtype=DTYPE, device=DEVICE)

    cache.decode_update_kv_cache(key_states, value_states, layer_idx=0)
    assert cache.layer_visible_lengths[0, 0].item() == 12
    assert cache.fixed_prompt_local_layer_visible_lengths_host == [13, 12]
    assert cache.visible_lengths[0].item() == 12

    cache.decode_update_kv_cache(key_states, value_states, layer_idx=1)
    assert cache.visible_lengths[0].item() == 13
    assert cache.fixed_prompt_local_layer_visible_lengths_host == [13, 13]


def test_same_length_batch_decode_update_appends_each_batch_row_once():
    cache = make_cache(
        layer_num=1,
        batch_size=2,
        valid_start=np.array([0, 0], dtype=np.int32),
        max_new_length=5,
    )
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    cache.context = 12

    key_states = torch.empty((2, 1, cache.kv_head, cache.head_dim), dtype=DTYPE, device=DEVICE)
    value_states = torch.empty_like(key_states)
    for batch_idx in range(2):
        for head_idx in range(cache.kv_head):
            key_states[batch_idx, 0, head_idx, :].fill_(batch_idx * 1000 + head_idx * 100 + 7)
            value_states[batch_idx, 0, head_idx, :].fill_(batch_idx * 1000 + head_idx * 100 + 700)

    cache.decode_update_kv_cache(key_states, value_states, layer_idx=0)
    torch.cuda.synchronize()

    assert torch.equal(cache.visible_lengths, torch.full((2,), 13, dtype=torch.int32, device=DEVICE))
    assert cache.context == 13
    assert cache.lockstep_decode_step_host == 1
    assert torch.equal(cache.decode_hot_keys[0][:, :, 0, :], key_states[:, 0])
    assert torch.equal(cache.decode_hot_values[0][:, :, 0, :], value_states[:, 0])
    assert torch.equal(
        cache.decode_hot_token_ids[0][:, :, 0],
        torch.full((2, cache.kv_head), -1, dtype=torch.int32, device=DEVICE),
    )


def test_decode_update_skips_signature_when_static_prompt_retrieval_uses_recent_only():
    cache = make_cache(layer_num=1, max_new_length=5)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    cache.context = 12
    assert cache.use_static_prompt_retrieval is True
    assert not hasattr(cometkv_cache_module, "build_packed_signatures_into")

    key_states = torch.randn((1, 1, 2, 128), dtype=DTYPE, device=DEVICE)
    value_states = torch.randn((1, 1, 2, 128), dtype=DTYPE, device=DEVICE)

    cache.decode_update_kv_cache(key_states, value_states, layer_idx=0)

    assert cache.layer_visible_lengths[0, 0].item() == 12
    assert cache.visible_lengths[0].item() == 13
    assert cache.fixed_prompt_local_layer_visible_lengths_host == [13]
    assert torch.allclose(cache.decode_hot_keys[0][0, :, 0, :], key_states[0, 0])
    assert torch.allclose(cache.decode_hot_values[0][0, :, 0, :], value_states[0, 0])
    assert torch.equal(cache.decode_hot_token_ids[0][0, :, 0], torch.full((2,), -1, dtype=torch.int32, device=DEVICE))
    assert torch.count_nonzero(cache.signature_index[0][0, :, 12:13, :]).item() == 0


def test_decode_update_lockstep_fast_path_does_not_write_token_ids():
    cache = make_cache(layer_num=1, max_new_length=130, static_pattern_start=4, static_pattern_end=32)
    cache.prompt_lengths.fill_(64)
    cache.visible_lengths.fill_(64)
    cache.layer_visible_lengths.fill_(64)
    cache.prepare_cache()
    cache.decode_hot_token_ids[0].fill_(-1)
    key_states = torch.randn((1, 1, cache.kv_head, cache.head_dim), dtype=DTYPE, device=DEVICE)
    value_states = torch.randn_like(key_states)

    cache.decode_update_kv_cache(key_states, value_states, 0)

    assert torch.equal(cache.decode_hot_token_ids[0], torch.full_like(cache.decode_hot_token_ids[0], -1))


def test_decode_update_records_lockstep_profile_fields():
    cache = make_cache(layer_num=1, max_new_length=130, static_pattern_start=4, static_pattern_end=32)
    cache.prompt_lengths.fill_(64)
    cache.visible_lengths.fill_(64)
    cache.layer_visible_lengths.fill_(64)
    cache.prepare_cache()
    cache.context = 64
    cache.profile_decode_update = True
    key_states = torch.randn((1, 1, cache.kv_head, cache.head_dim), dtype=DTYPE, device=DEVICE)
    value_states = torch.randn_like(key_states)

    cache.decode_update_kv_cache(key_states, value_states, 0)

    profile = cache.last_decode_update_profile
    assert "hash_lockstep_append_ms" in profile
    assert "hash_evicted_decode_kv_index_update_ms" in profile
    assert "hash_lockstep_retrieval_plan_update_ms" in profile
    assert profile["hash_lockstep_append_ms"] >= 0.0
    assert profile["hash_evicted_decode_kv_index_update_ms"] == 0.0
    assert profile["hash_lockstep_retrieval_plan_update_ms"] == 0.0


def test_decode_update_uses_lockstep_append_for_static_prompt_retrieval(monkeypatch):
    cache = make_cache(layer_num=1, max_new_length=5)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    cache.context = 12
    assert cache.use_static_prompt_retrieval is True

    append_calls = []

    def fake_lockstep_append(*_args, **_kwargs):
        raise AssertionError("last layer should use fused append-advance")

    def fake_lockstep_append_and_advance(
        key_states,
        value_states,
        decode_hot_keys,
        decode_hot_values,
        visible_lengths,
        decode_step,
        local_capacity,
        slide_stride,
    ):
        append_calls.append((tuple(key_states.shape), tuple(value_states.shape), decode_step, local_capacity, slide_stride))
        decode_hot_keys[0, :, decode_step, :].copy_(key_states[0, 0])
        decode_hot_values[0, :, decode_step, :].copy_(value_states[0, 0])
        visible_lengths.add_(1)

    monkeypatch.setattr(cometkv_cache_module, "append_lockstep_local_kv_cache", fake_lockstep_append)
    monkeypatch.setattr(cometkv_cache_module, "append_lockstep_local_kv_cache_and_advance", fake_lockstep_append_and_advance)

    key_states = torch.randn((1, 1, 2, 128), dtype=DTYPE, device=DEVICE)
    value_states = torch.randn((1, 1, 2, 128), dtype=DTYPE, device=DEVICE)

    cache.decode_update_kv_cache(key_states, value_states, layer_idx=0)

    assert append_calls == [((1, 1, 2, 128), (1, 1, 2, 128), 0, 132, 128)]
    assert torch.allclose(cache.decode_hot_keys[0][0, :, 0, :], key_states[0, 0])
    assert torch.allclose(cache.decode_hot_values[0][0, :, 0, :], value_states[0, 0])
    assert torch.equal(cache.decode_hot_token_ids[0][0, :, 0], torch.full((2,), -1, dtype=torch.int32, device=DEVICE))


def test_decode_update_keeps_fixed_local_static_lengths_once_per_step():
    cache = make_cache(layer_num=2, max_new_length=5)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    cache.context = 12
    assert cache.use_static_prompt_retrieval is True

    key_states = torch.randn((1, 1, 2, 128), dtype=DTYPE, device=DEVICE)
    value_states = torch.randn((1, 1, 2, 128), dtype=DTYPE, device=DEVICE)

    cache.decode_update_kv_cache(key_states, value_states, layer_idx=0)
    assert cache._lockstep_fixed_prompt_recent_length_host(layer_idx=0) == 1
    assert torch.equal(cache.static_lengths_buffer, torch.full((cache.batch_groups,), 8, dtype=torch.int32, device=DEVICE))

    cache.decode_update_kv_cache(key_states, value_states, layer_idx=1)
    assert cache._lockstep_fixed_prompt_recent_length_host(layer_idx=0) == 1
    assert cache._lockstep_fixed_prompt_recent_length_host(layer_idx=1) == 1
    assert torch.equal(cache.static_lengths_buffer, torch.full((cache.batch_groups,), 8, dtype=torch.int32, device=DEVICE))


def test_decode_update_keeps_prompt_side_recent_suffix_in_static_kv():
    cache = make_cache(layer_num=1, max_new_length=5)
    query_states = torch.zeros((1, 12, 4, 128), dtype=DTYPE, device=DEVICE)
    key_states = torch.zeros((1, 12, 2, 128), dtype=DTYPE, device=DEVICE)
    value_states = torch.zeros_like(key_states)
    for token in range(12):
        key_states[:, token, :, :].fill_(token)
        value_states[:, token, :, :].fill_(token + 100)

    cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx=0, start_bdx=0)
    cache.sync(0, 0)
    cache.prepare_cache()

    decode_key = torch.zeros((1, 1, 2, 128), dtype=DTYPE, device=DEVICE)
    decode_value = torch.zeros_like(decode_key)
    cache.decode_update_kv_cache(decode_key, decode_value, layer_idx=0)

    assert torch.all(cache.static_keys[0][0, :, 4, :] == 8)
    assert torch.all(cache.static_keys[0][0, :, 5, :] == 9)
    assert torch.all(cache.static_keys[0][0, :, 6, :] == 10)
    assert torch.all(cache.static_keys[0][0, :, 7, :] == 11)
    assert torch.all(cache.static_values[0][0, :, 4, :] == 108)
    assert torch.all(cache.static_values[0][0, :, 5, :] == 109)
    assert torch.all(cache.static_values[0][0, :, 6, :] == 110)
    assert torch.all(cache.static_values[0][0, :, 7, :] == 111)


def test_sparse_attention_static_prompt_retrieval_uses_direct_concat(monkeypatch):
    cache = make_cache(layer_num=1, max_new_length=5)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    queries = torch.randn((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)
    called = {"flash": False}

    def fake_flash(*_args, **_kwargs):
        called["flash"] = True
        return torch.zeros((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)

    def fake_topk(_queries, _layer_idx):
        cache.selected_indices_buffer.fill_(0)
        cache.sparse_lengths_buffer.fill_(1)

    cache._static_fixed_concat_flash_attention_from_indices = fake_flash
    cache._grouped_query_asym_topk = fake_topk

    cache.sparse_attention(queries, layer_idx=0)

    assert called["flash"] is True


def test_sparse_attention_static_prompt_retrieval_reuses_initial_plan(monkeypatch):
    cache = make_cache(layer_num=2, max_new_length=5)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    assert cache.use_static_prompt_retrieval is True

    cache.visible_lengths.fill_(13)
    cache.layer_visible_lengths.fill_(13)
    queries = torch.randn((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)
    plan_calls = []
    original_update_plan = cache._update_retrieval_plan

    def counted_update_plan(visible_lengths):
        plan_calls.append(tuple(visible_lengths.detach().cpu().tolist()))
        return original_update_plan(visible_lengths)

    def fake_topk(_queries, _layer_idx):
        cache.selected_indices_buffer.fill_(0)
        cache.sparse_lengths_buffer.fill_(1)

    def fake_flash(*_args, **_kwargs):
        return torch.zeros((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)

    monkeypatch.setattr(cache, "_update_retrieval_plan", counted_update_plan)
    cache._grouped_query_asym_topk = fake_topk
    cache._static_fixed_concat_flash_attention_from_indices = fake_flash

    cache.sparse_attention(queries, layer_idx=0)
    cache.sparse_attention(queries, layer_idx=1)

    assert plan_calls == []


def test_removed_fixed_prompt_local_mode_flags_are_rejected():
    with pytest.raises(TypeError):
        make_cache(layer_num=1, sig_concat_flash_attention=True)
    with pytest.raises(TypeError):
        make_cache(layer_num=1, sig_static_fixed_prompt_local=False)


def test_static_prompt_retrieval_preallocates_max_decode_topk_capacity():
    cache = make_cache(layer_num=1, max_new_length=5, static_pattern_end=4, retrieval_budget=0.5)
    cache.prompt_lengths.fill_(32)
    cache.visible_lengths.fill_(32)
    cache.layer_visible_lengths.fill_(32)

    cache.prepare_cache()

    assert cache.use_static_prompt_retrieval is True
    assert cache.selected_indices_buffer.shape[1] == 18
    assert torch.equal(cache.topk_buffer, torch.full((cache.batch_groups,), 16, dtype=torch.int32, device=DEVICE))


def test_fixed_prompt_local_keeps_static_lengths_across_decode_update():
    cache = make_cache(layer_num=1, max_new_length=5)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    cache.context = 12
    assert cache.use_static_prompt_retrieval is True

    key_states = torch.randn((1, 1, 2, 128), dtype=DTYPE, device=DEVICE)
    value_states = torch.randn((1, 1, 2, 128), dtype=DTYPE, device=DEVICE)

    cache.decode_update_kv_cache(key_states, value_states, layer_idx=0)

    assert torch.equal(cache.static_lengths_buffer, torch.full((cache.batch_groups,), 8, dtype=torch.int32, device=DEVICE))
    assert cache._lockstep_fixed_prompt_recent_length_host(layer_idx=0) == 1


def test_fixed_prompt_local_reuses_initial_retrieval_plan(monkeypatch):
    cache = make_cache(layer_num=1, max_new_length=5)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    queries = torch.randn((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)
    plan_calls = []

    def fail_update_plan(visible_lengths):
        plan_calls.append(tuple(visible_lengths.detach().cpu().tolist()))
        raise AssertionError("fixed prompt local should reuse the initial retrieval plan")

    def fake_topk(_queries, _layer_idx):
        cache.selected_indices_buffer.fill_(0)
        cache.sparse_lengths_buffer.fill_(1)

    def fake_flash(*_args, **_kwargs):
        return torch.zeros((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)

    monkeypatch.setattr(cache, "_update_retrieval_plan", fail_update_plan)
    cache._grouped_query_asym_topk = fake_topk
    cache._static_fixed_concat_flash_attention_from_indices = fake_flash

    cache.sparse_attention(queries, layer_idx=0)

    assert plan_calls == []


def test_fixed_prompt_local_allows_decode_beyond_prompt_recent_window():
    cache = make_cache(layer_num=1, max_new_length=9)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)

    cache.prepare_cache()

    assert cache.use_static_prompt_retrieval is True
    assert cache.static_prompt_fast_recent_capacity == 8
    # At decode start range1 = [4, 8) -> 4 candidates, so topk is clamped to 4 (not the
    # sig_min_retrieval_topk=16 floor): requesting 16 would leave 12 empty (-1) slots that the
    # gather zero-fills into the attended window, diluting the softmax. See _compute_topk_for_lengths.
    assert torch.equal(cache.topk_buffer, torch.full((cache.batch_groups,), 4, dtype=torch.int32, device=DEVICE))


def test_fixed_prompt_local_compacts_decode_recent_window_after_stride():
    cache = make_cache(layer_num=1, max_new_length=10)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    cache.context = 12
    assert cache.use_static_prompt_retrieval is True

    for decode_idx in range(9):
        key_states = torch.full((1, 1, 2, 128), decode_idx, dtype=DTYPE, device=DEVICE)
        value_states = torch.full((1, 1, 2, 128), decode_idx + 100, dtype=DTYPE, device=DEVICE)
        cache.decode_update_kv_cache(key_states, value_states, layer_idx=0)

    assert cache._lockstep_fixed_prompt_recent_length_host(layer_idx=0) == 9
    assert torch.equal(cache.static_lengths_buffer, torch.full((cache.batch_groups,), 8, dtype=torch.int32, device=DEVICE))
    for slot, token in enumerate(range(9)):
        assert torch.all(cache.decode_hot_keys[0][0, :, slot, :] == token)
        assert torch.all(cache.decode_hot_values[0][0, :, slot, :] == token + 100)
        assert torch.equal(
            cache.decode_hot_token_ids[0][0, :, slot],
            torch.full((2,), -1, dtype=torch.int32, device=DEVICE),
        )


def test_fixed_prompt_local_keeps_append_only_fast_path_through_double_recent_window():
    cache = make_cache(layer_num=1, max_new_length=9)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()

    assert cache.use_static_prompt_retrieval is True
    assert cache.static_prompt_fast_recent_capacity == 8
    assert cache.use_chunked_fixed_prompt_local is False


def test_fixed_prompt_local_uses_128_decode_blocks_with_32_token_overlap():
    cache = make_cache(
        layer_num=1,
        max_length=256,
        max_new_length=130,
        static_pattern_end=32,
    )
    cache.prompt_lengths.fill_(64)
    cache.visible_lengths.fill_(64)
    cache.layer_visible_lengths.fill_(64)

    cache.prepare_cache()

    assert cache.use_static_prompt_retrieval is True
    assert cache.static_prompt_fast_recent_capacity == 129
    assert cache._fixed_prompt_local_recent_capacity() == 160
    assert cache._fixed_prompt_local_slide_stride() == 128
    assert cache.use_chunked_fixed_prompt_local is True
    assert cache._fixed_prompt_local_window_start(128) == 0
    assert cache._fixed_prompt_local_window_start(129) == 96

    cache._update_fixed_prompt_local_static_lengths(torch.tensor([64 + 33], dtype=torch.int32, device=DEVICE))
    assert torch.equal(cache.static_lengths_buffer, torch.full((cache.batch_groups,), 36, dtype=torch.int32, device=DEVICE))

    cache._update_fixed_prompt_local_static_lengths(torch.tensor([64 + 129], dtype=torch.int32, device=DEVICE))
    assert torch.equal(cache.static_lengths_buffer, torch.full((cache.batch_groups,), 4, dtype=torch.int32, device=DEVICE))

    previous_tokens = torch.arange(128, dtype=DTYPE, device=DEVICE).view(1, 128, 1).expand(2, 128, 128)
    cache.decode_hot_keys[0][0, :, :128, :].copy_(previous_tokens)
    cache.decode_hot_values[0][0, :, :128, :].copy_(previous_tokens + 1000)

    key_states = torch.full((1, 1, 2, 128), 128, dtype=DTYPE, device=DEVICE)
    value_states = torch.full((1, 1, 2, 128), 1128, dtype=DTYPE, device=DEVICE)
    cache._append_chunked_fixed_prompt_local_kv(
        key_states,
        value_states,
        torch.tensor([64 + 128], dtype=torch.int32, device=DEVICE),
        torch.tensor([64], dtype=torch.int32, device=DEVICE),
        layer_idx=0,
    )

    for slot, token in enumerate(range(96, 129)):
        assert torch.all(cache.decode_hot_keys[0][0, :, slot, :] == token)
        assert torch.all(cache.decode_hot_values[0][0, :, slot, :] == token + 1000)


def test_fixed_prompt_local_uses_prompt_recent_overlap_for_128_decode_blocks():
    cache = make_cache(
        layer_num=1,
        max_length=384,
        max_new_length=258,
        static_pattern_end=64,
    )
    cache.prompt_lengths.fill_(96)
    cache.visible_lengths.fill_(96)
    cache.layer_visible_lengths.fill_(96)

    cache.prepare_cache()

    assert cache.use_static_prompt_retrieval is True
    assert cache._fixed_prompt_local_recent_capacity() == 192
    assert cache._fixed_prompt_local_slide_stride() == 128
    assert cache.use_chunked_fixed_prompt_local is True
    assert cache._fixed_prompt_local_window_start(128) == 0
    assert cache._fixed_prompt_local_window_start(129) == 64
    assert cache._fixed_prompt_local_window_start(256) == 64
    assert cache._fixed_prompt_local_window_start(257) == 192

    cache._update_fixed_prompt_local_static_lengths(torch.tensor([96 + 65], dtype=torch.int32, device=DEVICE))
    assert torch.equal(cache.static_lengths_buffer, torch.full((cache.batch_groups,), 68, dtype=torch.int32, device=DEVICE))

    cache._update_fixed_prompt_local_static_lengths(torch.tensor([96 + 129], dtype=torch.int32, device=DEVICE))
    assert torch.equal(cache.static_lengths_buffer, torch.full((cache.batch_groups,), 4, dtype=torch.int32, device=DEVICE))

    previous_tokens = torch.arange(192, dtype=DTYPE, device=DEVICE).view(1, 192, 1).expand(2, 192, 128)
    cache.decode_hot_keys[0][0].copy_(previous_tokens)
    cache.decode_hot_values[0][0].copy_(previous_tokens + 1000)

    key_states = torch.full((1, 1, 2, 128), 128, dtype=DTYPE, device=DEVICE)
    value_states = torch.full((1, 1, 2, 128), 1128, dtype=DTYPE, device=DEVICE)
    cache._append_chunked_fixed_prompt_local_kv(
        key_states,
        value_states,
        torch.tensor([96 + 128], dtype=torch.int32, device=DEVICE),
        torch.tensor([96], dtype=torch.int32, device=DEVICE),
        layer_idx=0,
    )

    for slot, token in enumerate(range(64, 129)):
        assert torch.all(cache.decode_hot_keys[0][0, :, slot, :] == token)
        assert torch.all(cache.decode_hot_values[0][0, :, slot, :] == token + 1000)


def test_fixed_prompt_local_static_length_update_only_when_crossing_capacity():
    cache = make_cache(
        layer_num=1,
        max_length=256,
        max_new_length=130,
        static_pattern_end=32,
    )
    cache.prompt_lengths.fill_(64)
    cache.visible_lengths.fill_(64)
    cache.layer_visible_lengths.fill_(64)
    cache.prepare_cache()

    prompt_lengths = torch.tensor([64], dtype=torch.int32, device=DEVICE)
    assert cache._should_update_fixed_prompt_local_static_lengths(
        torch.tensor([64 + 33], dtype=torch.int32, device=DEVICE),
        torch.tensor([64 + 34], dtype=torch.int32, device=DEVICE),
        prompt_lengths,
    ) is False
    assert cache._should_update_fixed_prompt_local_static_lengths(
        torch.tensor([64 + 128], dtype=torch.int32, device=DEVICE),
        torch.tensor([64 + 129], dtype=torch.int32, device=DEVICE),
        prompt_lengths,
    ) is True
    assert cache._should_update_fixed_prompt_local_static_lengths(
        torch.tensor([64 + 129], dtype=torch.int32, device=DEVICE),
        torch.tensor([64 + 130], dtype=torch.int32, device=DEVICE),
        prompt_lengths,
    ) is False


def test_evicted_generated_kv_enters_retrieval_index():
    cache = make_cache(
        layer_num=1,
        max_length=256,
        max_new_length=131,
        static_pattern_start=4,
        static_pattern_end=32,
        sig_topk=8,
    )
    cache.prompt_lengths.fill_(64)
    cache.visible_lengths.fill_(64)
    cache.layer_visible_lengths.fill_(64)
    cache.prepare_cache()
    cache.context = 64

    first_generated_token = 64
    assert cache.lockstep_evicted_retrieval_end_host <= first_generated_token

    for step in range(129):
        key_states = torch.full((1, 1, cache.kv_head, cache.head_dim), step, dtype=DTYPE, device=DEVICE)
        value_states = torch.full_like(key_states, step + 100)
        cache.decode_update_kv_cache(key_states, value_states, layer_idx=0)

    torch.cuda.synchronize()
    assert cache.lockstep_evicted_retrieval_end_host > first_generated_token
    assert torch.all(cache.range_ends_buffer[:, 0] == 160)
    rows = cache.signature_index[0].view(cache.batch_groups, -1, cache.sig_bytes)
    assert torch.count_nonzero(rows[:, first_generated_token, :]).item() > 0
    assert torch.equal(
        cache.cpu_kv_cache[0][:, first_generated_token, 0, :].to(DEVICE),
        torch.full((cache.batch_groups, cache.head_dim), 0, dtype=DTYPE, device=DEVICE),
    )
    assert torch.equal(
        cache.cpu_kv_cache[0][:, first_generated_token, 1, :].to(DEVICE),
        torch.full((cache.batch_groups, cache.head_dim), 100, dtype=DTYPE, device=DEVICE),
    )


def test_lockstep_retrieval_plan_stops_at_evicted_generated_tokens():
    cache = make_cache(
        layer_num=1,
        max_length=256,
        max_new_length=130,
        static_pattern_start=4,
        static_pattern_end=32,
        sig_topk=8,
    )
    cache.prompt_lengths.fill_(64)
    cache.visible_lengths.fill_(64 + 129)
    cache.layer_visible_lengths.fill_(64 + 129)
    cache.use_static_prompt_retrieval = True
    cache.lockstep_prompt_length_host = 64
    cache.lockstep_evicted_retrieval_end_host = 160
    cache._allocate_decode_buffers()

    cache._update_retrieval_plan(cache.visible_lengths)

    starts = cache.range_starts_buffer[:, 0]
    ends = cache.range_ends_buffer[:, 0]
    assert torch.all(starts == cache.static_pattern_start)
    assert torch.all(ends == 160)


def test_decode_update_uses_fused_fixed_window_append_after_capacity(monkeypatch):
    cache = make_cache(
        layer_num=1,
        max_length=256,
        max_new_length=130,
        static_pattern_end=32,
    )
    cache.prompt_lengths.fill_(64)
    cache.visible_lengths.fill_(64)
    cache.layer_visible_lengths.fill_(64)
    cache.prepare_cache()
    cache.context = 64 + 128
    cache.visible_lengths.fill_(64 + 128)
    cache.layer_visible_lengths.fill_(64 + 128)
    cache.fixed_prompt_local_layer_visible_lengths_host[0] = 64 + 128
    assert cache.use_chunked_fixed_prompt_local is True

    calls = []

    def fake_lockstep_append(*_args, **_kwargs):
        raise AssertionError("last layer should use fused append-advance")

    def fake_lockstep_append_and_advance(
        key_states,
        value_states,
        decode_hot_keys,
        decode_hot_values,
        visible_lengths,
        decode_step,
        local_capacity,
        slide_stride,
    ):
        calls.append(
            (
                int(decode_step),
                int(local_capacity),
                int(slide_stride),
            )
        )
        decode_hot_keys[0, :, 32, :].copy_(key_states[0, 0])
        decode_hot_values[0, :, 32, :].copy_(value_states[0, 0])
        visible_lengths.add_(1)

    def fail_python_append(*_args, **_kwargs):
        raise AssertionError("fixed prompt local should use fused append after capacity")

    monkeypatch.setattr(cometkv_cache_module, "append_lockstep_local_kv_cache", fake_lockstep_append)
    monkeypatch.setattr(cometkv_cache_module, "append_lockstep_local_kv_cache_and_advance", fake_lockstep_append_and_advance)
    monkeypatch.setattr(cache, "_append_chunked_fixed_prompt_local_kv", fail_python_append)

    key_states = torch.full((1, 1, 2, 128), 7, dtype=DTYPE, device=DEVICE)
    value_states = torch.full_like(key_states, 107)

    cache.decode_update_kv_cache(key_states, value_states, layer_idx=0)

    assert calls == [(128, 160, 128)]
    assert torch.all(cache.decode_hot_keys[0][0, :, 32, :] == 7)
    assert torch.all(cache.decode_hot_values[0][0, :, 32, :] == 107)
    assert torch.equal(
        cache.decode_hot_token_ids[0][0, :, 32],
        torch.full((2,), -1, dtype=torch.int32, device=DEVICE),
    )


def test_fixed_concat_decode_update_skips_layer_visible_length_broadcast():
    cache = make_cache(layer_num=2, max_new_length=5)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    cache.context = 12
    assert cache._can_use_static_fixed_concat_flash_attention() is True

    key_states = torch.randn((1, 1, 2, 128), dtype=DTYPE, device=DEVICE)
    value_states = torch.randn((1, 1, 2, 128), dtype=DTYPE, device=DEVICE)

    cache.decode_update_kv_cache(key_states, value_states, layer_idx=0)

    assert cache.visible_lengths[0].item() == 12
    assert cache.layer_visible_lengths[0, 0].item() == 12
    assert cache.fixed_prompt_local_layer_visible_lengths_host[0] == 13

    cache.decode_update_kv_cache(key_states, value_states, layer_idx=1)

    assert cache.visible_lengths[0].item() == 13
    assert torch.equal(cache.layer_visible_lengths, torch.full_like(cache.layer_visible_lengths, 12))
    assert cache.fixed_prompt_local_layer_visible_lengths_host == [13, 13]


def test_fixed_concat_decode_update_advances_visible_lengths_in_last_append(monkeypatch):
    cache = make_cache(layer_num=2, max_new_length=5)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    cache.context = 12
    assert cache._can_use_static_fixed_concat_flash_attention() is True

    old_append_calls = []
    fused_append_calls = []

    def fake_lockstep_append(
        key_states,
        value_states,
        decode_hot_keys,
        decode_hot_values,
        decode_step,
        local_capacity,
        slide_stride,
    ):
        old_append_calls.append(int(decode_step))
        if len(old_append_calls) > 1:
            raise AssertionError("last layer should advance visible_lengths inside fused append")
        decode_hot_keys[0, :, decode_step, :].copy_(key_states[0, 0])
        decode_hot_values[0, :, decode_step, :].copy_(value_states[0, 0])

    def fake_lockstep_append_and_advance(
        key_states,
        value_states,
        decode_hot_keys,
        decode_hot_values,
        visible_lengths,
        decode_step,
        local_capacity,
        slide_stride,
    ):
        fused_append_calls.append((int(decode_step), visible_lengths.data_ptr()))
        decode_hot_keys[0, :, decode_step, :].copy_(key_states[0, 0])
        decode_hot_values[0, :, decode_step, :].copy_(value_states[0, 0])
        visible_lengths.add_(1)

    monkeypatch.setattr(cometkv_cache_module, "append_lockstep_local_kv_cache", fake_lockstep_append)
    monkeypatch.setattr(
        cometkv_cache_module,
        "append_lockstep_local_kv_cache_and_advance",
        fake_lockstep_append_and_advance,
        raising=False,
    )

    key_states = torch.randn((1, 1, 2, 128), dtype=DTYPE, device=DEVICE)
    value_states = torch.randn((1, 1, 2, 128), dtype=DTYPE, device=DEVICE)

    cache.decode_update_kv_cache(key_states, value_states, layer_idx=0)
    assert cache.visible_lengths[0].item() == 12

    cache.decode_update_kv_cache(key_states, value_states, layer_idx=1)

    assert old_append_calls == [0]
    assert fused_append_calls == [(0, cache.visible_lengths.data_ptr())]
    assert cache.visible_lengths[0].item() == 13
    assert cache.fixed_prompt_local_layer_visible_lengths_host == [13, 13]


def test_fixed_prompt_local_sparse_attention_stays_sparse_after_sliding_capacity(monkeypatch):
    cache = make_cache(
        layer_num=1,
        max_length=256,
        max_new_length=130,
        static_pattern_end=32,
    )
    cache.prompt_lengths.fill_(64)
    cache.visible_lengths.fill_(64)
    cache.layer_visible_lengths.fill_(64)
    cache.prepare_cache()
    cache.visible_lengths.fill_(64 + 129)
    cache.layer_visible_lengths.fill_(64 + 129)
    queries = torch.randn((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)
    calls = {"flash": 0}

    def fake_topk(_queries, _layer_idx):
        cache.selected_indices_buffer.fill_(0)
        cache.sparse_lengths_buffer.fill_(1)

    def fake_flash(*_args, **_kwargs):
        calls["flash"] += 1
        return torch.zeros((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)

    cache._grouped_query_asym_topk = fake_topk
    cache._static_fixed_concat_flash_attention_from_indices = fake_flash

    cache.sparse_attention(queries, layer_idx=0)

    assert calls == {"flash": 1}


def test_static_fixed_concat_flash_kv_preserves_static_recent_sparse_order():
    cache = make_cache(layer_num=1, max_new_length=5, sig_topk=2)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    rows = cache.batch_groups

    for token in range(cache.static_pattern_total):
        cache.static_keys[0][:, :, token, :].fill_(10 + token)
        cache.static_values[0][:, :, token, :].fill_(110 + token)
    cache.decode_hot_keys[0][:, :, 0, :].fill_(20)
    cache.decode_hot_values[0][:, :, 0, :].fill_(120)
    cache.decode_hot_keys[0][:, :, 1, :].fill_(21)
    cache.decode_hot_values[0][:, :, 1, :].fill_(121)
    cache.cpu_kv_cache[0][:, 1, 0, :].fill_(30)
    cache.cpu_kv_cache[0][:, 1, 1, :].fill_(130)
    cache.cpu_kv_cache[0][:, 2, 0, :].fill_(31)
    cache.cpu_kv_cache[0][:, 2, 1, :].fill_(131)
    cache.fixed_prompt_local_layer_visible_lengths_host[0] = 14
    cache.sparse_lengths_buffer.fill_(2)
    selected_indices = torch.tensor([[1, 2]] * rows, dtype=torch.int32, device=DEVICE)

    flash_keys, flash_values, total_len = cache._build_static_fixed_concat_flash_kv_from_indices(
        0,
        selected_indices,
    )

    expected_keys = [10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 30, 31]
    expected_values = [110, 111, 112, 113, 114, 115, 116, 117, 120, 121, 130, 131]
    assert total_len == len(expected_keys)
    assert flash_keys.shape == (cache.batch_size, total_len, cache.kv_head, cache.head_dim)
    for idx, value in enumerate(expected_keys):
        assert torch.all(flash_keys[:, idx, :, :] == value)
    for idx, value in enumerate(expected_values):
        assert torch.all(flash_values[:, idx, :, :] == value)


def test_static_fixed_concat_flash_attention_matches_reference():
    cache = make_cache(layer_num=1, max_new_length=5, sig_topk=3)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    rows = cache.batch_groups
    queries = torch.randn((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)
    sparse_len = 3
    sparse_keys = torch.randn((rows, sparse_len, cache.head_dim), dtype=DTYPE, device=DEVICE)
    sparse_values = torch.randn_like(sparse_keys)
    recent_len = 2

    cache.static_keys[0].normal_()
    cache.static_values[0].normal_()
    cache.decode_hot_keys[0].normal_()
    cache.decode_hot_values[0].normal_()
    cache.fixed_prompt_local_layer_visible_lengths_host[0] = 12 + recent_len
    for token in range(sparse_len):
        cache.cpu_kv_cache[0][:, token, 0, :].copy_(sparse_keys[:, token, :].cpu())
        cache.cpu_kv_cache[0][:, token, 1, :].copy_(sparse_values[:, token, :].cpu())
    cache.sparse_lengths_buffer.fill_(sparse_len)
    selected_indices = torch.tensor([[0, 1, 2]] * rows, dtype=torch.int32, device=DEVICE)

    out = cache._static_fixed_concat_flash_attention_from_indices(
        queries,
        layer_idx=0,
        selected_indices=selected_indices,
    )
    ref = _reference_flash_merge_attention(
        cache,
        queries,
        0,
        sparse_keys,
        sparse_values,
        torch.full((rows,), recent_len, dtype=torch.int32, device=DEVICE),
        torch.full((rows,), sparse_len, dtype=torch.int32, device=DEVICE),
    )

    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2)


def test_same_length_batch_static_fixed_concat_flash_attention_matches_reference():
    cache = make_cache(
        layer_num=1,
        batch_size=2,
        valid_start=np.array([0, 0], dtype=np.int32),
        max_new_length=5,
        sig_topk=3,
    )
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    rows = cache.batch_groups
    queries = torch.randn((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)
    sparse_len = 3
    recent_len = 2
    sparse_keys = torch.randn((rows, sparse_len, cache.head_dim), dtype=DTYPE, device=DEVICE)
    sparse_values = torch.randn_like(sparse_keys)

    cache.static_keys[0].normal_()
    cache.static_values[0].normal_()
    cache.decode_hot_keys[0].normal_()
    cache.decode_hot_values[0].normal_()
    cache.fixed_prompt_local_layer_visible_lengths_host[0] = 12 + recent_len

    selected_indices = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
            [1, 3, 5],
            [2, 4, 6],
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    for row in range(rows):
        for request_idx, token in enumerate(selected_indices[row].tolist()):
            cache.cpu_kv_cache[0][row, token, 0, :].copy_(sparse_keys[row, request_idx, :].cpu())
            cache.cpu_kv_cache[0][row, token, 1, :].copy_(sparse_values[row, request_idx, :].cpu())
    cache.sparse_lengths_buffer.fill_(sparse_len)

    out = cache._static_fixed_concat_flash_attention_from_indices(
        queries,
        layer_idx=0,
        selected_indices=selected_indices,
    )
    ref = _reference_flash_merge_attention(
        cache,
        queries,
        0,
        sparse_keys,
        sparse_values,
        torch.full((rows,), recent_len, dtype=torch.int32, device=DEVICE),
        torch.full((rows,), sparse_len, dtype=torch.int32, device=DEVICE),
    )

    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2)


def test_static_fixed_concat_flash_attention_ignores_sparse_padding():
    cache = make_cache(layer_num=1, max_new_length=5, sig_topk=3, sig_min_retrieval_topk=0)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    rows = cache.batch_groups
    queries = torch.zeros((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)
    sparse_len = 1
    sparse_keys = torch.zeros((rows, sparse_len, cache.head_dim), dtype=DTYPE, device=DEVICE)
    sparse_values = torch.ones_like(sparse_keys)

    cache.static_keys[0].zero_()
    cache.static_values[0].fill_(1)
    cache.decode_hot_keys[0].zero_()
    cache.decode_hot_values[0].zero_()
    cache.fixed_prompt_local_layer_visible_lengths_host[0] = 12
    cache.cpu_kv_cache[0][:, 0, 0, :].copy_(sparse_keys[:, 0, :].cpu())
    cache.cpu_kv_cache[0][:, 0, 1, :].copy_(sparse_values[:, 0, :].cpu())
    cache.sparse_lengths_buffer.fill_(sparse_len)
    selected_indices = torch.tensor([[0, -1, -1]] * rows, dtype=torch.int32, device=DEVICE)

    out = cache._static_fixed_concat_flash_attention_from_indices(
        queries,
        layer_idx=0,
        selected_indices=selected_indices,
    )
    ref = _reference_flash_merge_attention(
        cache,
        queries,
        0,
        sparse_keys,
        sparse_values,
        torch.zeros((rows,), dtype=torch.int32, device=DEVICE),
        torch.full((rows,), sparse_len, dtype=torch.int32, device=DEVICE),
    )

    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2)


def test_active_sparse_len_uses_cached_host_length_for_plan_buffer():
    cache = make_cache(layer_num=1, max_new_length=5, sig_topk=3, sig_min_retrieval_topk=0)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()

    cache.active_sparse_len_host = 1
    cache.selected_indices_buffer.fill_(0)
    cache.sparse_lengths_buffer.fill_(3)

    assert cache._active_sparse_len_for_selected_indices(cache.selected_indices_buffer) == 1


def test_sparse_attention_prefers_static_fixed_concat_flash_for_single_batch(monkeypatch):
    cache = make_cache(layer_num=1, max_new_length=5, sig_topk=2)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    queries = torch.randn((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)
    calls = {"concat": 0}

    def fake_topk(_queries, _layer_idx):
        cache.selected_indices_buffer.fill_(0)
        cache.sparse_lengths_buffer.fill_(2)

    def fake_concat(*_args, **_kwargs):
        calls["concat"] += 1
        return torch.zeros((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)

    cache._grouped_query_asym_topk = fake_topk
    cache._static_fixed_concat_flash_attention_from_indices = fake_concat

    cache.sparse_attention(queries, layer_idx=0)

    assert calls == {"concat": 1}


def test_static_fixed_concat_flash_skips_sparse_buffer_gather(monkeypatch):
    cache = make_cache(layer_num=1, max_new_length=5, sig_topk=2)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    queries = torch.randn((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)
    calls = {"direct": 0}

    def fake_topk(_queries, _layer_idx):
        cache.selected_indices_buffer.fill_(0)
        cache.sparse_lengths_buffer.fill_(2)

    def fake_direct(_queries, _layer_idx, selected_indices):
        calls["direct"] += 1
        assert selected_indices is cache.selected_indices_buffer
        return torch.zeros((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)

    cache._grouped_query_asym_topk = fake_topk
    cache._static_fixed_concat_flash_attention_from_indices = fake_direct

    cache.sparse_attention(queries, layer_idx=0)

    assert calls == {"direct": 1}


def test_lockstep_batch_sparse_attention_uses_direct_concat_without_segment(monkeypatch):
    cache = make_cache(
        layer_num=1,
        batch_size=2,
        valid_start=np.array([0, 0], dtype=np.int32),
        max_new_length=5,
        sig_topk=2,
    )
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    queries = torch.randn((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)
    calls = {"direct": 0}

    def fake_topk(_queries, _layer_idx):
        cache.selected_indices_buffer.fill_(0)
        cache.sparse_lengths_buffer.fill_(2)

    def fake_direct(_queries, _layer_idx, selected_indices):
        calls["direct"] += 1
        assert selected_indices is cache.selected_indices_buffer
        return torch.zeros((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)

    cache._grouped_query_asym_topk = fake_topk
    cache._static_fixed_concat_flash_attention_from_indices = fake_direct

    cache.sparse_attention(queries, layer_idx=0)

    assert calls == {"direct": 1}


def test_decode_update_uses_layer_device_for_lockstep_append_when_current_device_differs():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")

    torch.cuda.set_device("cuda:0")
    cache = make_cache(layer_num=1, layer_mapping={"0": "cuda:1"})
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()
    cache.context = 12

    key_states = torch.randn((1, 1, 2, 128), dtype=DTYPE, device="cuda:1")
    value_states = torch.randn((1, 1, 2, 128), dtype=DTYPE, device="cuda:1")

    cache.decode_update_kv_cache(key_states, value_states, layer_idx=0)
    torch.cuda.synchronize("cuda:1")

    assert cache.visible_lengths[0].item() == 13
    assert torch.allclose(cache.decode_hot_keys[0][0, :, 0, :], key_states[0, 0])


def test_refresh_static_recent_state_guards_kernel_onto_primary_device():
    # Regression: the fused refresh kernel used to launch on the AMBIENT device; with the
    # cache on cuda:1 and ambient cuda:0 it dereferenced foreign pointers (illegal memory
    # access on multi-GPU hosts). The fast path must guard onto primary_device and match
    # the device-safe Python fallback bit-for-bit.
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")

    torch.cuda.set_device("cuda:0")
    cache = make_cache(layer_num=1, layer_mapping={"0": "cuda:1"})
    seq = 24
    cache.prompt_lengths.fill_(seq)
    cache.visible_lengths.fill_(seq)
    cache.layer_visible_lengths.fill_(seq)
    cache.prepare_cache()
    torch.manual_seed(3)
    cache.prompt_recent_keys[0].normal_()
    cache.prompt_recent_values[0].normal_()

    cache._refresh_static_recent_state(0, cache.visible_lengths)
    torch.cuda.synchronize("cuda:1")
    fast_lengths = cache.static_lengths_buffer.clone()
    fast_keys = cache.static_keys[0].clone()
    fast_values = cache.static_values[0].clone()

    cache.static_lengths_buffer.zero_()
    cache.static_keys[0].zero_()
    cache.static_values[0].zero_()
    # CPU visible_lengths fails the fast-path device condition -> Python fallback loop.
    cache._refresh_static_recent_state(0, cache.visible_lengths.cpu())
    torch.cuda.synchronize("cuda:1")

    assert torch.equal(fast_lengths, cache.static_lengths_buffer)
    assert torch.equal(fast_keys, cache.static_keys[0])
    assert torch.equal(fast_values, cache.static_values[0])


def test_random_projection_matches_seeded_baseline_and_is_shared_across_layers():
    cache_a = make_cache(layer_num=2)
    cache_b = make_cache(layer_num=2)
    reference = build_reference_random_orth_projection().to(device=DEVICE)

    assert torch.allclose(cache_a.signature_projections[0], reference)
    assert torch.equal(cache_a.signature_projections[0], cache_a.signature_projections[1])
    assert torch.equal(cache_a.signature_projections[0], cache_b.signature_projections[0])


def _reference_flash_merge_attention(cache, queries, layer_idx, sparse_keys, sparse_values, recent_lengths, sparse_lengths):
    rows = cache.batch_groups
    outputs = torch.zeros((rows, cache.group_size, cache.head_dim), dtype=DTYPE, device=DEVICE)
    scale = 1.0 / math.sqrt(cache.head_dim)
    static_rows_k = cache.static_keys[layer_idx].view(rows, cache.static_pattern_total, cache.head_dim)
    static_rows_v = cache.static_values[layer_idx].view(rows, cache.static_pattern_total, cache.head_dim)
    recent_rows_k = cache.decode_hot_keys[layer_idx].view(rows, cache._decode_hot_capacity(), cache.head_dim)
    recent_rows_v = cache.decode_hot_values[layer_idx].view(rows, cache._decode_hot_capacity(), cache.head_dim)

    for row in range(rows):
        static_k = static_rows_k[row]
        static_v = static_rows_v[row]
        recent_k = recent_rows_k[row, : int(recent_lengths[row].item())]
        recent_v = recent_rows_v[row, : int(recent_lengths[row].item())]
        sparse_k = sparse_keys[row, : int(sparse_lengths[row].item())]
        sparse_v = sparse_values[row, : int(sparse_lengths[row].item())]
        merged_k = torch.cat((static_k, recent_k, sparse_k), dim=0).float()
        merged_v = torch.cat((static_v, recent_v, sparse_v), dim=0).float()
        scores = torch.matmul(queries.view(rows, cache.group_size, cache.head_dim)[row].float(), merged_k.transpose(0, 1)) * scale
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        outputs[row] = torch.matmul(probs, merged_v).to(dtype=DTYPE)

    return outputs.view(cache.batch_size, 1, cache.num_heads, cache.head_dim)


def test_sparse_attention_prefers_flash_merge_on_static_prompt_retrieval():
    cache = make_cache(layer_num=1, max_new_length=5)
    cache.prompt_lengths.fill_(12)
    cache.visible_lengths.fill_(12)
    cache.layer_visible_lengths.fill_(12)
    cache.prepare_cache()

    queries = torch.randn((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)
    called = {"flash": False}

    def fake_flash(*args, **kwargs):
        called["flash"] = True
        return torch.zeros((cache.batch_size, 1, cache.num_heads, cache.head_dim), dtype=DTYPE, device=DEVICE)

    def fake_topk(_queries, _layer_idx):
        cache.selected_indices_buffer.fill_(0)
        cache.sparse_lengths_buffer.fill_(1)

    cache._static_fixed_concat_flash_attention_from_indices = fake_flash
    cache._grouped_query_asym_topk = fake_topk
    cache.signature_index[0].zero_()

    cache.sparse_attention(queries, layer_idx=0)

    assert called["flash"] is True


def test_default_selector_allocates_no_score_key_index():
    cache = make_cache(layer_num=1)
    assert cache.selector_mode == "asym_n8"
    assert not hasattr(cache, "score_key_index")


def _unpack_pm1(packed, bits):
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    b = ((packed.unsqueeze(-1) >> shifts) & 1).view(*packed.shape[:-1], -1)[..., :bits]
    return b.to(torch.float32) * 2 - 1


def test_asym_n8_selector_kernel_matches_torch_reference():
    seq_len = 96
    cache = make_cache(layer_num=1, max_length=128, sig_selector="asym_n8", sig_topk=9,
                       sig_min_retrieval_topk=0, stats_mode="frozen", query_aggregation="q_sum")
    q = torch.randn((1, seq_len, 4, 128), dtype=DTYPE, device=DEVICE)
    k = torch.randn((1, seq_len, 2, 128), dtype=DTYPE, device=DEVICE)
    v = torch.randn_like(k)
    cache.prefill_update_kv_cache(q, k, v, layer_idx=0, start_bdx=0)
    cache.sync(0, 0)
    cache.prompt_lengths.fill_(seq_len)
    cache.visible_lengths.fill_(seq_len)
    cache.layer_visible_lengths.fill_(seq_len)
    cache.prepare_cache()
    torch.cuda.synchronize()

    queries = torch.randn((1, 1, 4, 128), dtype=DTYPE, device=DEVICE)
    cache._grouped_query_asym_topk(queries, 0)
    torch.cuda.synchronize()

    start, end = cache.plan_range1_start_host, cache.plan_range1_end_host
    kk = cache.active_sparse_len_host
    P = build_reference_random_orth_projection().to(DEVICE)
    qsum = queries.view(1, 2, 2, 128).sum(dim=2, dtype=torch.float32).view(2, 128)
    x = qsum @ P.t()[:, :120]
    sig = cache.signature_index[0].view(2, -1, 16)
    pm1 = _unpack_pm1(sig, 120)
    deq = (cache.sig_norm_lo[0].unsqueeze(1)
           + sig[..., 15].float() * cache.sig_norm_step[0].unsqueeze(1)).exp()
    ref_scores = torch.einsum("rb,rtb->rt", x, pm1) * deq

    scores_buf = cache._asym_scores_buffers[str(queries.device)]
    got_scores = scores_buf[:, start:end]
    assert torch.allclose(got_scores, ref_scores[:, start:end], atol=1e-3, rtol=1e-4)

    ref_sel = (ref_scores[:, start:end].topk(kk, dim=1).indices + start).sort(dim=1).values
    got_sel = cache.selected_indices_buffer[:, :kk].long().sort(dim=1).values
    assert torch.equal(got_sel, ref_sel)


def test_asym_n8_prefill_packs_sign_bits_and_norm_byte():
    seq_len = 40
    cache = make_cache(layer_num=1, max_length=64, sig_selector="asym_n8")
    q = torch.randn((1, seq_len, 4, 128), dtype=DTYPE, device=DEVICE)
    k = torch.randn((1, seq_len, 2, 128), dtype=DTYPE, device=DEVICE)
    v = torch.randn_like(k)
    cache.prefill_update_kv_cache(q, k, v, layer_idx=0, start_bdx=0)
    cache.sync(0, 0)
    torch.cuda.synchronize()

    P = build_reference_random_orth_projection().to(DEVICE)
    keys = k.transpose(1, 2).reshape(2, seq_len, 128).float()
    # P0 centering: signatures/norms are built from k - mu, mu frozen from the prompt.
    mu_ref = keys.mean(dim=1, keepdim=True)
    assert torch.allclose(cache.sig_key_mean[0], mu_ref.squeeze(1), atol=1e-4)
    centered = keys - mu_ref
    bits_ref = (torch.matmul(centered, P.t()) >= 0).to(torch.uint8)[..., :120]
    sig = cache.signature_index[0].view(2, -1, 16)[:, :seq_len, :]
    pm1 = _unpack_pm1(sig, 120)
    assert torch.equal((pm1 > 0).to(torch.uint8), bits_ref)

    norms = centered.norm(dim=-1).clamp_min(1e-6).log()
    lo = cache.sig_norm_lo[0]
    step = cache.sig_norm_step[0]
    assert torch.allclose(lo, norms.min(dim=1).values, atol=1e-4)
    code_ref = torch.round((norms - lo.unsqueeze(1)) / step.unsqueeze(1)).clamp(0, 255).to(torch.uint8)
    assert torch.equal(sig[..., 15], code_ref)


def test_asym_n8_decode_selects_norm_weighted_topk():
    seq_len = 96
    cache = make_cache(layer_num=1, max_length=128, sig_selector="asym_n8", sig_topk=7,
                       sig_min_retrieval_topk=0, stats_mode="frozen", query_aggregation="q_sum")
    q = torch.randn((1, seq_len, 4, 128), dtype=DTYPE, device=DEVICE)
    k = torch.randn((1, seq_len, 2, 128), dtype=DTYPE, device=DEVICE)
    v = torch.randn_like(k)
    cache.prefill_update_kv_cache(q, k, v, layer_idx=0, start_bdx=0)
    cache.sync(0, 0)
    cache.prompt_lengths.fill_(seq_len)
    cache.visible_lengths.fill_(seq_len)
    cache.layer_visible_lengths.fill_(seq_len)
    cache.prepare_cache()
    torch.cuda.synchronize()

    queries = torch.randn((1, 1, 4, 128), dtype=DTYPE, device=DEVICE)
    cache._grouped_query_asym_topk(queries, 0)
    torch.cuda.synchronize()

    start, end = cache.plan_range1_start_host, cache.plan_range1_end_host
    kk = cache.active_sparse_len_host
    P = build_reference_random_orth_projection().to(DEVICE)
    qsum = queries.view(1, 2, 2, 128).sum(dim=2, dtype=torch.float32).view(2, 128)
    x = qsum @ P.t()[:, :120]
    sig = cache.signature_index[0].view(2, -1, 16)
    pm1 = _unpack_pm1(sig, 120)
    deq = (cache.sig_norm_lo[0].unsqueeze(1)
           + sig[..., 15].float() * cache.sig_norm_step[0].unsqueeze(1)).exp()
    ref_scores = torch.einsum("rb,rtb->rt", x, pm1) * deq

    ref_sel = (ref_scores[:, start:end].topk(kk, dim=1).indices + start).sort(dim=1).values
    got_sel = cache.selected_indices_buffer[:, :kk].long().sort(dim=1).values
    assert torch.equal(got_sel, ref_sel)


def test_asym_n8_score_policy_gather_evicts_worst_priority_way_not_lru(monkeypatch):
    monkeypatch.setenv("COMETKV_TOKEN_CACHE_POLICY", "score")
    monkeypatch.setenv("COMETKV_TOKEN_CACHE_WAYS", "4")
    monkeypatch.setenv("COMETKV_TOKEN_CACHE_MULT", "1")
    cache = make_cache(
        layer_num=1,
        max_length=64,
        sig_selector="asym_n8",
        sig_token_cache_size=16,
    )
    cache.prompt_lengths.fill_(32)
    cache.visible_lengths.fill_(32)
    cache.layer_visible_lengths.fill_(32)
    cache.prepare_cache()

    # One four-way set: token 4 is LRU, while token 8 has the worst current asym_n8 score.
    # Inserting token 20 must therefore evict token 8, not token 4.
    resident_ids = torch.tensor([4, 8, 12, 16], dtype=torch.int32, device=DEVICE)
    cache.token_cache_ids[0][:, :4].copy_(resident_ids.unsqueeze(0).expand(cache.batch_groups, -1))
    cache.token_cache_stamps[0][:, :4].copy_(
        torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=DEVICE)
    )
    cache.lockstep_decode_step_host = 10
    scores = cache._asym_scores_for(0, DEVICE)
    scores[:, resident_ids.long()] = torch.tensor(
        [5.0, -3.0, 7.0, 9.0], dtype=torch.float32, device=DEVICE
    )
    scores[:, 20] = 10.0
    prio_buckets, prio_scores = cache._token_cache_prio_args(0)
    assert prio_buckets is None
    assert prio_scores is scores

    cache.selected_indices_buffer.fill_(-1)
    cache.selected_indices_buffer[:, 0] = 20
    cache.active_sparse_len_host = 1
    cache.sparse_lengths_buffer.fill_(1)
    cache._build_static_fixed_concat_flash_kv_from_indices(0, cache.selected_indices_buffer)
    torch.cuda.synchronize()

    assert set(cache.token_cache_ids[0][0, :4].tolist()) == {4, 12, 16, 20}


def _make_sampled_cache(monkeypatch, frac, selector="asym_n8", seq_len=96, sig_topk=12, clip=0.0):
    monkeypatch.setenv("COMETKV_SELECTOR", selector)
    monkeypatch.setenv("COMETKV_SAMPLE_FRAC", str(frac))
    monkeypatch.setenv("COMETKV_SAMPLE_TAU", "1.0")
    monkeypatch.setenv("COMETKV_SAMPLE_SEED", "77")
    monkeypatch.setenv("COMETKV_SAMPLE_MIN_M", "1")   # tiny test k -> disable the guard
    # Pin the clip: the repo default is 4.0, but the contract oracles below model the
    # UNCLIPPED estimator unless a test opts in explicitly (see the mean-cap clip test).
    monkeypatch.setenv("COMETKV_SAMPLE_CLIP", str(clip))
    torch.manual_seed(5)
    cache = make_cache(layer_num=1, max_length=128, sig_selector=selector, sig_topk=sig_topk,
                       sig_min_retrieval_topk=0)
    q = torch.randn((1, seq_len, 4, 128), dtype=DTYPE, device=DEVICE)
    k = torch.randn((1, seq_len, 2, 128), dtype=DTYPE, device=DEVICE)
    v = torch.randn_like(k)
    cache.prefill_update_kv_cache(q, k, v, layer_idx=0, start_bdx=0)
    cache.sync(0, 0)
    cache.prompt_lengths.fill_(seq_len)
    cache.visible_lengths.fill_(seq_len)
    cache.layer_visible_lengths.fill_(seq_len)
    cache.prepare_cache()
    torch.cuda.synchronize()
    return cache


def test_sampled_tail_is_additional_to_head_budget(monkeypatch):
    cache0 = _make_sampled_cache(monkeypatch, 0.0)
    total_k = cache0.active_sparse_len_host
    assert cache0.active_sample_len_host == 0
    cache = _make_sampled_cache(monkeypatch, 0.25)
    assert cache.active_sample_len_host == int(round(0.25 * total_k))
    assert cache.active_sparse_len_host == total_k
    assert cache.active_retrieval_len_host == total_k + cache.active_sample_len_host


def test_sampled_tail_decode_matches_contract_oracle(monkeypatch):
    # Full eager decode step with sampling on, checked against a contract-derived fp32
    # reference: (a) the draws must equal inverse-CDF sampling of the proposal rebuilt from
    # the full score buffer plus uniform mixture; (b) the returned attention must equal ONE
    # softmax over [main concat rows] U [sampled slots with -log(m*q_j) corrected logits] —
    # which independently validates both the correction wiring and the LSE merge identity.
    monkeypatch.setenv("COMETKV_SAMPLE_AUTOSCALE", "1")
    cache = _make_sampled_cache(monkeypatch, 0.25)
    m, kh = cache.active_sample_len_host, cache.active_sparse_len_host
    assert m > 0
    torch.manual_seed(9)
    queries = torch.randn((1, 1, 4, 128), dtype=DTYPE, device=DEVICE)
    out = cache.sparse_attention(queries, 0)
    torch.cuda.synchronize()

    st = cache._sample_state[str(queries.device)]
    scores = cache._asym_scores_buffers[str(queries.device)]
    tokens = scores.size(1)
    ch = cache._sample_chunk
    tokens_pad = (tokens + ch - 1) // ch * ch
    prop = torch.full((scores.size(0), tokens_pad), float("-inf"), device=scores.device)
    assert cache.sample_autoscale is True
    finite = torch.isfinite(scores)
    count = finite.sum(dim=1, keepdim=True).clamp_min(1)
    finite_scores = torch.where(finite, scores, torch.zeros_like(scores))
    mean = finite_scores.sum(dim=1, keepdim=True) / count
    variance = (
        torch.where(finite, scores - mean, torch.zeros_like(scores)) ** 2
    ).sum(dim=1, keepdim=True) / count
    scale = (cache.sample_sigma / cache.sample_tau) / variance.sqrt().clamp_min(1e-6)
    torch.mul(scores - mean, scale, out=prop[:, :tokens])
    head = cache.selected_indices_buffer[:, :kh].long()
    draws_rt = st["draws"].long()
    # Contract (a): draws cover all candidates, including the owner's head. The merge
    # excludes the CURRENT head without changing proposal probabilities.
    assert int(draws_rt.max()) < tokens
    assert torch.isfinite(torch.gather(prop, 1, draws_rt)).all(), "sampled a -inf slot"
    # Contract (b): the stored correction equals -log(m * P(draw)) under the actual proposal
    # Mirror normalization; full support and exclusion of the current head are also required.
    w = (prop - prop.amax(dim=1, keepdim=True)).exp()
    w.div_(w.sum(1, keepdim=True)).mul_(1 - cache.sample_uniform_mix)
    w[:, cache.plan_range1_start_host:cache.plan_range1_end_host].add_(
        cache.sample_uniform_mix / cache.active_candidates_host)
    z = w.view(w.size(0), -1, ch).sum(dim=2).cumsum(dim=1)[:, -1:].clamp_min(1e-30)
    q_sel_ref = (torch.gather(w, 1, draws_rt) / z).clamp_min(1e-30)
    corr_ref = -(math.log(m) + torch.log(q_sel_ref))
    torch.testing.assert_close(st["corr"], corr_ref, atol=1e-4, rtol=1e-5)
    corr = st["corr"]

    static_len = int(cache.fixed_prompt_local_static_length_host)
    recent_len = cache._lockstep_fixed_prompt_recent_length_host(0)
    main_len = static_len + recent_len + kh
    rows, dim = cache.batch_groups, cache.head_dim
    k_main = cache._static_fixed_concat_rows_view(
        cache.static_fixed_concat_flash_key_storage[0], main_len).reshape(rows, main_len, dim).float()
    v_main = cache._static_fixed_concat_rows_view(
        cache.static_fixed_concat_flash_value_storage[0], main_len).reshape(rows, main_len, dim).float()
    # single-layer test cache -> the batched window path stores layer 0's tail in wk/wv[0]
    k_tail, v_tail = st["wk"][0].float(), st["wv"][0].float()
    qg = queries.reshape(rows, cache.group_size, dim).float()
    lg_main = torch.bmm(qg, k_main.transpose(1, 2)) / math.sqrt(dim)
    lg_tail = torch.bmm(qg, k_tail.transpose(1, 2)) / math.sqrt(dim) + corr.unsqueeze(1)
    overlap = (draws_rt[:, :, None] == head[:, None, :]).any(-1)
    lg_tail.masked_fill_(overlap[:, None], -float("inf"))
    w = torch.softmax(torch.cat((lg_main, lg_tail), dim=-1), dim=-1)
    ref = torch.bmm(w, torch.cat((v_main, v_tail), dim=1))
    torch.testing.assert_close(out.float().reshape(rows, cache.group_size, dim), ref,
                               atol=3e-2, rtol=3e-2)


def test_sampled_tail_clip_caps_at_mean_plus_clip(monkeypatch):
    # The truncated-IS clip (default COMETKV_SAMPLE_CLIP=4.0) caps each tail logit at the
    # per-(row,head) mean of the m JOINT logits (q.k/sqrt(d) + corr) plus clip nats, inside
    # the fused kernel, before the joint softmax. Mirror that in the fp32 oracle with a clip
    # tight enough to actually bite on this workload (margins over the mean stay ~1-2 nats,
    # so the 4.0 default never engages here).
    clip = 0.5
    cache = _make_sampled_cache(monkeypatch, 0.25, clip=clip)
    kh = cache.active_sparse_len_host
    torch.manual_seed(9)
    queries = torch.randn((1, 1, 4, 128), dtype=DTYPE, device=DEVICE)
    out = cache.sparse_attention(queries, 0)
    torch.cuda.synchronize()

    st = cache._sample_state[str(queries.device)]
    corr = st["corr"]
    static_len = int(cache.fixed_prompt_local_static_length_host)
    recent_len = cache._lockstep_fixed_prompt_recent_length_host(0)
    main_len = static_len + recent_len + kh
    rows, dim = cache.batch_groups, cache.head_dim
    k_main = cache._static_fixed_concat_rows_view(
        cache.static_fixed_concat_flash_key_storage[0], main_len).reshape(rows, main_len, dim).float()
    v_main = cache._static_fixed_concat_rows_view(
        cache.static_fixed_concat_flash_value_storage[0], main_len).reshape(rows, main_len, dim).float()
    k_tail, v_tail = st["wk"][0].float(), st["wv"][0].float()
    qg = queries.reshape(rows, cache.group_size, dim).float()
    lg_main = torch.bmm(qg, k_main.transpose(1, 2)) / math.sqrt(dim)
    lg_tail = torch.bmm(qg, k_tail.transpose(1, 2)) / math.sqrt(dim) + corr.unsqueeze(1)
    head = cache.selected_indices_buffer[:, :kh].long()
    valid = ~(st["draws"].long()[:, :, None] == head[:, None, :]).any(-1)
    cap = (lg_tail.masked_fill(~valid[:, None], 0).sum(-1, keepdim=True)
           / valid.sum(-1)[:, None, None].clamp_min(1)) + clip
    lg_tail.masked_fill_(~valid[:, None], -float("inf"))
    assert (lg_tail > cap).any(), "clip must actually bite on this workload"
    lg_tail = torch.minimum(lg_tail, cap)
    w = torch.softmax(torch.cat((lg_main, lg_tail), dim=-1), dim=-1)
    ref = torch.bmm(w, torch.cat((v_main, v_tail), dim=1))
    torch.testing.assert_close(out.float().reshape(rows, cache.group_size, dim), ref,
                               atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("selector", ["asym_n8"])
def test_sampled_tail_is_seed_reproducible(monkeypatch, selector):
    # asym_n8 also covers the AUTOSCALE z-scored proposal path (default-on for the asym
    # family, and asym_n8 is the repo-default selector with sampling now on by default).
    torch.manual_seed(9)
    queries = torch.randn((1, 1, 4, 128), dtype=DTYPE, device=DEVICE)
    outs = []
    for _ in range(2):
        cache = _make_sampled_cache(monkeypatch, 0.25, selector=selector)
        outs.append(cache.sparse_attention(queries.clone(), 0))
        torch.cuda.synchronize()
    assert torch.equal(outs[0], outs[1]), "same seed must reproduce draws and output"
