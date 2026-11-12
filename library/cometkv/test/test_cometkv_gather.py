import pytest
import torch

from cometkv import (
    append_lockstep_local_kv_cache,
    append_lockstep_local_kv_cache_and_advance,
    concat_static_recent_lookup_gather_uva_kv_update_cache,
    refresh_static_prompt_recent_state,
)


if not torch.cuda.is_available():
    pytest.skip("CometKV gather tests require a CUDA GPU.", allow_module_level=True)


DTYPE = torch.bfloat16


def _stack_cpu_kv(cpu_keys, cpu_values):
    return torch.stack((cpu_keys, cpu_values), dim=2).contiguous().pin_memory()


def _primary_cache_slot(token, cache_size):
    if cache_size <= 1:
        return 0
    return (token % ((cache_size + 1) >> 1)) << 1


def _cache_contains(cache_token_ids, row, token):
    primary = _primary_cache_slot(token, cache_token_ids.size(1))
    secondary = min(primary + 1, cache_token_ids.size(1) - 1)
    return (
        int(cache_token_ids[row, primary].item()) == token
        or int(cache_token_ids[row, secondary].item()) == token
    )


def test_append_lockstep_local_kv_cache_compacts_without_token_ids():
    batch_size = 2
    kv_heads = 2
    dim = 128
    local_capacity = 192
    slide_stride = 128
    decode_step = 128
    key_states = torch.full((batch_size, 1, kv_heads, dim), 9, dtype=DTYPE, device="cuda")
    value_states = torch.full_like(key_states, 109)
    local_keys = torch.empty((batch_size, kv_heads, local_capacity, dim), dtype=DTYPE, device="cuda")
    local_values = torch.empty_like(local_keys)
    for slot in range(local_capacity):
        local_keys[:, :, slot, :].fill_(slot)
        local_values[:, :, slot, :].fill_(slot + 100)

    append_lockstep_local_kv_cache(
        key_states,
        value_states,
        local_keys,
        local_values,
        decode_step,
        local_capacity,
        slide_stride,
    )

    assert torch.equal(local_keys[0, 0, 0], torch.full((dim,), 64, dtype=DTYPE, device="cuda"))
    assert torch.equal(local_keys[0, 0, 63], torch.full((dim,), 127, dtype=DTYPE, device="cuda"))
    assert torch.equal(local_keys[0, 0, 64], torch.full((dim,), 9, dtype=DTYPE, device="cuda"))
    assert torch.equal(local_values[1, 1, 64], torch.full((dim,), 109, dtype=DTYPE, device="cuda"))


def test_append_lockstep_local_kv_cache_and_advance_updates_visible_lengths_once_per_batch():
    batch_size = 2
    kv_heads = 2
    dim = 128
    local_capacity = 192
    slide_stride = 128
    decode_step = 128
    key_states = torch.full((batch_size, 1, kv_heads, dim), 11, dtype=DTYPE, device="cuda")
    value_states = torch.full_like(key_states, 111)
    local_keys = torch.empty((batch_size, kv_heads, local_capacity, dim), dtype=DTYPE, device="cuda")
    local_values = torch.empty_like(local_keys)
    visible_lengths = torch.tensor([200, 300], dtype=torch.int32, device="cuda")
    for slot in range(local_capacity):
        local_keys[:, :, slot, :].fill_(slot)
        local_values[:, :, slot, :].fill_(slot + 100)

    append_lockstep_local_kv_cache_and_advance(
        key_states,
        value_states,
        local_keys,
        local_values,
        visible_lengths,
        decode_step,
        local_capacity,
        slide_stride,
    )

    assert torch.equal(local_keys[0, 0, 0], torch.full((dim,), 64, dtype=DTYPE, device="cuda"))
    assert torch.equal(local_keys[0, 0, 63], torch.full((dim,), 127, dtype=DTYPE, device="cuda"))
    assert torch.equal(local_keys[0, 0, 64], torch.full((dim,), 11, dtype=DTYPE, device="cuda"))
    assert torch.equal(local_values[1, 1, 64], torch.full((dim,), 111, dtype=DTYPE, device="cuda"))
    assert torch.equal(visible_lengths, torch.tensor([201, 301], dtype=torch.int32, device="cuda"))


def test_refresh_static_prompt_recent_state_matches_python_reference():
    batch_size = 1
    kv_heads = 2
    static_pattern_start = 4
    static_pattern_end = 4
    static_total = static_pattern_start + static_pattern_end
    dim = 128

    prompt_lengths = torch.tensor([12], dtype=torch.int32, device="cuda")
    visible_lengths = torch.tensor([14], dtype=torch.int32, device="cuda")
    prompt_recent_keys = torch.randn((batch_size, kv_heads, static_pattern_end, dim), dtype=DTYPE, device="cuda")
    prompt_recent_values = torch.randn_like(prompt_recent_keys)
    static_keys = torch.zeros((batch_size, kv_heads, static_total, dim), dtype=DTYPE, device="cuda")
    static_values = torch.zeros_like(static_keys)
    static_lengths = torch.zeros((batch_size * kv_heads,), dtype=torch.int32, device="cuda")

    refresh_static_prompt_recent_state(
        prompt_lengths,
        visible_lengths,
        prompt_recent_keys,
        prompt_recent_values,
        static_keys,
        static_values,
        static_lengths,
        static_pattern_start,
        static_pattern_end,
    )

    assert torch.equal(static_lengths, torch.full((batch_size * kv_heads,), 6, dtype=torch.int32, device="cuda"))
    assert torch.equal(static_keys[:, :, :static_pattern_start, :], torch.zeros_like(static_keys[:, :, :static_pattern_start, :]))
    assert torch.equal(static_values[:, :, :static_pattern_start, :], torch.zeros_like(static_values[:, :, :static_pattern_start, :]))
    assert torch.equal(static_keys[:, :, static_pattern_start:static_pattern_start + 2, :], prompt_recent_keys[:, :, 2:4, :])
    assert torch.equal(static_values[:, :, static_pattern_start:static_pattern_start + 2, :], prompt_recent_values[:, :, 2:4, :])
    assert torch.equal(static_keys[:, :, static_pattern_start + 2:, :], torch.zeros_like(static_keys[:, :, static_pattern_start + 2:, :]))


def test_concat_static_recent_lookup_gather_writes_final_flash_buffer_and_updates_cache():
    rows = 2
    request_count = 4
    cache_size = 8
    total_tokens = 16
    static_capacity = 5
    recent_capacity = 6
    static_len = 2
    recent_len = 3
    dim = 128
    total_len = static_len + recent_len + request_count

    static_keys = torch.randn((rows, static_capacity, dim), dtype=DTYPE, device="cuda")
    static_values = torch.randn_like(static_keys)
    recent_keys = torch.randn((rows, recent_capacity, dim), dtype=DTYPE, device="cuda")
    recent_values = torch.randn_like(recent_keys)
    cpu_keys = torch.randn((rows, total_tokens, dim), dtype=DTYPE, pin_memory=True)
    cpu_values = torch.randn((rows, total_tokens, dim), dtype=DTYPE, pin_memory=True)
    cpu_kv = _stack_cpu_kv(cpu_keys, cpu_values)
    request_token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32, device="cuda")

    cache_token_ids = torch.full((rows, cache_size), -1, dtype=torch.int32, device="cuda")
    cache_locks = torch.zeros((rows, cache_size), dtype=torch.int32, device="cuda")
    cache_keys = torch.zeros((rows, cache_size, dim), dtype=DTYPE, device="cuda")
    cache_values = torch.zeros_like(cache_keys)
    for row, token in enumerate((1, 5)):
        slot = _primary_cache_slot(token, cache_size)
        cache_token_ids[row, slot] = token
        cache_keys[row, slot].copy_(cpu_keys[row, token].to("cuda"))
        cache_values[row, slot].copy_(cpu_values[row, token].to("cuda"))

    fused_keys = torch.empty((rows, total_len, 1, dim), dtype=DTYPE, device="cuda")
    fused_values = torch.empty_like(fused_keys)
    hit_mask = torch.zeros((rows, request_count), dtype=torch.int32, device="cuda")

    concat_static_recent_lookup_gather_uva_kv_update_cache(
        static_keys,
        static_values,
        recent_keys,
        recent_values,
        request_token_ids,
        cpu_kv,
        fused_keys,
        fused_values,
        hit_mask,
        cache_token_ids,
        cache_locks,
        cache_keys,
        cache_values,
        static_len,
        recent_len,
        request_count,
    )
    torch.cuda.synchronize()

    expected_keys = torch.empty_like(fused_keys)
    expected_values = torch.empty_like(fused_values)
    expected_keys[:, :static_len, 0].copy_(static_keys[:, :static_len])
    expected_values[:, :static_len, 0].copy_(static_values[:, :static_len])
    expected_keys[:, static_len:static_len + recent_len, 0].copy_(recent_keys[:, :recent_len])
    expected_values[:, static_len:static_len + recent_len, 0].copy_(recent_values[:, :recent_len])
    for row in range(rows):
        for request_idx, token in enumerate(request_token_ids[row].tolist()):
            dst = static_len + recent_len + request_idx
            expected_keys[row, dst, 0].copy_(cpu_keys[row, token].to("cuda"))
            expected_values[row, dst, 0].copy_(cpu_values[row, token].to("cuda"))

    assert torch.equal(fused_keys, expected_keys)
    assert torch.equal(fused_values, expected_values)
    assert torch.equal(hit_mask, torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.int32, device="cuda"))
    for row in range(rows):
        for token in request_token_ids[row].tolist():
            assert _cache_contains(cache_token_ids, row, int(token))


def test_concat_static_recent_lookup_gather_zeroes_invalid_sparse_slots():
    rows = 2
    request_count = 4
    cache_size = 8
    total_tokens = 16
    static_capacity = 5
    recent_capacity = 6
    static_len = 2
    recent_len = 3
    dim = 128
    total_len = static_len + recent_len + request_count

    static_keys = torch.randn((rows, static_capacity, dim), dtype=DTYPE, device="cuda")
    static_values = torch.randn_like(static_keys)
    recent_keys = torch.randn((rows, recent_capacity, dim), dtype=DTYPE, device="cuda")
    recent_values = torch.randn_like(recent_keys)
    cpu_keys = torch.randn((rows, total_tokens, dim), dtype=DTYPE, pin_memory=True)
    cpu_values = torch.randn((rows, total_tokens, dim), dtype=DTYPE, pin_memory=True)
    cpu_kv = _stack_cpu_kv(cpu_keys, cpu_values)
    request_token_ids = torch.tensor([[1, -1, 3, 4], [-1, 6, 7, 8]], dtype=torch.int32, device="cuda")

    cache_token_ids = torch.full((rows, cache_size), -1, dtype=torch.int32, device="cuda")
    cache_locks = torch.zeros((rows, cache_size), dtype=torch.int32, device="cuda")
    cache_keys = torch.zeros((rows, cache_size, dim), dtype=DTYPE, device="cuda")
    cache_values = torch.zeros_like(cache_keys)
    hit_mask = torch.zeros((rows, request_count), dtype=torch.int32, device="cuda")

    fused_keys = torch.full((rows, total_len, 1, dim), 7, dtype=DTYPE, device="cuda")
    fused_values = torch.full_like(fused_keys, 13)
    concat_static_recent_lookup_gather_uva_kv_update_cache(
        static_keys,
        static_values,
        recent_keys,
        recent_values,
        request_token_ids,
        cpu_kv,
        fused_keys,
        fused_values,
        hit_mask,
        cache_token_ids,
        cache_locks,
        cache_keys,
        cache_values,
        static_len,
        recent_len,
        request_count,
    )

    first_invalid = static_len + recent_len + 1
    second_invalid = static_len + recent_len
    assert torch.equal(fused_keys[0, first_invalid, 0], torch.zeros((dim,), dtype=DTYPE, device="cuda"))
    assert torch.equal(fused_values[0, first_invalid, 0], torch.zeros((dim,), dtype=DTYPE, device="cuda"))
    assert torch.equal(fused_keys[1, second_invalid, 0], torch.zeros((dim,), dtype=DTYPE, device="cuda"))
    assert torch.equal(fused_values[1, second_invalid, 0], torch.zeros((dim,), dtype=DTYPE, device="cuda"))
    assert torch.equal(hit_mask, torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.int32, device="cuda"))


def test_concat_gather_wide_out_rows_matches_exact_width_and_leaves_tail():
    # out rows wider than [static|recent|sparse] (out_row_len > total_len): the produced prefix
    # must be identical to the exact-width call and the tail beyond total_len untouched, so the
    # CUDA-graph path can gather straight into its fixed-width concat buffer.
    rows = 2
    request_count = 4
    cache_size = 8
    total_tokens = 16
    static_capacity = 5
    recent_capacity = 6
    static_len = 2
    recent_len = 3
    dim = 128
    total_len = static_len + recent_len + request_count
    out_row_len = total_len + 9

    static_keys = torch.randn((rows, static_capacity, dim), dtype=DTYPE, device="cuda")
    static_values = torch.randn_like(static_keys)
    recent_keys = torch.randn((rows, recent_capacity, dim), dtype=DTYPE, device="cuda")
    recent_values = torch.randn_like(recent_keys)
    cpu_keys = torch.randn((rows, total_tokens, dim), dtype=DTYPE, pin_memory=True)
    cpu_values = torch.randn((rows, total_tokens, dim), dtype=DTYPE, pin_memory=True)
    cpu_kv = _stack_cpu_kv(cpu_keys, cpu_values)
    request_token_ids = torch.tensor([[1, 2, 3, -1], [5, 6, 7, 8]], dtype=torch.int32, device="cuda")

    def run(width):
        cache_token_ids = torch.full((rows, cache_size), -1, dtype=torch.int32, device="cuda")
        cache_locks = torch.zeros((rows, cache_size), dtype=torch.int32, device="cuda")
        cache_keys = torch.zeros((rows, cache_size, dim), dtype=DTYPE, device="cuda")
        cache_values = torch.zeros_like(cache_keys)
        out_keys = torch.full((rows, width, 1, dim), 7, dtype=DTYPE, device="cuda")
        out_values = torch.full_like(out_keys, 13)
        hit_mask = torch.zeros((rows, request_count), dtype=torch.int32, device="cuda")
        concat_static_recent_lookup_gather_uva_kv_update_cache(
            static_keys, static_values, recent_keys, recent_values,
            request_token_ids, cpu_kv,
            out_keys, out_values, hit_mask,
            cache_token_ids, cache_locks, cache_keys, cache_values,
            static_len, recent_len, request_count,
        )
        torch.cuda.synchronize()
        return out_keys, out_values

    exact_k, exact_v = run(total_len)
    wide_k, wide_v = run(out_row_len)

    assert torch.equal(wide_k[:, :total_len], exact_k)
    assert torch.equal(wide_v[:, :total_len], exact_v)
    assert torch.equal(wide_k[:, total_len:], torch.full_like(wide_k[:, total_len:], 7))
    assert torch.equal(wide_v[:, total_len:], torch.full_like(wide_v[:, total_len:], 13))
