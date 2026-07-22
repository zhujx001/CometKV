import pytest
import torch

try:
    from cometkv import concat_static_recent_gpu_gather
except ImportError:  # pragma: no cover - extension not built
    concat_static_recent_gpu_gather = None

DTYPE = torch.bfloat16

pytestmark = pytest.mark.skipif(
    concat_static_recent_gpu_gather is None or not torch.cuda.is_available(),
    reason="cometkv extension or CUDA unavailable",
)


def _stack_gpu_kv(keys, values):
    # [rows, tokens, 2, dim]: K and V of one token adjacent, device memory
    return torch.stack((keys, values), dim=2).contiguous()


def test_gpu_gather_wide_out_rows_matches_exact_width_and_leaves_tail():
    # Same contract as the UVA kernel (test_cometkv_gather.py): out rows wider than
    # [static|recent|sparse] must produce an identical prefix and untouched tail, so the
    # CometKV_GPU CUDA-graph path can gather straight into its fixed-width concat buffer.
    rows = 2
    request_count = 4
    total_tokens = 16
    static_capacity = 5
    recent_capacity = 6
    static_len = 2
    recent_len = 3
    dim = 128
    total_len = static_len + recent_len + request_count
    out_row_len = total_len + 7

    static_keys = torch.randn((rows, static_capacity, dim), dtype=DTYPE, device="cuda")
    static_values = torch.randn_like(static_keys)
    recent_keys = torch.randn((rows, recent_capacity, dim), dtype=DTYPE, device="cuda")
    recent_values = torch.randn_like(recent_keys)
    gpu_keys = torch.randn((rows, total_tokens, dim), dtype=DTYPE, device="cuda")
    gpu_values = torch.randn_like(gpu_keys)
    gpu_kv = _stack_gpu_kv(gpu_keys, gpu_values)
    request_token_ids = torch.tensor([[1, 2, 3, -1], [5, 6, 7, 8]], dtype=torch.int32, device="cuda")

    def run(width):
        out_keys = torch.full((rows, width, 1, dim), 7, dtype=DTYPE, device="cuda")
        out_values = torch.full_like(out_keys, 13)
        concat_static_recent_gpu_gather(
            static_keys, static_values, recent_keys, recent_values,
            request_token_ids, gpu_kv,
            out_keys, out_values,
            static_len, recent_len, request_count,
        )
        torch.cuda.synchronize()
        return out_keys, out_values

    exact_k, exact_v = run(total_len)
    wide_k, wide_v = run(out_row_len)

    # exact-width output itself must match the reference layout
    assert torch.equal(exact_k[:, :static_len, 0], static_keys[:, :static_len])
    assert torch.equal(
        exact_k[:, static_len:static_len + recent_len, 0], recent_keys[:, :recent_len]
    )
    assert torch.equal(exact_k[0, static_len + recent_len:, 0][:3], gpu_keys[0, 1:4])
    assert torch.equal(
        exact_k[0, static_len + recent_len + 3, 0], torch.zeros(dim, dtype=DTYPE, device="cuda")
    )  # negative id zero-fills

    assert torch.equal(wide_k[:, :total_len], exact_k)
    assert torch.equal(wide_v[:, :total_len], exact_v)
    assert torch.equal(wide_k[:, total_len:], torch.full_like(wide_k[:, total_len:], 7))
    assert torch.equal(wide_v[:, total_len:], torch.full_like(wide_v[:, total_len:], 13))
