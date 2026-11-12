import sys
from pathlib import Path

import pytest
import torch

KERNEL_LIB = Path(__file__).resolve().parents[1]
if str(KERNEL_LIB) not in sys.path:
    sys.path.insert(0, str(KERNEL_LIB))

import cometkv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DIM = 128


def _window_inputs(n_layers=3, rows=4, m=37, total_tokens=500, seed=0):
    torch.manual_seed(seed)
    # Pinned host store per layer: [rows, total_tokens, 2 (K|V), dim] bf16, read over UVA.
    stores = [torch.randn((rows, total_tokens, 2, DIM), dtype=torch.bfloat16).pin_memory()
              for _ in range(n_layers)]
    ptrs = torch.tensor([s.data_ptr() for s in stores], dtype=torch.int64, device="cuda")
    draws = torch.randint(0, total_tokens, (rows, m), dtype=torch.int32, device="cuda")
    draws[:, :8] = draws[:, 8:16]          # duplicate draws, as sampling with replacement produces
    draws[0, -1] = -1                      # padding slots are skipped
    return stores, ptrs, draws


def _empty_out(n_layers, rows, m):
    shape = (n_layers, rows, m, DIM)
    return (torch.zeros(shape, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(shape, dtype=torch.bfloat16, device="cuda"))


def _leader_slots(draws):
    # Same contract as cometkv_cache._duplicate_leader_slots: every slot points at a slot with
    # the same token, and each distinct token has exactly one self-pointing leader.
    srt, perm = draws.sort(dim=1, stable=True)
    first = torch.ones_like(srt, dtype=torch.bool)
    first[:, 1:] = srt[:, 1:] != srt[:, :-1]
    pos = torch.arange(srt.size(1), device=draws.device).expand_as(srt)
    lead_pos = torch.where(first, pos, torch.zeros_like(pos)).cummax(dim=1).values
    slots = torch.empty_like(draws)
    slots.scatter_(1, perm, torch.gather(perm, 1, lead_pos).to(draws.dtype))
    return slots


def test_persist_window_gather_is_exported():
    assert hasattr(cometkv, "uva_gather_kv_rows_window_persist")


@pytest.mark.parametrize("n_blocks", [1, 8, 64])
def test_persist_window_gather_matches_window_gather(n_blocks):
    n_layers, rows, m, total = 3, 4, 37, 500
    stores, ptrs, draws = _window_inputs(n_layers, rows, m, total)
    ref_k, ref_v = _empty_out(n_layers, rows, m)
    cometkv.uva_gather_kv_rows_window(draws, ptrs, ref_k, ref_v, m, total)
    out_k, out_v = _empty_out(n_layers, rows, m)
    cometkv.uva_gather_kv_rows_window_persist(draws, ptrs, out_k, out_v, m, total, n_blocks)
    torch.cuda.synchronize()
    assert torch.equal(out_k, ref_k)
    assert torch.equal(out_v, ref_v)
    # Spot-check the reference against the host store itself.
    row, j = 1, 5
    tok = int(draws[row, j])
    assert torch.equal(ref_k[2, row, j].cpu(), stores[2][row, tok, 0])
    assert torch.equal(ref_v[2, row, j].cpu(), stores[2][row, tok, 1])


def test_persist_window_gather_dedup_fills_duplicates_from_leader():
    n_layers, rows, m, total = 2, 3, 64, 300
    _stores, ptrs, draws = _window_inputs(n_layers, rows, m, total, seed=1)
    ref_k, ref_v = _empty_out(n_layers, rows, m)
    cometkv.uva_gather_kv_rows_window(draws, ptrs, ref_k, ref_v, m, total)
    out_k, out_v = _empty_out(n_layers, rows, m)
    cometkv.uva_gather_kv_rows_window_persist(draws, ptrs, out_k, out_v, m, total, 8, _leader_slots(draws))
    torch.cuda.synchronize()
    assert torch.equal(out_k, ref_k)
    assert torch.equal(out_v, ref_v)


def test_persist_window_gather_rejects_capacity_mismatch():
    n_layers, rows, m, total = 1, 2, 16, 100
    _stores, ptrs, draws = _window_inputs(n_layers, rows, m, total)
    out_k, out_v = _empty_out(n_layers, rows, m + 1)
    with pytest.raises(RuntimeError, match="m == indices/out_k capacity"):
        cometkv.uva_gather_kv_rows_window_persist(draws, ptrs, out_k, out_v, m, total, 8)
