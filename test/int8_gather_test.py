"""Correctness test for the int8 concat-gather kernel (K per-channel, V per-token dequant).

Verifies: (1) static/recent regions copied verbatim; (2) sparse miss path dequantizes to match a
Python reference; (3) the token cache stores dequantized values so a second identical gather (cache
hit) reproduces the first result.

Run: conda run -n cometkv python test/int8_gather_test.py
"""
import torch
from cometkv import concat_static_recent_lookup_gather_uva_kv_update_cache_int8

DEV = "cuda"
DTYPE = torch.float16


def quant_k_per_channel(K):  # K: [rows, T, dim] -> int8 codes, scale [rows, dim]
    amax = K.abs().amax(dim=1).clamp(min=1e-8)          # [rows, dim]
    scale = amax / 127.0
    codes = torch.clamp(torch.round(K / scale.unsqueeze(1)), -127, 127).to(torch.int8)
    return codes, scale


def quant_v_per_token(V):    # V: [rows, T, dim] -> int8 codes, scale [rows, T]
    amax = V.abs().amax(dim=-1).clamp(min=1e-8)          # [rows, T]
    scale = amax / 127.0
    codes = torch.clamp(torch.round(V / scale.unsqueeze(-1)), -127, 127).to(torch.int8)
    return codes, scale


def main():
    torch.manual_seed(0)
    rows, request_count, cache_size, total_tokens = 2, 4, 8, 16
    static_cap, recent_cap, static_len, recent_len, dim = 5, 6, 2, 3, 128
    total_len = static_len + recent_len + request_count

    static_keys = torch.randn(rows, static_cap, dim, dtype=DTYPE, device=DEV)
    static_values = torch.randn_like(static_keys)
    recent_keys = torch.randn(rows, recent_cap, dim, dtype=DTYPE, device=DEV)
    recent_values = torch.randn_like(recent_keys)

    # full-precision retrieval KV, then quantize the way the runtime will
    Kf = torch.randn(rows, total_tokens, dim, dtype=DTYPE, device=DEV)
    Vf = torch.randn(rows, total_tokens, dim, dtype=DTYPE, device=DEV)
    k_codes, k_scale = quant_k_per_channel(Kf)
    v_codes, v_scale = quant_v_per_token(Vf)
    deq_K = (k_codes.float() * k_scale.unsqueeze(1)).to(DTYPE)   # reference dequant
    deq_V = (v_codes.float() * v_scale.unsqueeze(-1)).to(DTYPE)

    cpu_kv_int8 = torch.empty(rows, total_tokens, 2, dim, dtype=torch.int8, pin_memory=True)
    cpu_kv_int8[:, :, 0, :].copy_(k_codes)
    cpu_kv_int8[:, :, 1, :].copy_(v_codes)
    k_scale_gpu = k_scale.to(DEV).float().contiguous()
    v_scale_pin = v_scale.float().cpu().pin_memory().contiguous()

    request_token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32, device=DEV)
    cache_token_ids = torch.full((rows, cache_size), -1, dtype=torch.int32, device=DEV)
    cache_locks = torch.zeros((rows, cache_size), dtype=torch.int32, device=DEV)
    cache_keys = torch.zeros((rows, cache_size, dim), dtype=DTYPE, device=DEV)
    cache_values = torch.zeros_like(cache_keys)

    def gather():
        fk = torch.empty((rows, total_len, 1, dim), dtype=DTYPE, device=DEV)
        fv = torch.empty_like(fk)
        hm = torch.zeros((rows, request_count), dtype=torch.int32, device=DEV)
        concat_static_recent_lookup_gather_uva_kv_update_cache_int8(
            static_keys, static_values, recent_keys, recent_values, request_token_ids,
            cpu_kv_int8, k_scale_gpu, v_scale_pin, fk, fv, hm,
            cache_token_ids, cache_locks, cache_keys, cache_values,
            static_len, recent_len, request_count)
        torch.cuda.synchronize()
        return fk, fv, hm

    # ---- first gather: all misses, dequant from cpu int8 ----
    fk, fv, hm = gather()

    # static / recent must be copied verbatim
    assert torch.equal(fk[:, :static_len, 0], static_keys[:, :static_len]), "static K mismatch"
    assert torch.equal(fk[:, static_len:static_len+recent_len, 0], recent_keys[:, :recent_len]), "recent K mismatch"
    assert torch.equal(fv[:, static_len:static_len+recent_len, 0], recent_values[:, :recent_len]), "recent V mismatch"

    # sparse region must equal the reference dequant of the requested tokens
    max_k_err = max_v_err = 0.0
    for r in range(rows):
        for j, tok in enumerate(request_token_ids[r].tolist()):
            dst = static_len + recent_len + j
            max_k_err = max(max_k_err, (fk[r, dst, 0].float() - deq_K[r, tok].float()).abs().max().item())
            max_v_err = max(max_v_err, (fv[r, dst, 0].float() - deq_V[r, tok].float()).abs().max().item())
    assert hm.sum().item() == 0, f"expected all misses, hit_mask={hm}"
    print(f"first gather (all miss): max|out-ref_dequant| K={max_k_err:.3e} V={max_v_err:.3e} (must be 0)")
    assert max_k_err == 0 and max_v_err == 0, "dequant output != python reference dequant"

    # ---- second gather: same requests -> cache hits, must reproduce result ----
    fk2, fv2, hm2 = gather()
    assert torch.equal(fk, fk2) and torch.equal(fv, fv2), "cache-hit result differs from miss result"
    assert hm2.sum().item() == rows * request_count, f"expected all hits, hit_mask={hm2}"
    print(f"second gather (all hit): identical to first = {torch.equal(fk, fk2) and torch.equal(fv, fv2)}, "
          f"hits={hm2.sum().item()}/{rows*request_count}")

    # end-to-end vs the FULL-PRECISION (unquantized) values: shows the quantization error magnitude
    rel_k = sum(((fk[r, static_len+recent_len+j, 0].float() - Kf[r, tok].float()).norm() / Kf[r, tok].float().norm()).item()
                for r in range(rows) for j, tok in enumerate(request_token_ids[r].tolist())) / (rows*request_count)
    rel_v = sum(((fv[r, static_len+recent_len+j, 0].float() - Vf[r, tok].float()).norm() / Vf[r, tok].float().norm()).item()
                for r in range(rows) for j, tok in enumerate(request_token_ids[r].tolist())) / (rows*request_count)
    print(f"int8 quant error vs fp16 truth: K rel-L2={rel_k*100:.3f}%  V rel-L2={rel_v*100:.3f}%")
    print("PASS")


if __name__ == "__main__":
    main()
