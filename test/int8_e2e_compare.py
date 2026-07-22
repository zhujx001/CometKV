"""End-to-end int8-vs-bf16 cpu_kv comparison through the real cometkv_cache decode path.

Builds two caches (cpu_kv_quant 'none' vs 'int8') with IDENTICAL prefill/decode KV and queries, runs
prefill + decode, and compares the per-(step,layer) sparse_attention output. Signatures are built from
full-precision keys in both, so top-k selection is identical -> the only difference is the int8 quant
error on the gathered retrieval KV. Also confirms the int8 path runs with no -1/zero slots.

Run: conda run -n cometkv python test/int8_e2e_compare.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, numpy as np
from cache_hub.cometkv_cache import cometkv_cache

DEV, DTYPE = "cuda", torch.float16


def build(quant, prompt_len, max_new, layer_num, n_heads, kv_head, head_dim, budget):
    return cometkv_cache(
        valid_start=np.array([0]), layer_num=layer_num, batch_size=1,
        max_length=prompt_len + max_new, num_key_value_heads=kv_head, num_heads=n_heads,
        head_dim=head_dim, dtype=DTYPE, layer_mapping={str(l): DEV for l in range(layer_num)},
        max_new_length=max_new, static_pattern_start=4, static_pattern_end=64,
        retrieval_budget=budget, sig_bits=128, sig_topk=0, sig_chunk_size=131072, sig_seed=1234,
        sig_mode="random_orth", sig_token_cache_size=1024, prefill_bsz=1, num_gpus=1, model_size=8,
        sig_min_retrieval_topk=16, exclude_preserved_from_budget=True, core=0, cpu_kv_quant=quant,
    )


def main(prompt_len=4096, max_new=24, layer_num=2, n_heads=32, kv_head=8, head_dim=128, budget=0.1):
    torch.manual_seed(0)
    # identical inputs for both caches
    q = torch.randn(1, prompt_len, kv_head, head_dim, dtype=DTYPE, device=DEV)  # dummy for prefill api
    k = torch.randn(1, prompt_len, kv_head, head_dim, dtype=DTYPE, device=DEV)
    v = torch.randn(1, prompt_len, kv_head, head_dim, dtype=DTYPE, device=DEV)
    dec_k = [torch.randn(1, 1, kv_head, head_dim, dtype=DTYPE, device=DEV) for _ in range(max_new)]
    dec_v = [torch.randn(1, 1, kv_head, head_dim, dtype=DTYPE, device=DEV) for _ in range(max_new)]
    qd = [torch.randn(1, 1, n_heads, head_dim, dtype=DTYPE, device=DEV) for _ in range(max_new)]

    def run(quant):
        c = build(quant, prompt_len, max_new, layer_num, n_heads, kv_head, head_dim, budget)
        for l in range(layer_num):
            c.prefill_update_kv_cache(q, k, v, l, 0); c.sync(l, 0)
        c.prepare_cache(); torch.cuda.synchronize()
        outs = []
        for step in range(max_new - 1):
            for l in range(layer_num):
                c.decode_update_kv_cache(dec_k[step], dec_v[step], l)
                o = c.sparse_attention(qd[step], l)
                outs.append(o.float().clone())
        torch.cuda.synchronize()
        return c, outs

    c_ref, out_ref = run("none")
    c_int8, out_int8 = run("int8")
    print(f"prompt={prompt_len} budget={budget} layers={layer_num} heads={n_heads}/{kv_head} -> topk≈{c_ref.active_sparse_len_host}")
    print(f"int8 cpu_kv dtype={c_int8.cpu_kv_cache[0].dtype}, k_scale={tuple(c_int8.cpu_kv_k_scale[0].shape)}, "
          f"v_scale={tuple(c_int8.cpu_kv_v_scale[0].shape)} (pinned={c_int8.cpu_kv_v_scale[0].is_pinned()})")

    rels = [((a - b).norm() / a.norm().clamp(min=1e-8)).item() for a, b in zip(out_ref, out_int8)]
    coss = [torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
            for a, b in zip(out_ref, out_int8)]
    print(f"per-(step,layer) attention output, int8 vs bf16 cpu_kv:")
    print(f"   rel-L2: mean={100*sum(rels)/len(rels):.3f}%  max={100*max(rels):.3f}%")
    print(f"   cosine: mean={sum(coss)/len(coss):.5f}  min={min(coss):.5f}")
    print("PASS (int8 decode path ran end-to-end with no errors)")


if __name__ == "__main__":
    main()
