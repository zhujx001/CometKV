"""Stage-1 gate: cache-side CUDA-graph sparse_attention path.

(a) _cg_sparse_attention (fixed-W [static|sparse|recent]+cache_seqlens) matches eager sparse_attention.
(b) Capture _cg_sparse_attention into a CUDA graph, replay with a DIFFERENT query (driver sets the
    device cache_seqlens), and confirm it matches eager-with-that-query — the discriminating test.

Run: conda run -n cometkv python test/cg_stage1_test.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, numpy as np
from cache_hub.cometkv_cache import cometkv_cache

DEV, DT = "cuda", torch.float16


def build(prompt_len, max_new, layer_num, n_heads, kv_head, head_dim, budget, quant="none"):
    return cometkv_cache(
        valid_start=np.array([0]), layer_num=layer_num, batch_size=1,
        num_key_value_heads=kv_head, num_heads=n_heads, head_dim=head_dim, dtype=DT,
        layer_mapping={str(l): DEV for l in range(layer_num)}, max_length=prompt_len + max_new,
        max_new_length=max_new, static_pattern_start=4, static_pattern_end=64, retrieval_budget=budget,
        sig_bits=128, sig_topk=0, sig_chunk_size=131072, sig_seed=1234, sig_mode="random_orth",
        sig_token_cache_size=1024, prefill_bsz=1, num_gpus=1, model_size=8, sig_min_retrieval_topk=16,
        exclude_preserved_from_budget=True, core=0, cpu_kv_quant=quant,
    )


def live_seqlen(c, layer):
    static_len = int(c.fixed_prompt_local_static_length_host)
    recent_len = c._lockstep_fixed_prompt_recent_length_host(layer)
    sparse_len = c.active_sparse_len_host
    return static_len + sparse_len + recent_len


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def run(quant="none", prompt_len=4096, max_new=40, budget=0.1):
    torch.manual_seed(0)
    n_heads, kv_head, head_dim, layer = 32, 8, 128, 0
    c = build(prompt_len, max_new, 1, n_heads, kv_head, head_dim, budget, quant)
    q = torch.randn(1, prompt_len, kv_head, head_dim, dtype=DT, device=DEV)
    k = torch.randn(1, prompt_len, kv_head, head_dim, dtype=DT, device=DEV)
    v = torch.randn(1, prompt_len, kv_head, head_dim, dtype=DT, device=DEV)
    c.prefill_update_kv_cache(q, k, v, 0, 0); c.sync(0, 0); c.prepare_cache()
    kd = torch.randn(1, 1, kv_head, head_dim, dtype=DT, device=DEV)
    vd = torch.randn(1, 1, kv_head, head_dim, dtype=DT, device=DEV)
    for _ in range(12):
        c.decode_update_kv_cache(kd, vd, 0)
    torch.cuda.synchronize()

    # (a) eager vs cg on the same query
    qd = torch.randn(1, 1, n_heads, head_dim, dtype=DT, device=DEV)
    c.use_cuda_graph = False
    out_eager = c.sparse_attention(qd, layer).float().clone()
    c.use_cuda_graph = True
    c._cg_cache_seqlens.fill_(live_seqlen(c, layer))
    out_cg = c.sparse_attention(qd, layer).float().clone()
    print(f"[{quant}] (a) cg vs eager same-query: cos={cos(out_eager, out_cg):.5f} "
          f"rel-L2={(out_eager-out_cg).norm()/out_eager.norm()*100:.2f}%")

    # (b) capture cg path, replay with NEW query + driver-set cache_seqlens, compare to eager
    qbuf = qd.clone()  # static query input buffer the graph reads
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            c._cg_cache_seqlens.fill_(live_seqlen(c, layer))
            c._cg_sparse_attention(qbuf, layer)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            cg_out = c._cg_sparse_attention(qbuf, layer)
    except Exception as e:
        print(f"[{quant}] (b) CAPTURE FAILED: {type(e).__name__}: {str(e)[:100]}"); return
    ok = True
    for trial in range(4):
        qnew = torch.randn(1, 1, n_heads, head_dim, dtype=DT, device=DEV)
        qbuf.copy_(qnew)
        c._cg_cache_seqlens.fill_(live_seqlen(c, layer))
        g.replay(); torch.cuda.synchronize()
        c.use_cuda_graph = False
        ref = c.sparse_attention(qnew, layer).float()
        c.use_cuda_graph = True
        cc = cos(ref, cg_out.float())
        ok = ok and cc > 0.99
        print(f"[{quant}] (b) replay trial {trial}: cos(graph,eager)={cc:.5f}")
    print(f"[{quant}] => stage-1 capture+replay correct: {ok}")


if __name__ == "__main__":
    run("none")
    print()
    run("int8")
