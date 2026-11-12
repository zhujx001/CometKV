"""Bisect the CUDA-graph replay drift at the CACHE level (no model, fast).

Captures a full per-step cache decode op for all layers — [cg-append (device step counter) + sparse_attention]
— then replays it across MANY steps with the driver advancing host counters (cg_advance_host_step) and
setting device inputs (cg_set_step_inputs + fixed kd/vd/q buffers). Compares the per-step attention outputs
to a cg-EAGER reference run (same inputs). cg_stage1 only captured sparse_attention with state pre-advanced;
this adds the captured append + the growing recent window across replays — the untested piece.

Run: conda run -n cometkv python test/cg_bisect_cache.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, numpy as np
from cache_hub.cometkv_cache import cometkv_cache

DEV, DT = "cuda", torch.float16


def build(prompt_len, max_new, layer_num, n_heads, kv_head, head_dim, budget):
    return cometkv_cache(
        valid_start=np.array([0]), layer_num=layer_num, batch_size=1,
        num_key_value_heads=kv_head, num_heads=n_heads, head_dim=head_dim, dtype=DT,
        layer_mapping={str(l): DEV for l in range(layer_num)}, max_length=prompt_len + max_new,
        max_new_length=max_new, static_pattern_start=32, static_pattern_end=64, retrieval_budget=budget,
        sig_bits=128, sig_topk=0, sig_chunk_size=131072, sig_seed=1234, sig_mode="random_orth",
        sig_token_cache_size=1024, prefill_bsz=1, num_gpus=1, model_size=8, sig_min_retrieval_topk=16,
        exclude_preserved_from_budget=True, core=0, cpu_kv_quant="none",
    )


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def main(prompt_len=4096, layer_num=2, n_heads=32, kv_head=8, head_dim=128, budget=0.1, nsteps=20):
    torch.manual_seed(0)
    # fixed per-step input buffers the graph reads (driver fills each step)
    qd = torch.zeros(1, 1, n_heads, head_dim, dtype=DT, device=DEV)
    kd = torch.zeros(1, 1, kv_head, head_dim, dtype=DT, device=DEV)
    vd = torch.zeros(1, 1, kv_head, head_dim, dtype=DT, device=DEV)
    # pre-generate the per-step inputs so eager and graph see identical sequences
    qs = [torch.randn(1, 1, n_heads, head_dim, dtype=DT, device=DEV) for _ in range(nsteps)]
    ks = [torch.randn(1, 1, kv_head, head_dim, dtype=DT, device=DEV) for _ in range(nsteps)]
    vs = [torch.randn(1, 1, kv_head, head_dim, dtype=DT, device=DEV) for _ in range(nsteps)]

    def fresh_cache():
        torch.manual_seed(0)
        c = build(prompt_len, nsteps + 8, layer_num, n_heads, kv_head, head_dim, budget)
        q = torch.randn(1, prompt_len, kv_head, head_dim, dtype=DT, device=DEV)
        k = torch.randn(1, prompt_len, kv_head, head_dim, dtype=DT, device=DEV)
        v = torch.randn(1, prompt_len, kv_head, head_dim, dtype=DT, device=DEV)
        for l in range(layer_num):
            c.prefill_update_kv_cache(q, k, v, l, 0); c.sync(l, 0)
        c.prepare_cache(); torch.cuda.synchronize()
        return c

    def step(c, qd_s, kd_s, vd_s):
        # one decode step over all layers (cg mode): append + attention per layer; returns last-layer out
        c.cg_set_step_inputs()
        outs = []
        for l in range(layer_num):
            c.decode_update_kv_cache(kd_s, vd_s, l)
            outs.append(c.sparse_attention(qd_s, l))
        c.cg_advance_host_step()
        return outs

    # ---- cg-EAGER reference ----
    c1 = fresh_cache(); c1.use_cuda_graph = True
    ref = []
    for s in range(nsteps):
        outs = step(c1, qs[s], ks[s], vs[s])
        ref.append([o.float().clone() for o in outs])

    # ---- capture + replay ----
    c2 = fresh_cache(); c2.use_cuda_graph = True
    for ldx in range(layer_num):
        c2._ensure_cg_concat_buffers(ldx)
    # warmup a couple cg-eager steps to match c1's state progression AND warm flash/concat
    for s in range(3):
        step(c2, qs[s], ks[s], vs[s])
    torch.cuda.synchronize()
    # capture step `3`: fixed buffers qd/kd/vd; driver sets them + cg_set_step_inputs before replay
    def graph_step():
        outs = []
        for l in range(layer_num):
            c2.decode_update_kv_cache(kd, vd, l)
            outs.append(c2.sparse_attention(qd, l))
        return outs
    # warmup on default stream then capture
    qd.copy_(qs[3]); kd.copy_(ks[3]); vd.copy_(vs[3]); c2.cg_set_step_inputs()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cap_outs = graph_step()
    c2.cg_advance_host_step()
    # the capture executed step 3; compare to ref[3]
    drift = []
    cap_cos = min(cos(cap_outs[l], ref[3][l]) for l in range(layer_num))
    drift.append((3, cap_cos))
    # replay steps 4..nsteps-1
    for s in range(4, nsteps):
        qd.copy_(qs[s]); kd.copy_(ks[s]); vd.copy_(vs[s])
        c2.cg_set_step_inputs()
        g.replay(); torch.cuda.synchronize()
        c2.cg_advance_host_step()
        c = min(cos(cap_outs[l], ref[s][l]) for l in range(layer_num))
        drift.append((s, c))
    print(f"prompt={prompt_len} layers={layer_num} budget={budget}  capture step=3, replay 4..{nsteps-1}")
    for s, c in drift:
        tag = "CAPTURE" if s == 3 else "replay "
        flag = "" if c > 0.99 else "   <-- DRIFT"
        print(f"  {tag} step {s:>3}: min cos(graph,cg-eager)={c:.5f}{flag}")


if __name__ == "__main__":
    main()
