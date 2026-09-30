"""Diagnose frozen CometKV statistics on full-attention Llama trajectories.

This is an offline selector study, not an end-to-end sparse-decoding benchmark.
It generates two small workloads, replays their tokens with full attention, and
compares selectors on identical post-RoPE Q/K without invoking sparse decoding.
For this statistics ablation, k is fixed from the initial prompt; it does not
reproduce the runtime's growing requested budget and near-prompt buffer cap.
"""

import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import time

import torch


def encode(keys, mean, projection, bounds=None, margin=0.0):
    residual = keys - mean[:, None, :]
    log_norm = residual.norm(dim=-1).clamp_min(1e-6).log()
    if bounds is None:
        lo, hi = log_norm.amin(dim=1), log_norm.amax(dim=1)
        span = (hi - lo).clamp_min(1e-6)
        lo, hi = lo - margin * span, hi + margin * span
        step = ((hi - lo) / 255).clamp_min(1e-8)
    else:
        lo, step = bounds
    code = ((log_norm - lo[:, None]) / step[:, None]).round().clamp(0, 255)
    reconstructed = (lo[:, None] + code * step[:, None]).exp()
    signs = (residual @ projection.T >= 0).float().mul_(2).sub_(1)
    return signs * reconstructed[..., None], (lo, step)


def eviction_blocks(prompt_length, token_count, stride=128, overlap=32):
    # The current runtime's first eviction is stride-overlap tokens; later
    # evictions have stride tokens. Only use complete, already sealed blocks.
    blocks = []
    start = prompt_length
    end = prompt_length + stride - overlap
    while end + overlap < token_count:
        blocks.append((start, end))
        start, end = end, end + stride
    return blocks


def analyze_layer(keys_cpu, queries_cpu, positions, prompt_length, args):
    device = args.device
    keys = keys_cpu.to(device).float()  # [kv_heads, tokens, dim]
    queries = queries_cpu.to(device).float()  # [kv_heads, group, probes, dim]
    heads, tokens, dim = keys.shape
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    basis, _ = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
    projection = basis[:, :120].T.contiguous().to(device)
    # E_P[||x|| (Pq).sign(Px)] = c * q.x for Haar orthogonal P.
    score_scale = 120 * math.exp(
        math.lgamma(dim / 2) - 0.5 * math.log(math.pi) - math.lgamma((dim + 1) / 2)
    )
    mean0 = keys[:, :prompt_length].mean(dim=1)
    rms0 = (keys[:, :prompt_length] - mean0[:, None]).square().sum(-1).mean(1).sqrt()
    first, bounds = encode(keys[:, :prompt_length], mean0, projection)
    margin_first, margin_bounds = encode(keys[:, :prompt_length], mean0, projection, margin=0.3)
    methods = {
        "frozen": [first], "margin_0.3": [margin_first], "forward_ema_0.1": [first],
        "block_norm": [first], "block_center_uncorrected": [first],
        "block_center_corrected": [first],
    }
    centers = [mean0[:, None].expand(-1, prompt_length, -1)]
    moving = mean0.clone()
    drift = []
    blocks = eviction_blocks(prompt_length, tokens)
    for start, end in blocks:
        batch = keys[:, start:end]
        block_mean = batch.mean(dim=1)
        logs = (batch - mean0[:, None]).norm(dim=-1).clamp_min(1e-6).log()
        lo, step = bounds
        for head in range(heads):
            drift.append({
                "head": head, "start": start, "end": end,
                "mean_shift_over_prompt_rms": float((block_mean[head] - mean0[head]).norm() / rms0[head].clamp_min(1e-6)),
                "mean_cosine": float(torch.nn.functional.cosine_similarity(block_mean[head], mean0[head], dim=0)),
                "norm_clip_high": float((logs[head] > lo[head] + 255 * step[head] + 1e-6).float().mean()),
                "norm_clip_low": float((logs[head] < lo[head] - 1e-6).float().mean()),
            })
        frozen, _ = encode(batch, mean0, projection, bounds)
        margin, _ = encode(batch, mean0, projection, margin_bounds)
        ema, _ = encode(batch, moving, projection, bounds)
        norm_only, _ = encode(batch, mean0, projection)
        centered, _ = encode(batch, block_mean, projection)
        for name, encoded in (
            ("frozen", frozen), ("margin_0.3", margin), ("forward_ema_0.1", ema),
            ("block_norm", norm_only), ("block_center_uncorrected", centered),
            ("block_center_corrected", centered),
        ):
            methods[name].append(encoded)
        centers.append(block_mean[:, None].expand(-1, end - start, -1))
        moving.lerp_(block_mean, 0.1)  # historical forward-only EMA updated AFTER encoding
    encodings = {name: torch.cat(chunks, dim=1) for name, chunks in methods.items()}
    centers = torch.cat(centers, dim=1)
    # Isolate statistics changes using a fixed prompt-derived diagnostic k.
    # The runtime instead recomputes requested k on slide steps, then caps it
    # by selected_indices_buffer width. Do not label this a runtime replay.
    budget = max(16, int(prompt_length * args.budget))
    records = []
    for probe, position in enumerate(positions):
        end = max([prompt_length] + [hi for _, hi in blocks if hi + 32 <= position])
        if end == prompt_length:
            continue
        candidates = keys[:, 4:end]
        qheads = queries[:, :, probe]
        qsum = qheads.sum(dim=1)
        exact = torch.einsum("hd,htd->ht", qsum, candidates)
        k = min(budget, end - 4)
        oracle = exact.topk(k, dim=1).indices
        # Mass is conditional on the retrieval pool, averaged over the actual
        # GQA query heads. It is not the mass of complete sparse attention.
        probabilities = torch.einsum("hgd,htd->hgt", qheads, candidates).div(math.sqrt(dim)).softmax(-1).mean(1)
        mass_oracle = probabilities.topk(k, dim=1).indices
        mass_oracle_mask = torch.zeros_like(exact, dtype=torch.bool).scatter_(1, mass_oracle, True)
        projected_q = qsum @ projection.T
        oracle_mask = torch.zeros_like(exact, dtype=torch.bool).scatter_(1, oracle, True)
        scores = {
            name: torch.einsum("hm,htm->ht", projected_q, encoded[:, 4:end])
            for name, encoded in encodings.items()
        }
        scores["block_center_corrected"] = scores["block_center_corrected"] + score_scale * torch.einsum("hd,htd->ht", qsum, centers[:, 4:end])
        projected_heads = qheads @ projection.T
        for name in ("frozen", "block_center_corrected"):
            head_scores = torch.einsum("hgm,htm->hgt", projected_heads, encodings[name][:, 4:end])
            if name == "block_center_corrected":
                head_scores = head_scores + score_scale * torch.einsum("hgd,htd->hgt", qheads, centers[:, 4:end])
            scores[name + "_head_sum"] = head_scores.sum(1)
            logits = head_scores / (score_scale * math.sqrt(dim))
            scores[name + "_mean_prob"] = torch.logsumexp(logits.log_softmax(-1), dim=1) - math.log(qheads.size(1))
        # Ideal synchronous rebuild at this checkpoint, using only past keys.
        rebuilt, _ = encode(keys[:, :end], keys[:, :end].mean(1), projection)
        scores["full_rebuild_oracle"] = torch.einsum("hm,htm->ht", projected_q, rebuilt[:, 4:end])
        scores["exact_group_topk"] = exact
        scores["exact_mass_topk"] = probabilities
        for name, score in scores.items():
            selected = score.topk(k, dim=1).indices
            recall = oracle_mask.gather(1, selected).float().mean(1)
            mass_recall = mass_oracle_mask.gather(1, selected).float().mean(1)
            mass = probabilities.gather(1, selected).sum(1)
            new_share = ((selected + 4) >= prompt_length).float().mean(1)
            equivalence = {}
            if name.endswith("_head_sum"):
                reference = scores[name.removesuffix("_head_sum")]
                reference_selection = reference.topk(k, dim=1).indices
                reference_mask = torch.zeros_like(exact, dtype=torch.bool).scatter_(1, reference_selection, True)
                equivalence = {
                    "linear_max_abs_error": (score - reference).abs().amax(1),
                    "linear_relative_l2_error": (score - reference).norm(dim=1) / reference.norm(dim=1).clamp_min(1e-12),
                    "linear_topk_overlap": reference_mask.gather(1, selected).float().mean(1),
                }
            for head in range(heads):
                records.append({
                    "method": name, "head": head, "position": position,
                    "new_context_tokens": position - prompt_length,
                    "indexed_tokens": end, "k": k,
                    "recall": float(recall[head]), "candidate_attention_mass": float(mass[head]),
                    "recall_to_mass_topk": float(mass_recall[head]),
                    "selected_new_fraction": float(new_share[head]),
                    **{metric: float(value[head]) for metric, value in equivalence.items()},
                })
    return drift, records


@torch.inference_mode()
def generate_workloads(model, tokenizer, args):
    def prompt_ids(text):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        )
        ids = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
        return ids.to(args.device)

    def generate(ids, cap):
        result = model.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=cap,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
            eos_token_id=model.generation_config.eos_token_id,
            logits_to_keep=1,
        )
        new = result.shape[1] - ids.shape[1]
        print(f"generated {new} tokens (cap {cap})", flush=True)
        return result, {"generated_tokens": new, "hit_cap": new == cap}

    reasoning = (
        "Write a detailed mathematical tutorial, solving each of the following problems with "
        "derivations and numerical verification. Complete all twelve sections in order. "
        "1. Derive the coupon collector expectation for 8 coupons. "
        "2. Derive its variance. 3. Compute the probability all 8 appear in 20 draws. "
        "4. Derive the birthday collision probability for 23 people. "
        "5. Bound it with the exponential approximation. "
        "6. Solve a biased random walk hitting probability on 0..10 with p=0.6. "
        "7. Compute its expected absorption time. 8. Derive gambler's ruin for p=0.5. "
        "9. Explain conditional expectation using a two-dice example. "
        "10. Prove the law of total variance. 11. Derive a Chernoff bound for 100 coin tosses. "
        "12. Compare this bound with an exact binomial tail. Show intermediate calculations."
    )
    ids = prompt_ids(reasoning)
    initial_length = ids.shape[1]
    ids, generation = generate(ids, args.reasoning_tokens)
    yield "reasoning", ids.cpu(), initial_length, [generation]
    del ids
    gc.collect()
    torch.cuda.empty_cache()

    data = json.loads(Path(args.data_path).read_text())
    if isinstance(data, list):
        data = data[0]
    document_ids = tokenizer.encode(data["input"], add_special_tokens=False)[:args.document_tokens]
    document = tokenizer.decode(document_ids)
    turns = [
        "Read these reference documents and summarize the most important facts. Remember them "
        "for later questions.\n\n" + document,
        "Switch to programming. Write and explain a Python LRU cache using a doubly linked "
        "list and dictionary, including update and eviction cases.",
        "Switch to probability. Derive the expected number and variance of fair die rolls "
        "needed to observe all six faces. Show the calculations.",
        "Return to the original reference documents. List five concrete facts from them "
        "and explain how each fact relates to its document. Do not use the later programming "
        "or probability discussion as your source.",
    ]
    ids = prompt_ids(turns[0])
    initial_length = ids.shape[1]
    generations = []
    for index, turn in enumerate(turns):
        if index:
            # Preserve every previous token. A capped response is explicitly
            # terminated before the next user turn, and the cap is reported.
            terminator = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if ids[0, -1].item() != terminator:
                ids = torch.cat((ids, ids.new_tensor([[terminator]])), dim=1)
            suffix = "<|start_header_id|>user<|end_header_id|>\n\n" + turn + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            appended = tokenizer(suffix, add_special_tokens=False, return_tensors="pt").input_ids.to(args.device)
            ids = torch.cat((ids, appended), dim=1)
        ids, generation = generate(ids, args.turn_tokens)
        generations.append(generation)
    yield "multiturn", ids.cpu(), initial_length, generations


@torch.inference_mode()
def capture_qk(model, ids, prompt_length, args):
    from transformers.models.llama import modeling_llama

    positions = list(range(prompt_length + 128, ids.shape[1], 128))
    captures = {layer: {"keys": [], "queries": []} for layer in args.layers}
    state = {"layer": None, "offset": 0}
    handles = []
    original = modeling_llama.apply_rotary_pos_emb

    def choose_layer(index):
        def pre_hook(_module, _inputs):
            state["layer"] = index
        return pre_hook

    def capture(*call_args, **call_kwargs):
        q, k = original(*call_args, **call_kwargs)
        layer, offset = state["layer"], state["offset"]
        if layer in captures:
            captures[layer]["keys"].append(k[0].cpu())
            local = [position - offset for position in positions if offset <= position < offset + k.shape[2]]
            if local:
                group = q.shape[1] // k.shape[1]
                captures[layer]["queries"].append(q[0, :, local].reshape(k.shape[1], group, len(local), k.shape[-1]).cpu())
        return q, k

    for index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(choose_layer(index)))
    modeling_llama.apply_rotary_pos_emb = capture
    past = None
    try:
        for start in range(0, ids.shape[1], args.replay_chunk):
            state["offset"] = start
            output = model(
                ids[:, start:start + args.replay_chunk].to(args.device),
                past_key_values=past, use_cache=True, logits_to_keep=1,
            )
            past = output.past_key_values
        for layer, tensors in captures.items():
            if not tensors["queries"]:
                raise RuntimeError("No query probes; increase generation length.")
            captures[layer] = {
                "keys": torch.cat(tensors["keys"], dim=1),
                "queries": torch.cat(tensors["queries"], dim=2),
            }
    finally:
        modeling_llama.apply_rotary_pos_emb = original
        for handle in handles:
            handle.remove()
    return captures, positions


def summarize(output_dir, records, drift):
    summaries = {}
    workloads = sorted({row["workload"] for row in records})
    for workload in workloads:
        subset = [row for row in drift if row["workload"] == workload]
        summaries[workload] = {
            "drift": {
                key: statistics.mean(row[key] for row in subset)
                for key in ("mean_shift_over_prompt_rms", "mean_cosine", "norm_clip_high", "norm_clip_low")
            },
            "selectors": {},
        }
        for method in sorted({row["method"] for row in records}):
            subset = [row for row in records if row["workload"] == workload and row["method"] == method]
            summaries[workload]["selectors"][method] = {
                key: statistics.mean(row[key] for row in subset)
                for key in ("recall", "candidate_attention_mass", "recall_to_mass_topk", "selected_new_fraction")
            }
            if method.endswith("_head_sum"):
                summaries[workload]["selectors"][method].update({
                    "linear_max_abs_error": max(row["linear_max_abs_error"] for row in subset),
                    "linear_relative_l2_error": max(row["linear_relative_l2_error"] for row in subset),
                    "linear_topk_overlap": statistics.mean(row["linear_topk_overlap"] for row in subset),
                })
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; summary.json contains the numerical results", flush=True)
        return
    fig, axes = plt.subplots(2, len(workloads), figsize=(11, 7), squeeze=False)
    methods = ["frozen", "forward_ema_0.1", "block_center_corrected", "full_rebuild_oracle"]
    for column, workload in enumerate(workloads):
        for method in methods:
            subset = [row for row in records if row["workload"] == workload and row["method"] == method]
            positions = sorted({row["new_context_tokens"] for row in subset})
            values = [100 * statistics.mean(row["recall"] for row in subset if row["new_context_tokens"] == pos) for pos in positions]
            axes[0, column].plot(positions, values, marker=".", label=method)
        axes[0, column].set(title=workload, ylabel="Exact group top-k recall (%)")
        axes[0, column].legend(fontsize=7)
        subset = [row for row in drift if row["workload"] == workload]
        origin = min(row["start"] for row in subset)
        ends = sorted({row["end"] for row in subset})
        values = [100 * statistics.mean(row["norm_clip_high"] for row in subset if row["end"] == end) for end in ends]
        axes[1, column].plot([end - origin for end in ends], values, color="tab:red", marker=".")
        axes[1, column].set(xlabel="Tokens appended after initial prompt", ylabel="Frozen norm upper clipping (%)")
        for axis in axes[:, column]:
            axis.grid(alpha=0.25)
    fig.suptitle("Llama-3.1-8B-Instruct: two diagnostic trajectories, five sampled layers")
    fig.tight_layout()
    fig.savefig(output_dir / "drift_diagnostic.png", dpi=180)
    fig.savefig(output_dir / "drift_diagnostic.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="results/drift_diagnostic")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-path", default="throughput_eval/test_data/qa1.json")
    parser.add_argument("--reasoning-tokens", type=int, default=2048)
    parser.add_argument("--turn-tokens", type=int, default=384)
    parser.add_argument("--document-tokens", type=int, default=3072)
    parser.add_argument("--replay-chunk", type=int, default=512)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 7, 15, 23, 31])
    parser.add_argument("--budget", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reuse-trajectories", action="store_true", help="Replay tokens already saved in --output, without generating new responses.")
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True,
    ).to(args.device).eval()
    metadata = {
        "args": vars(args), "torch": torch.__version__, "transformers": __version__,
        "gpu": torch.cuda.get_device_name(args.device), "workloads": {},
        "scope": "Full-attention trajectories; offline selector diagnostics; no task-accuracy or runtime-speed claim.",
        "budget_policy": "Fixed diagnostic k=max(16,floor(budget*initial_prompt_tokens)); not the runtime's capped dynamic budget schedule.",
    }
    if args.reuse_trajectories:
        previous = json.loads((output_dir / "metadata.json").read_text())
        if previous["args"]["model"] != args.model:
            raise ValueError("Saved trajectories were generated by a different model.")
        workloads = (
            (name, torch.load(output_dir / f"{name}_tokens.pt", weights_only=True), values["prompt_tokens"], values["generations"])
            for name, values in previous["workloads"].items()
        )
    else:
        workloads = generate_workloads(model, tokenizer, args)
    all_records, all_drift = [], []
    for name, ids, prompt_length, generations in workloads:
        print(f"replay {name}: prompt={prompt_length}, total={ids.shape[1]}", flush=True)
        metadata["workloads"][name] = {"prompt_tokens": prompt_length, "total_tokens": ids.shape[1], "generations": generations}
        (output_dir / f"{name}.txt").write_text(tokenizer.decode(ids[0]))
        torch.save(ids, output_dir / f"{name}_tokens.pt")
        captures, positions = capture_qk(model, ids, prompt_length, args)
        with torch.inference_mode():
            for layer, tensors in captures.items():
                drift, records = analyze_layer(tensors["keys"], tensors["queries"], positions, prompt_length, args)
                all_records.extend(dict(workload=name, layer=layer, **row) for row in records)
                all_drift.extend(dict(workload=name, layer=layer, **row) for row in drift)
        del captures
        gc.collect()
        torch.cuda.empty_cache()
        print(f"analyzed {name}", flush=True)
    metadata["elapsed_seconds"] = time.perf_counter() - started
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    for name, records in (("selectors", all_records), ("drift", all_drift)):
        with (output_dir / f"{name}.jsonl").open("w") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
    summarize(output_dir, all_records, all_drift)
    print(f"results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
