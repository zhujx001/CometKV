import math

import torch
from flash_attn import flash_attn_with_kvcache
from .flash_attn_cache import flash_attn_cache


class exact_topk_cache(flash_attn_cache):
    """
    Oracle exact top-k retrieval baseline ("Exact_TopK").

    Dense full prefill with the KV cache fully resident on each layer's GPU (inherits the
    Full_Flash_Attn cache, so the multi-GPU pipeline layer mapping works unchanged). At decode,
    every step attends to exactly `fixed_topk` tokens selected by exact q.k scores over ALL
    cached tokens — pure retrieval, nothing is force-included unless force_sink/force_recent
    are explicitly set (>0 rows are pinned into the selection, counted inside the budget).

      - GQA: candidates are scored once per kv head with the group-MEAN query; the actual
        attention then runs every real query head over the group's shared selection.
      - fixed_topk = max(min_retrieval_topk, int(retrieval_budget * prompt_len)) is frozen from
        the prompt length at the first decode step and never grows as decode proceeds.

    Hybrid sampled-tail estimator (sample_frac > 0): the same total budget k is split into an
    exact head (top-(k-m) by score) plus m tokens sampled WITH replacement from the remaining
    tail with proposal q_j = softmax(score_j / (sqrt(d) * tau)); each sampled slot's attention
    logit gets the importance correction -log(m * q_j), making the union softmax a
    self-normalized importance-sampling estimate of FULL attention (head exact, tail unbiased
    in expectation) instead of the truncate-and-renormalize top-k estimate. Motivation: on
    aggregation tasks the top-k ESTIMATOR bias — not selection quality — is the accuracy wall.
    Same gather volume as the pure top-k path.
    """

    def __init__(
        self,
        valid_start,
        layer_num: int,
        batch_size: int,
        max_length: int,
        num_key_value_heads: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        layer_mapping: dict,
        prefill_bsz: int,
        num_gpus: int,
        model_size: int,
        retrieval_budget: float = 0.1,
        force_sink: int = 0,
        force_recent: int = 0,
        min_retrieval_topk: int = 1,
        sample_frac: float = 0.0,
        sample_tau: float = 1.0,
        sample_seed: int = 1234,
    ) -> None:
        super().__init__(valid_start, layer_num, batch_size, max_length, num_key_value_heads,
                         num_heads, head_dim, dtype, layer_mapping, prefill_bsz, num_gpus, model_size)
        if not (self.valid_start_list == 0).all():
            raise ValueError("Exact_TopK requires same-length inputs with valid_start == 0.")
        self.retrieval_budget = float(retrieval_budget)
        self.force_sink = max(int(force_sink), 0)
        self.force_recent = max(int(force_recent), 0)
        self.min_retrieval_topk = max(int(min_retrieval_topk), 1)
        self.sample_frac = min(max(float(sample_frac), 0.0), 1.0)
        self.sample_tau = float(sample_tau)
        assert self.sample_tau > 0.0, "sample_tau must be positive"
        self.sample_seed = int(sample_seed)
        self._sample_generators = {}   # device str -> seeded torch.Generator
        self.fixed_topk = None   # frozen from the prompt length at the first decode step
        self.use_cuda_graph = False   # eager decode only
        self.attn_func = self.sparse_attention

    def _sample_generator(self, device):
        key = str(device)
        gen = self._sample_generators.get(key)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.sample_seed)
            self._sample_generators[key] = gen
        return gen

    def _ensure_plan(self):
        if self.fixed_topk is None:
            # First decode step, before this step's context increment (which happens at the
            # last layer's decode_update): self.context is exactly the prompt length.
            prompt_len = int(self.context)
            k = max(self.min_retrieval_topk, int(self.retrieval_budget * prompt_len))
            self.fixed_topk = min(k, prompt_len)

    def sparse_attention(self, query_states, layer_idx):
        """
        Args:
            query_states: (bsz, 1, num_heads, head_dim), on this layer's device.
        Returns:
            attn_out: (bsz, 1, num_heads, head_dim)
        """
        self._ensure_plan()
        # Tokens currently in the cache for this step. decode_update_kv_cache increments
        # self.context at the last layer only — mirror full_decode_attn's off-by-one handling.
        total = int(self.context) if layer_idx == self.layer_num - 1 else int(self.context) + 1
        k = min(self.fixed_topk, total)
        bsz, _, n_heads, dim = query_states.shape
        group = n_heads // self.kv_head

        keys = self.key_cache[layer_idx][:, :total]      # (bsz, total, kv_head, dim)
        values = self.value_cache[layer_idx][:, :total]

        # GQA retrieval scoring: exact q.k with the group-mean query per kv head.
        q_mean = (
            query_states.view(bsz, self.kv_head, group, dim)
            .to(torch.float32).mean(dim=2)
            .to(self.dtype)
            .view(bsz, self.kv_head, dim, 1)
        )
        # (bsz, kv_head, total, dim) @ (bsz, kv_head, dim, 1); the permute is stride-only.
        scores = torch.matmul(keys.permute(0, 2, 1, 3), q_mean).view(bsz, self.kv_head, total).float()

        if self.force_sink > 0:
            scores[:, :, : min(self.force_sink, total)] = float("inf")
        if self.force_recent > 0:
            scores[:, :, max(total - self.force_recent, 0):] = float("inf")

        m = int(round(self.sample_frac * k))
        m = min(m, k - 1)   # keep at least one exact head slot
        if m > 0 and total > k:
            return self._hybrid_sampled_attention(query_states, keys, values, scores, k, m)

        # sorted=False: selection is consumed as a set (see cometkv_cache selector notes).
        sel = torch.topk(scores, k, dim=-1, sorted=False).indices     # (bsz, kv_head, k)
        idx = sel.permute(0, 2, 1).unsqueeze(-1).expand(bsz, k, self.kv_head, dim)
        k_sel = torch.gather(keys, 1, idx)
        v_sel = torch.gather(values, 1, idx)

        # Per-head attention over the group's shared selection (no mask needed: every
        # gathered token is attendable by construction, q_len == 1).
        attn_out = flash_attn_with_kvcache(q=query_states, k_cache=k_sel, v_cache=v_sel)
        return attn_out

    def _hybrid_sampled_attention(self, query_states, keys, values, scores, k, m):
        """
        Budget-split hybrid estimator: exact top-(k-m) head + m tail tokens sampled with
        replacement from proposal q_j = softmax(score_j / (sqrt(d) * tau)) over the non-head
        candidates; each sampled slot's logit gets -log(m * q_j) so the union softmax is a
        self-normalized importance-sampling estimate of full attention. Same k gathered
        tokens as the pure top-k path. Attention is computed in fp32 torch (eager prototype;
        the sampled-slot logit bias rules out flash_attn here).
        """
        bsz, _, n_heads, dim = query_states.shape
        group = n_heads // self.kv_head
        total = scores.shape[-1]
        kh = k - m
        inv_sqrt_d = 1.0 / math.sqrt(dim)

        sel_head = torch.topk(scores, kh, dim=-1, sorted=False).indices   # (bsz, kv_head, kh)

        # Tail proposal at attention scale, head slots masked out of the support.
        prop = scores * (inv_sqrt_d / self.sample_tau)
        prop = prop.scatter(-1, sel_head, float("-inf"))
        q_prob = torch.softmax(prop, dim=-1)                              # (bsz, kv_head, total)
        draws = torch.multinomial(
            q_prob.view(bsz * self.kv_head, total), m, replacement=True,
            generator=self._sample_generator(q_prob.device),
        ).view(bsz, self.kv_head, m)
        # Drawn entries have q_prob > 0 by construction; clamp only guards fp rounding.
        log_q = torch.log(torch.gather(q_prob, -1, draws).clamp_min(1e-30))
        is_corr = -(math.log(m) + log_q)                                  # (bsz, kv_head, m)

        sel = torch.cat((sel_head, draws), dim=-1)                        # (bsz, kv_head, k)
        idx = sel.permute(0, 2, 1).unsqueeze(-1).expand(bsz, k, self.kv_head, dim)
        k_sel = torch.gather(keys, 1, idx)                                # (bsz, k, kv_head, dim)
        v_sel = torch.gather(values, 1, idx)

        q = query_states.view(bsz, self.kv_head, group, dim).float()
        ks = k_sel.permute(0, 2, 1, 3).float()                            # (bsz, kv_head, k, dim)
        vs = v_sel.permute(0, 2, 1, 3).float()
        logits = torch.einsum("bhgd,bhkd->bhgk", q, ks) * inv_sqrt_d
        logits[..., kh:] += is_corr.unsqueeze(2)
        attn = torch.softmax(logits, dim=-1)
        out = torch.einsum("bhgk,bhkd->bhgd", attn, vs)
        return out.reshape(bsz, n_heads, dim).to(query_states.dtype).view(bsz, 1, n_heads, dim)
