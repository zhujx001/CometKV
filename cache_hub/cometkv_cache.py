import math
import os
import sys
import time
from contextlib import nullcontext

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KERNEL_LIB = os.path.join(PROJECT_ROOT, "library", "cometkv")
if KERNEL_LIB not in sys.path:
    sys.path.insert(0, KERNEL_LIB)

from cometkv import (
    append_lockstep_local_kv_cache,
    append_lockstep_local_kv_cache_and_advance,
    append_lockstep_local_kv_cache_dev,
    asym_signature_score_into,
    grouped_signature_score_into,
    concat_static_recent_gpu_gather,
    concat_static_recent_gpu_gather_int8,
    concat_static_recent_lookup_gather_uva_kv_update_cache,
    concat_static_recent_lookup_gather_uva_kv_update_cache_int8,
    copy_interleaved_int8_kv_to_cpu,
    copy_interleaved_kv_to_cpu,
    refresh_static_prompt_recent_state,
    sampled_tail_attention_merge,
    uva_gather_kv_rows,
    uva_gather_kv_rows_window,
    exact_topk_indices_into,
)
try:  # absent from extension builds older than the persistent tail gather
    from cometkv import uva_gather_kv_rows_window_persist
except ImportError:
    uva_gather_kv_rows_window_persist = None
try:
    from flash_attn import flash_attn_with_kvcache
except ImportError as flash_attn_import_error:
    def flash_attn_with_kvcache(*_args, **_kwargs):
        raise ImportError(
            "flash_attn_with_kvcache is required for CometKV concat FlashAttention path. "
            "Install flash-attn before running that path."
        ) from flash_attn_import_error
from .cache import KV_Cache

# Tail-traffic diagnostic (COMETKV_TAIL_DIAG_OUT, eager only): per layer, how many of the m sampled
# tail fetches are duplicates within a row, land in the layer's head (masked by the merge kernel),
# were already drawn last step, or sit in the head token cache. Saved as JSON at exit.
_TAIL_DIAG = {}


def _tail_diag(path, layers, device):
    if path not in _TAIL_DIAG:
        if not _TAIL_DIAG:
            import atexit
            atexit.register(_save_tail_diag)
        _TAIL_DIAG[path] = {"acc": torch.zeros(6, layers, dtype=torch.float64, device=device), "prev": {}}
    return _TAIL_DIAG[path]


def _save_tail_diag():
    import json
    names = ["draws", "unique", "in_head", "unique_not_head", "unique_seen_prev_step", "unique_in_token_cache"]
    for path, d in _TAIL_DIAG.items():
        acc = d["acc"].cpu()
        with open(path, "w") as f:
            json.dump({"per_layer": {n: acc[i].tolist() for i, n in enumerate(names)},
                       "total": {n: float(acc[i].sum()) for i, n in enumerate(names)}}, f, indent=1)


DEFAULT_MIN_RETRIEVAL_TOPK = 0


class cometkv_cache(KV_Cache):
    """Single-GPU CometKV cache runtime skeleton."""

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
        max_new_length: int,
        static_pattern_start: int,
        static_pattern_end: int,
        retrieval_budget: float,
        sig_bits: int,
        sig_topk: int,
        sig_chunk_size: int,
        sig_seed: int,
        sig_mode: str,
        sig_token_cache_size: int,
        prefill_bsz: int,
        num_gpus: int,
        model_size: int,
        sig_min_retrieval_topk: int = DEFAULT_MIN_RETRIEVAL_TOPK,
        exclude_preserved_from_budget: bool = True,
        core: int = 0,
        cpu_kv_quant: str = "none",
        kv_store_device: str = "cpu",
        sig_selector: str = "asym_n8",
        mean_update_alpha: float = 0.0,
        norm_margin: float = 0.0,
        full_recompute_interval: int = 0,
        sample_frac: float = 0.25,
        sample_tau: float = 1.0,
        sample_seed: int = 1234,
        stats_mode: str = "block",
        query_aggregation: str = "q_sum",
        sample_size: int | None = 256,
        sig_max_retrieval_topk: int = 0,
        head_extra_tokens: int = 0,
        tail_in_budget: bool = True,
        sample_min_context: int = 32768,
        sample_mode: str = "fraction",
        token_cache_mult: float = 2.0,
        token_cache_round: int = 8,
        sample_max_m: int = 256,
        sample_min_m: int = 64,
    ) -> None:
        super().__init__(
            layer_num,
            batch_size,
            max_length,
            num_key_value_heads,
            num_heads,
            head_dim,
            dtype,
            layer_mapping,
            prefill_bsz,
            num_gpus,
            model_size,
        )
        self.device_list = sorted(set(self.layer_mapping.values()), key=self._device_sort_key)
        self.primary_device = self.layer_mapping[str(0)]
        self.valid_start_list = valid_start
        self.max_new_length = max_new_length

        self.static_pattern_start = static_pattern_start
        self.static_pattern_end = static_pattern_end
        self.static_pattern_total = static_pattern_start + static_pattern_end
        self.retrieval_budget = retrieval_budget
        self.head_extra_tokens = int(head_extra_tokens)
        if self.head_extra_tokens < 0:
            raise ValueError("head_extra_tokens must be nonnegative")
        self.sig_bits = sig_bits
        self.sig_bytes = sig_bits // 8
        self.sig_topk = sig_topk
        self.sig_chunk_size = sig_chunk_size
        self.sig_seed = int(sig_seed)
        self.sig_mode = sig_mode
        self.base_sig_token_cache_size = int(sig_token_cache_size)
        # Token-cache capacity per (row, layer), set in _refresh_token_cache_sizes_for_temporal_reuse:
        # COMETKV_TOKEN_CACHE_SLOTS, when set, is the exact capacity; otherwise MULT x K_t rounded up to
        # ROUND slots (MULT <= 1: legacy 1024-rounded top-k sizing with the config floor).
        self.token_cache_mult = float(os.environ.get("COMETKV_TOKEN_CACHE_MULT", token_cache_mult))
        self.token_cache_round = int(os.environ.get("COMETKV_TOKEN_CACHE_ROUND", token_cache_round))
        slots = os.environ.get("COMETKV_TOKEN_CACHE_SLOTS")
        self.token_cache_slots = int(slots) if slots else None
        if self.token_cache_slots is not None:
            # Also the floor of the init-time sizing (concat-buffer bound), as before.
            self.base_sig_token_cache_size = self.token_cache_slots
        self.sig_min_retrieval_topk = max(int(sig_min_retrieval_topk), 0)
        self.sig_max_retrieval_topk = int(os.environ.get("COMETKV_MAX_RETRIEVAL_TOPK", sig_max_retrieval_topk))
        if self.sig_max_retrieval_topk < 0:
            raise ValueError("sig_max_retrieval_topk must be nonnegative (0 = no explicit cap)")
        self.exclude_preserved_from_budget = bool(exclude_preserved_from_budget)
        # Retrieval selector: 120 sign bits + 1 byte log-quantized ||k|| (16B/token), scored
        # asymmetrically with a float query projection and norm weighting.
        self.scorer_mode = os.environ.get("COMETKV_SCORER", "asym")
        if self.scorer_mode not in ("asym", "sym"):
            raise ValueError("COMETKV_SCORER must be asym or sym")
        self.disable_token_cache = os.environ.get("COMETKV_DISABLE_TOKEN_CACHE", "0") == "1"
        self.selector_mode = str(os.environ.get("COMETKV_SELECTOR", sig_selector or "asym_n8")).lower()
        assert self.selector_mode in ("asym_n8",), (
            f"Unsupported sig_selector: {self.selector_mode}"
        )
        # asym_n8 signature layout: bytes 0..14 = 120 sign bits, byte 15 = log-norm code.
        self.asym_sig_bits = self.sig_bits - 8
        # Query-centered scoring (COMETKV_QCENTER = auto | off | fp | 8bit). Exact split
        #   q.r = (q - mu_q).r + mu_q.r,  r = k - mu_block,
        # with one query mean mu_q per (layer, row) frozen from the prompt (GQA q_sum, post-RoPE).
        # The sign estimator only sees (q - mu_q), whose norm is smaller; c = mu_q.r is computed exactly
        # when a key is encoded (prefill or block seal) and stored per token (fp32, or an 8-bit code with a
        # per-(row, encode call) range). COMETKV_SIG_BITS_USED < 120 frees signature bytes: with 112 bits
        # and 8bit the code lives in byte 14, keeping 16 B/token (the kernel reads only bits_used/8 bytes).
        # auto (default) = 8bit with 112 sign bits whenever its preconditions hold (q_sum aggregation,
        # asym scorer, key centering), otherwise off. COMETKV_QCENTER=off restores the 120-bit scorer.
        self.qcenter = os.environ.get("COMETKV_QCENTER", "auto").lower()
        if self.qcenter not in ("auto", "off", "fp", "8bit"):
            raise ValueError("COMETKV_QCENTER must be auto, off, fp or 8bit")
        if self.qcenter == "auto":
            qcenter_ok = (os.environ.get("COMETKV_QUERY_AGG", query_aggregation).lower() == "q_sum"
                          and self.scorer_mode == "asym"
                          and os.environ.get("COMETKV_NO_KEY_CENTER", "0") != "1")
            self.qcenter = "8bit" if qcenter_ok else "off"
        default_bits = self.asym_sig_bits - 8 if self.qcenter == "8bit" else self.asym_sig_bits
        bits_used = int(os.environ.get("COMETKV_SIG_BITS_USED", default_bits))
        if bits_used % 8 or not 0 < bits_used <= self.asym_sig_bits:
            raise ValueError("COMETKV_SIG_BITS_USED must be a positive multiple of 8 <= 120")
        self.asym_sig_bits = bits_used
        self._last_qbias = None
        self.core = core
        # Optional int8 quantization of the CPU-resident retrieval KV store to halve UVA/PCIe gather
        # traffic. K is dequantized per-channel, V per-token, back to self.dtype on gather. The on-GPU
        # static/recent regions, signatures, and token cache stay full precision. Default off so the
        # bf16 path is unchanged (clean A/B for accuracy).
        self.cpu_kv_quant = str(cpu_kv_quant or "none").lower()
        assert self.cpu_kv_quant in ("none", "int8"), f"Unsupported cpu_kv_quant: {self.cpu_kv_quant}"
        self.quantize_cpu_kv = self.cpu_kv_quant == "int8"
        if self.disable_token_cache and (self.quantize_cpu_kv or self.dtype != torch.bfloat16):
            raise ValueError("COMETKV_DISABLE_TOKEN_CACHE requires a BF16 KV store")
        # CometKV_GPU backend: when "gpu", the retrieval store (otherwise pinned host memory read over
        # UVA) lives entirely in device memory and the gather uses the dedicated token-cache-free GPU
        # kernels. Numerically identical to the "cpu" path; only the memory location and speed change.
        self.kv_store_device = str(kv_store_device or "cpu").lower()
        assert self.kv_store_device in ("cpu", "gpu"), f"Unsupported kv_store_device: {self.kv_store_device}"
        self.kv_on_gpu = self.kv_store_device == "gpu"
        self.stats_mode = os.environ.get("COMETKV_STATS_MODE", stats_mode).lower()
        self.query_aggregation = os.environ.get("COMETKV_QUERY_AGG", query_aggregation).lower()
        if self.stats_mode not in ("block", "mu_frozen", "frozen"):
            raise ValueError("stats_mode must be 'block', 'mu_frozen' or 'frozen'")
        if self.query_aggregation not in ("mean_prob", "q_sum"):
            raise ValueError("query_aggregation must be 'mean_prob' or 'q_sum'")
        # mu_frozen keeps block norm updates and the same scoring implementation.
        self.block_stats = self.stats_mode in ("block", "mu_frozen")
        # Auto uses FP32 lookup scoring for block/q_sum and exact radix selection
        # for long, sparse ranges. Explicit switches support controlled comparisons.
        self.signature_score_impl = os.environ.get("COMETKV_SCORE_IMPL", "auto").lower()
        self.exact_topk_impl = os.environ.get("COMETKV_TOPK_IMPL", "auto").lower()
        if self.signature_score_impl not in ("auto", "scalar", "lookup"):
            raise ValueError("COMETKV_SCORE_IMPL must be auto, scalar or lookup")
        if self.exact_topk_impl not in ("auto", "torch", "radix"):
            raise ValueError("COMETKV_TOPK_IMPL must be auto, torch or radix")
        self._exact_topk_workspace = {}
        # Forward-only EMA omitted a first-order, query-dependent offset between
        # signatures encoded with different centers. Use sealed block statistics.
        self.mean_update_alpha = float(mean_update_alpha)
        if self.mean_update_alpha != 0.0:
            raise ValueError("Forward-only mean_update_alpha is unsupported; use stats_mode='block'.")
        # Widen each newly sealed block's log-norm range by this fraction on each side.
        # Frozen mode applies this only at prefill; block mode also updates new blocks.
        self.norm_margin = float(norm_margin)
        # Full recompute interval: every N decode tokens, recompute μ and norm range from ALL keys
        # in the retrieval index and rebuild ALL signatures. 0 = disabled. Runs on the eviction path.
        self.full_recompute_interval = int(full_recompute_interval)
        if self.full_recompute_interval < 0 or self.norm_margin < 0:
            raise ValueError("full_recompute_interval and norm_margin must be nonnegative")
        if self.block_stats and self.full_recompute_interval:
            raise ValueError("Full recompute requires stats_mode='frozen'; block statistics are immutable.")
        self._recompute_decode_counter = [0] * self.layer_num
        self.last_full_recompute_ms = 0.0
        # Tail draw count m, decided per plan in _plan_tail_size (COMETKV_SAMPLE_MODE / sample_mode):
        #   fraction (default): m = min(floor(SAMPLE_FRAC * K_t), SAMPLE_MAX_M); no tail if m < SAMPLE_MIN_M
        #   size:               m = min(SAMPLE_SIZE, SAMPLE_MAX_M)
        #   off:                no tail
        # COMETKV_SAMPLE_SIZE used to take precedence over everything; setting it without an explicit
        # COMETKV_SAMPLE_MODE is rejected rather than silently ignored under the fraction default.
        mode_env = os.environ.get("COMETKV_SAMPLE_MODE")
        if mode_env is None and "COMETKV_SAMPLE_SIZE" in os.environ:
            raise ValueError("COMETKV_SAMPLE_SIZE is set without COMETKV_SAMPLE_MODE; set COMETKV_SAMPLE_MODE=size "
                             "for a fixed draw count (the default mode is fraction)")
        self.sample_mode = str(mode_env if mode_env is not None else sample_mode).lower()
        if self.sample_mode not in ("fraction", "size", "off"):
            raise ValueError(f"COMETKV_SAMPLE_MODE / sample_mode must be fraction, size or off, got {self.sample_mode!r}")
        self.sample_size = int(os.environ.get("COMETKV_SAMPLE_SIZE", 256 if sample_size is None else sample_size))
        if self.sample_size < 0:
            raise ValueError(f"COMETKV_SAMPLE_SIZE / sample_size must be nonnegative, got {self.sample_size}")
        self.sample_frac = float(os.environ.get("COMETKV_SAMPLE_FRAC", sample_frac))
        if not 0.0 <= self.sample_frac < 1.0:
            raise ValueError(f"COMETKV_SAMPLE_FRAC / sample_frac must be in [0, 1), got {self.sample_frac}")
        self.sample_tau = float(os.environ.get("COMETKV_SAMPLE_TAU", sample_tau))
        self.sample_seed = int(os.environ.get("COMETKV_SAMPLE_SEED", sample_seed))
        assert self.sample_tau > 0.0, "sample_tau must be positive"
        # Tail accounting (COMETKV_TAIL_IN_BUDGET, default 1 = paper Sec. V-A): the m draws count
        # against rho, so the head holds K_t - m records and a step reads K_t = floor(rho * L_t)
        # retrieved records in total. If K_t <= m the head would be empty, so that plan runs
        # head-only with K_t. 0 restores the earlier protocol: head K_t plus m additional draws.
        tail_flag = os.environ.get("COMETKV_TAIL_IN_BUDGET")
        if tail_flag is not None and tail_flag not in ("0", "1"):
            raise ValueError(f"COMETKV_TAIL_IN_BUDGET must be 0 or 1, got {tail_flag!r}")
        self.tail_in_budget = bool(tail_in_budget) if tail_flag is None else tail_flag == "1"
        # Sampling threshold (COMETKV_SAMPLE_MIN_CONTEXT, default 32768 = paper "from a 32K context
        # onward"): plans whose visible context L_t is shorter run head-only. 0 = always sample.
        self.sample_min_context = int(os.environ.get("COMETKV_SAMPLE_MIN_CONTEXT", sample_min_context))
        if self.sample_min_context < 0:
            raise ValueError(
                f"COMETKV_SAMPLE_MIN_CONTEXT / sample_min_context must be nonnegative, got {self.sample_min_context}")
        if self.sample_mode != "off":
            # Both score variants support the tail proposal: calibrated log probabilities
            # for mean_prob, auto-standardized projection scores for legacy q_sum.
            assert self.selector_mode in ("asym_n8",), (
                f"sampled-tail estimator unsupported for selector {self.selector_mode}"
            )
        # Proposal auto-standardization: z-score the candidate scores and rescale to a target
        # logit std (sigma/tau) instead of the analytic 1/(G*sqrt(d)) beta. Env overrides for A/B.
        _auto_default = "0" if self.query_aggregation == "mean_prob" else "1"
        self.sample_autoscale = os.environ.get("COMETKV_SAMPLE_AUTOSCALE", _auto_default) == "1"
        self.sample_sigma = float(os.environ.get("COMETKV_SAMPLE_SIGMA", "2.0"))
        # A uniform component prevents zero support from fp32 softmax underflow and
        # covers owner-head positions needed by subsequent layers with different heads.
        self.sample_uniform_mix = float(os.environ.get("COMETKV_SAMPLE_UNIFORM_MIX", "0.1"))
        if not 0.0 < self.sample_uniform_mix <= 1.0:
            raise ValueError("COMETKV_SAMPLE_UNIFORM_MIX must be in (0, 1]")
        # Truncated-IS clip (environment override): cap each corrected tail logit at
        # per-(row,head) valid-slot mean + clip nats (inside the fused kernel). This
        # controls large importance weights at the cost of bias; 0 disables clipping.
        self.sample_clip = float(os.environ.get("COMETKV_SAMPLE_CLIP", "4.0"))
        # Resample stride (env-only): draws/corr are recomputed from the CURRENT layer's scores
        # every J-th layer and reused in between. All layers share a full-support proposal;
        # the merge kernel removes each current layer's head from its tail contributions.
        self.sample_stride = max(1, int(os.environ.get("COMETKV_SAMPLE_STRIDE", "8")))
        # Tail-only proposal (COMETKV_SAMPLE_EXCLUDE_HEAD, default 1): the softmax part of the proposal
        # excludes the owner layer's head (already exact), so draws are not spent on head tokens the
        # merge kernel masks. The uniform mix keeps full support for the other layers of the stride
        # window, whose heads differ, so the estimator stays unbiased for every layer. 0 restores the
        # full-support proposal (40-45% of its draws land in the owner head and are discarded).
        self.sample_exclude_head = os.environ.get("COMETKV_SAMPLE_EXCLUDE_HEAD", "1") == "1"
        # Per-layer pipelined window fetch (COMETKV_TAIL_PIPELINE, default 1): the owner layer waits
        # only for its own slice; later layers' slices stream under compute. 0 = one fetch per window.
        self.tail_pipeline = os.environ.get("COMETKV_TAIL_PIPELINE", "1") == "1"
        # COMETKV_TAIL_GATHER_BLOCKS>0: window tail gather uses the small-grid persistent kernel
        # (leaves most SMs to the main stream instead of flooding them with PCIe-stalled warps).
        # Default 8 (bit-identical outputs); 0 = the original one-block-per-8-tokens launch.
        # An extension build without the kernel falls back to 0 unless the env asks for it.
        _blocks_default = "8" if uva_gather_kv_rows_window_persist is not None else "0"
        self.tail_gather_blocks = int(os.environ.get("COMETKV_TAIL_GATHER_BLOCKS", _blocks_default))
        # COMETKV_TAIL_DEDUP=1 (needs the persistent gather): each distinct sampled token crosses PCIe
        # once per row; duplicate draws are filled from their leader slot on the GPU.
        self.tail_dedup = os.environ.get("COMETKV_TAIL_DEDUP", "0") == "1"
        if self.tail_dedup and self.tail_gather_blocks <= 0:
            raise RuntimeError("COMETKV_TAIL_DEDUP requires COMETKV_TAIL_GATHER_BLOCKS > 0")
        if self.tail_gather_blocks > 0 and uva_gather_kv_rows_window_persist is None:
            raise RuntimeError("COMETKV_TAIL_GATHER_BLOCKS > 0 needs a cometkv build with "
                               "uva_gather_kv_rows_window_persist (rebuild library/cometkv)")
        # Hard cap on raw draws. The BF16 CPU path bypasses the token cache, so each
        # additional slot reads KV across all layers regardless of head cache hit rate.
        self.sample_max_m = max(1, int(os.environ.get("COMETKV_SAMPLE_MAX_M", sample_max_m)))
        # Fraction mode only: a derived m below this many draws disables the tail (too few draws for
        # a stable estimate). Size mode uses SAMPLE_SIZE as given, still capped by SAMPLE_MAX_M.
        self.sample_min_m = max(0, int(os.environ.get("COMETKV_SAMPLE_MIN_M", sample_min_m)))
        self._sample_chunk = 512   # two-level inverse-CDF sampling chunk width
        self.active_sample_len_host = 0
        self._sample_state = {}     # device str -> exact-size sampling buffers (rebuilt on m change)
        self._sample_gen = {}       # device str -> seeded torch.Generator (noise refresh)
        self._tail_index_update_step = False
        # When True, sparse_attention produces a fixed-shape concat ([static|sparse|recent_capacity] of
        # width cg_concat_width) and flashes with the device cache_seqlens buffer, so the decode step is
        # CUDA-graph capturable. Set by the model before a captured/replayed decode step. Default off.
        self.use_cuda_graph = False

        self.group_size = self.num_heads // self.kv_head
        self.batch_groups = self.batch_size * self.kv_head
        self.RSQRT_DIM = 1.0 / math.sqrt(self.head_dim)
        self.prompt_capacity = self.max_length - self.max_new_length
        self.retrieval_capacity = self.max_length
        self.lockstep_prompt_length_host = 0
        self.lockstep_decode_step_host = 0
        self.lockstep_decode_block_size = int(os.environ.get("COMETKV_INDEX_UPDATE_INTERVAL", "128"))
        if self.lockstep_decode_block_size < 1:
            raise ValueError("COMETKV_INDEX_UPDATE_INTERVAL must be positive")
        self.lockstep_recent_overlap = max(int(self.static_pattern_end), 0)
        self.lockstep_local_capacity = self.lockstep_decode_block_size + self.lockstep_recent_overlap
        self.lockstep_update_interval = self.lockstep_decode_block_size
        self.lockstep_evicted_retrieval_end_host = 0
        self.lockstep_pending_update_tokens = 0
        self.sig_token_cache_layer_sizes = self._recommended_initial_token_cache_layer_sizes()
        self.sig_token_cache_size = max(self.sig_token_cache_layer_sizes)
        # Concat-buffer sparse-width bound (~1x max_topk, the legacy token-cache size). Captured
        # here because sig_token_cache_size itself is later resized to mult x K_t for temporal reuse
        # (_refresh_token_cache_sizes_for_temporal_reuse) and must not inflate concat storage.
        self.concat_sparse_capacity_bound = self.sig_token_cache_size
        self.sig_block_capacity = (
            2 + math.ceil((max(self.max_new_length, 0) + self.lockstep_recent_overlap)
                          / self.lockstep_decode_block_size)
            if self.block_stats else 1
        )
        self.sig_active_blocks = [1] * self.layer_num
        # E[||x|| (Pq).sign(Px)] = c * q.x for orthogonal unit rows P.
        self.signature_score_scale = self.asym_sig_bits * math.exp(
            math.lgamma(self.head_dim / 2.0) - 0.5 * math.log(math.pi)
            - math.lgamma((self.head_dim + 1.0) / 2.0)
        )

        self.batch_indices_dict = {}
        for device_idx in self.device_list:
            self.batch_indices_dict[device_idx] = torch.arange(self.batch_size, dtype=torch.int32, device=device_idx)
        self.batch_indices = self.batch_indices_dict[self.primary_device]

        self.prompt_lengths = torch.zeros((self.batch_size,), dtype=torch.int32, device=self.primary_device)
        self.visible_lengths = torch.zeros((self.batch_size,), dtype=torch.int32, device=self.primary_device)
        self.layer_visible_lengths = torch.zeros((self.layer_num, self.batch_size), dtype=torch.int32, device=self.primary_device)

        rounded_length = self._round_up(self.max_length, 8)
        # Flatten batch/head rows so prefill D2H writes land in contiguous pinned memory.
        cpu_kv_dtype = torch.int8 if self.quantize_cpu_kv else self.dtype
        self.cpu_kv_cache = [
            torch.empty(
                (self.batch_groups, self.retrieval_capacity, 2, self.head_dim),
                dtype=cpu_kv_dtype,
                device=self.layer_mapping[str(ldx)] if self.kv_on_gpu else "cpu",
                pin_memory=False if self.kv_on_gpu else True,
            ).contiguous()
            for ldx in range(self.layer_num)
        ]
        self.cpu_key_cache = [cache[:, :, 0, :] for cache in self.cpu_kv_cache]
        self.cpu_value_cache = [cache[:, :, 1, :] for cache in self.cpu_kv_cache]
        if self.quantize_cpu_kv:
            # Per-channel K scale (frozen at prefill, lives on GPU; read every gather) and per-token V
            # scale (pinned, read over UVA alongside the int8 codes on the miss path).
            self.cpu_kv_k_scale = [
                torch.ones(
                    (self.batch_groups, self.head_dim),
                    dtype=torch.float32,
                    device=self.layer_mapping[str(ldx)],
                ).contiguous()
                for ldx in range(self.layer_num)
            ]
            self.cpu_kv_v_scale = [
                torch.ones(
                    (self.batch_groups, self.retrieval_capacity),
                    dtype=torch.float32,
                    device=self.layer_mapping[str(ldx)] if self.kv_on_gpu else "cpu",
                    pin_memory=False if self.kv_on_gpu else True,
                ).contiguous()
                for ldx in range(self.layer_num)
            ]
        else:
            self.cpu_kv_k_scale = None
            self.cpu_kv_v_scale = None

        self.signature_index = [
            torch.zeros(
                (self.batch_size, self.kv_head, rounded_length, self.sig_bytes),
                dtype=torch.uint8,
                device=self.layer_mapping[str(ldx)],
            ).contiguous()
            for ldx in range(self.layer_num)
        ]
        self._asym_scores_buffers = {}
        self._grouped_score_buffers = {}
        # Block 0 is the prompt. Centers are stored relative to its center so
        # the prompt needs no compensation and logits have a common origin.
        self.sig_block_centers = [
            torch.zeros((self.batch_groups, self.sig_block_capacity, self.head_dim),
                        dtype=torch.float32, device=self.layer_mapping[str(ldx)])
            for ldx in range(self.layer_num)
        ]
        self.sig_block_norm_lo = [
            torch.zeros((self.batch_groups, self.sig_block_capacity),
                        dtype=torch.float32, device=self.layer_mapping[str(ldx)])
            for ldx in range(self.layer_num)
        ]
        self.sig_block_norm_step = [torch.ones_like(lo) for lo in self.sig_block_norm_lo]
        if self.selector_mode == "asym_n8":
            self.sig_norm_lo = [
                torch.zeros((self.batch_groups,), dtype=torch.float32, device=self.layer_mapping[str(ldx)])
                for ldx in range(self.layer_num)
            ]
            self.sig_norm_step = [
                torch.zeros((self.batch_groups,), dtype=torch.float32, device=self.layer_mapping[str(ldx)])
                for ldx in range(self.layer_num)
            ]
        else:
            self.sig_norm_lo = None
            self.sig_norm_step = None
        # Key centering: signatures/norms are built from k - mu with the
        # per-(layer,row) mean mu frozen from the prompt. Ranking-equivalent to scoring raw keys
        # (q . mu is constant across candidates) but decorrelates the sign bits of near-identical
        # keys, which otherwise collapse onto the shared mean direction. Query path unchanged.
        # Escape hatch for A/B: COMETKV_NO_KEY_CENTER=1.
        self.center_keys = (
            self.selector_mode == "asym_n8"
            and os.environ.get("COMETKV_NO_KEY_CENTER", "0") != "1"
        )
        # Token-cache associativity + replacement. ways==2 keeps the legacy 2-way always-insert path
        # (no stamps passed to the kernel -> bit-identical). ways>2 enables W-way set-associative LRU:
        # per-slot last-touch stamps + a monotonic decode-step counter let the gather kernel evict the
        # least-recently-used way, raising hit rate at a given capacity (equivalently, the same hit
        # rate at lower GPU residency). Accuracy-neutral (the cache is exact). Env: COMETKV_TOKEN_CACHE_WAYS.
        # Default 8 (paper operating point); set 2 to restore the legacy stamp-free path.
        self.token_cache_ways = max(2, int(os.environ.get("COMETKV_TOKEN_CACHE_WAYS", "8")))
        self.token_cache_lru = self.token_cache_ways > 2
        if self.token_cache_round < 1 or self.token_cache_round % self.token_cache_ways:
            raise ValueError(f"COMETKV_TOKEN_CACHE_ROUND / token_cache_round ({self.token_cache_round}) must be a "
                             f"positive multiple of the {self.token_cache_ways} cache ways")
        if self.token_cache_slots is not None and (
                self.token_cache_slots < self.token_cache_ways or self.token_cache_slots % self.token_cache_ways):
            raise ValueError(f"COMETKV_TOKEN_CACHE_SLOTS ({self.token_cache_slots}) must be a positive multiple "
                             f"of the {self.token_cache_ways} cache ways")
        self.token_cache_stamps = None
        self._token_cache_step_dev = None
        # Replacement policy for the W-way cache (requires ways > 2). "lru": evict the min-stamp
        # way (default, unchanged). "score": evict the way whose token has the WORST current-step
        # selector priority — the selector re-scores every candidate every step, so the current
        # score predicts next-step reuse better than recency. Uses the materialized fp32 score
        # buffer (larger = better). Accuracy-neutral (the cache is exact).
        # Default "score" (paper operating point; requires ways > 2, else the gather falls back to LRU).
        # Env: COMETKV_TOKEN_CACHE_POLICY.
        self.token_cache_policy = os.environ.get("COMETKV_TOKEN_CACHE_POLICY", "score").lower()
        assert self.token_cache_policy in ("lru", "score"), (
            f"Unsupported COMETKV_TOKEN_CACHE_POLICY: {self.token_cache_policy}"
        )
        if self.center_keys:
            self.sig_key_mean = [
                torch.zeros((self.batch_groups, self.head_dim), dtype=torch.float32,
                            device=self.layer_mapping[str(ldx)])
                for ldx in range(self.layer_num)
            ]
        else:
            self.sig_key_mean = None
        # c = mu_q . (k - mu_block) storage. 16 B/token path (8bit + block statistics, the default):
        # the 8-bit code lives in signature byte asym_sig_bits/8 and the grouped scorer decodes it
        # with per-(row, block) ranges, so no per-token buffer exists. fp mode (17 B ablation) and
        # the frozen-statistics scorer keep a per-token fp32 buffer (q_bias).
        self.q_bias = None
        self.sig_block_qbias_lo = None
        self.sig_block_qbias_step = None
        self.qbias_in_signature = (self.qcenter == "8bit" and self.block_stats
                                   and self.asym_sig_bits // 8 < self.sig_bytes - 1)
        if self.qcenter != "off":
            if self.query_aggregation != "q_sum" or self.scorer_mode != "asym" or not self.center_keys:
                raise ValueError("COMETKV_QCENTER requires q_sum aggregation, asym scorer and key centering")
            self.q_mean = [
                torch.zeros((self.batch_groups, self.head_dim), dtype=torch.float32,
                            device=self.layer_mapping[str(ldx)])
                for ldx in range(self.layer_num)
            ]
            if self.qbias_in_signature:
                self.sig_block_qbias_lo = [
                    torch.zeros((self.batch_groups, self.sig_block_capacity), dtype=torch.float32,
                                device=self.layer_mapping[str(ldx)])
                    for ldx in range(self.layer_num)
                ]
                self.sig_block_qbias_step = [torch.ones_like(lo) for lo in self.sig_block_qbias_lo]
            else:
                self.q_bias = [
                    torch.zeros((self.batch_groups, rounded_length), dtype=torch.float32,
                                device=self.layer_mapping[str(ldx)])
                    for ldx in range(self.layer_num)
                ]
        self.base_projection = self._build_random_orth_projection()
        self.projection_dict = {}
        self.projection_t_dict = {}
        self.prefill_signature_chunk_size = 16384
        self.signature_projections = [
            self._projection_for(self.layer_mapping[str(ldx)])
            for ldx in range(self.layer_num)
        ]

        self.static_keys = [
            torch.zeros(
                (self.batch_size, self.kv_head, self.static_pattern_total, self.head_dim),
                dtype=self.dtype,
                device=self.layer_mapping[str(ldx)],
            ).contiguous()
            for ldx in range(self.layer_num)
        ]
        self.static_values = [
            torch.zeros(
                (self.batch_size, self.kv_head, self.static_pattern_total, self.head_dim),
                dtype=self.dtype,
                device=self.layer_mapping[str(ldx)],
            ).contiguous()
            for ldx in range(self.layer_num)
        ]
        self.prompt_recent_keys = [
            torch.zeros(
                (self.batch_size, self.kv_head, self.static_pattern_end, self.head_dim),
                dtype=self.dtype,
                device=self.layer_mapping[str(ldx)],
            ).contiguous()
            for ldx in range(self.layer_num)
        ]
        self.prompt_recent_values = [
            torch.zeros(
                (self.batch_size, self.kv_head, self.static_pattern_end, self.head_dim),
                dtype=self.dtype,
                device=self.layer_mapping[str(ldx)],
            ).contiguous()
            for ldx in range(self.layer_num)
        ]

        self.decode_hot_keys = [
            torch.zeros(
                (self.batch_size, self.kv_head, self.lockstep_local_capacity, self.head_dim),
                dtype=self.dtype,
                device=self.layer_mapping[str(ldx)],
            ).contiguous()
            for ldx in range(self.layer_num)
        ]
        self.decode_hot_values = [
            torch.zeros(
                (self.batch_size, self.kv_head, self.lockstep_local_capacity, self.head_dim),
                dtype=self.dtype,
                device=self.layer_mapping[str(ldx)],
            ).contiguous()
            for ldx in range(self.layer_num)
        ]
        self.decode_hot_token_ids = [
            torch.full(
                (self.batch_size, self.kv_head, self.lockstep_local_capacity),
                -1,
                dtype=torch.int32,
                device=self.layer_mapping[str(ldx)],
            ).contiguous()
            for ldx in range(self.layer_num)
        ]

        self.token_cache_ids = None
        self.token_cache_locks = None
        self.token_cache_keys = None
        self.token_cache_values = None
        self.static_fixed_concat_flash_key_storage = [None for _ in range(self.layer_num)]
        self.static_fixed_concat_flash_value_storage = [None for _ in range(self.layer_num)]
        self.static_fixed_concat_flash_capacity = [0 for _ in range(self.layer_num)]

        # Side streams are shared per device ACROSS cache instances: benchmark drivers build a
        # fresh cache per sample, and per-instance streams strand the allocator's freed blocks
        # (blocks stay associated with the dead stream's pending events), growing reserved memory
        # by hundreds of MB per generate until the device fills and WSL2 starts thrashing.
        self.copystream = self._shared_side_stream(self.primary_device, "copy")
        self.sigstream = self._shared_side_stream(self.primary_device, "sig")
        self.prefill_ready_events = {}
        self.prefill_copy_events = {ldx: [] for ldx in range(self.layer_num)}
        self.prefill_copy_refs = {ldx: [] for ldx in range(self.layer_num)}
        self.prefill_signature_events = {ldx: [] for ldx in range(self.layer_num)}
        self.prefill_signature_refs = {ldx: [] for ldx in range(self.layer_num)}
        for device_idx in self.device_list:
            with torch.cuda.device(device_idx):
                self.prefill_ready_events[device_idx] = torch.cuda.Event()

        self.spill_events = {ldx: [] for ldx in range(self.layer_num)}
        self.profile_decode_update = False
        self.last_decode_update_profile = {}
        self.event_profile_enabled = os.environ.get("COMETKV_EVENT_PROFILE", "0").lower() in ("1", "true", "yes")
        self.event_profile = {}
        self.context = 0
        self.attn_func = self.sparse_attention
        self.range_starts_buffer = None
        self.range_ends_buffer = None
        self.topk_buffer = None
        self.selected_indices_buffer = None
        self.sparse_lengths_buffer = None
        self.active_sparse_len_host = None
        # Max candidate count for the topk launch grid; -1 => full padded signature width (set per plan).
        self.active_candidates_host = -1
        # Committed host copies of the decode retrieval range (set by _update_retrieval_plan).
        self.plan_range1_start_host = 0
        self.plan_range1_end_host = 0
        # Re-commit the per-plan constants (k, -1 padding) on the next decode step.
        self._asym_plan_dirty = True
        self.hit_mask_buffer = None
        self.static_lengths_buffer = None
        self.use_static_prompt_retrieval = False
        self.static_prompt_fast_topk_capacity = 0
        self.static_prompt_fast_recent_capacity = 0
        self.use_chunked_fixed_prompt_local = False
        self.fixed_prompt_local_prompt_lengths_host = None
        self.fixed_prompt_local_layer_visible_lengths_host = None
        self.fixed_prompt_local_static_length_host = 0

    _side_stream_cache = {}

    @classmethod
    def _shared_side_stream(cls, device, name: str):
        key = (str(device), name)
        stream = cls._side_stream_cache.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            cls._side_stream_cache[key] = stream
        return stream

    @staticmethod
    def _device_sort_key(device_name: str) -> int:
        if ":" not in device_name:
            return 0
        return int(device_name.split(":")[-1])

    @staticmethod
    def _round_up(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

    @staticmethod
    def _cuda_device_guard(device):
        target = torch.device(device)
        if target.type != "cuda" or target.index is None:
            return nullcontext()
        if torch.cuda.current_device() == target.index:
            return nullcontext()
        return torch.cuda.device(target)

    def _event_profile_start(self):
        if not self.event_profile_enabled:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _event_profile_end(self, name, start):
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.event_profile.setdefault(name, []).append((start, end))

    def event_profile_summary_s(self):
        return {
            name: sum(start.elapsed_time(end) for start, end in pairs) / 1000.0
            for name, pairs in self.event_profile.items()
        }

    def _build_random_orth_projection(self):
        if self.sig_mode != "random_orth":
            raise NotImplementedError(f"Unsupported sig_mode: {self.sig_mode}")
        if self.sig_bits > self.head_dim:
            raise ValueError(
                f"random_orth requires sig_bits <= head_dim, got {self.sig_bits} > {self.head_dim}"
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.sig_seed)
        basis = torch.randn((self.head_dim, self.head_dim), generator=generator, dtype=torch.float32)
        orth, _ = torch.linalg.qr(basis, mode="reduced")
        # Match the baseline Sig_topk projection, but store it row-major for the CUDA signature kernel.
        return orth[:, : self.sig_bits].t().contiguous()

    def _projection_for(self, device: str):
        device_key = str(device)
        if device_key not in self.projection_dict:
            self.projection_dict[device_key] = self.base_projection.to(device_key, non_blocking=True)
        return self.projection_dict[device_key]

    def _projection_t_for(self, device: str):
        cache_key = str(device)
        if cache_key not in self.projection_t_dict:
            self.projection_t_dict[cache_key] = self.base_projection.transpose(0, 1).contiguous().to(device=device, non_blocking=True)
        return self.projection_t_dict[cache_key]

    @staticmethod
    def _pack_signature_bits(bits: torch.Tensor):
        bit_groups = bits.view(*bits.shape[:-1], -1, 8)
        packed = (
            bit_groups[..., 0]
            | (bit_groups[..., 1] << 1)
            | (bit_groups[..., 2] << 2)
            | (bit_groups[..., 3] << 3)
            | (bit_groups[..., 4] << 4)
            | (bit_groups[..., 5] << 5)
            | (bit_groups[..., 6] << 6)
            | (bit_groups[..., 7] << 7)
        )
        return packed.contiguous()

    def _block0_stat_span(self, prompt_length: int):
        """[start, end) of the tokens that define the block-0 statistics: the prompt minus the sink.
        The sink stays on GPU and is never retrieved; the prompt's recent tokens become retrieval
        candidates at the first window slide with their block-0 signatures, so they must lie
        inside the block-0 quantization ranges. Falls back to the whole prompt if only sink."""
        sink_end = min(prompt_length, self.static_pattern_start)
        return (sink_end, prompt_length) if prompt_length > sink_end else (0, prompt_length)

    def _build_prefill_signatures(self, key_vectors: torch.Tensor, layer_idx: int,
                                  row_start: int = 0, freeze_norm_scale: bool = True,
                                  block_idx: int = 0, stat_span=None):
        """Encode keys into packed signatures. stat_span = [start, end) of the tokens whose
        statistics define the block (key mean, log-norm and c ranges); None = every token."""
        projection_t = self._projection_t_for(self.layer_mapping[str(layer_idx)])
        rows, valid_length, _ = key_vectors.shape
        row_end = row_start + rows
        stat_start, stat_end = stat_span if stat_span is not None else (0, valid_length)
        new_block = self.block_stats and block_idx > 0
        if not 0 <= block_idx < self.sig_block_capacity:
            raise ValueError("Signature block exceeds preallocated capacity")
        packed = torch.empty((rows, valid_length, self.sig_bytes), dtype=torch.uint8, device=key_vectors.device)
        pack_n8 = self.selector_mode == "asym_n8"
        if pack_n8:
            norms = torch.empty((rows, valid_length), dtype=torch.float32, device=key_vectors.device)
        mu = None
        if self.qcenter != "off":
            qbias = torch.empty((rows, valid_length), dtype=torch.float32, device=key_vectors.device)
        if self.center_keys:
            mu_slice = self.sig_key_mean[layer_idx][row_start:row_end]
            if new_block:
                mu = (key_vectors.float().mean(dim=1, keepdim=True)
                      if self.stats_mode == "block" else mu_slice.unsqueeze(1))
                self.sig_block_centers[layer_idx][row_start:row_end, block_idx].copy_(
                    mu.squeeze(1) - mu_slice
                )
            elif freeze_norm_scale:
                # Freeze the per-row key mean from the block-0 span (chunked to bound the fp32
                # transient).
                accum = torch.zeros((rows, self.head_dim), dtype=torch.float32, device=key_vectors.device)
                for chunk_start in range(stat_start, stat_end, self.prefill_signature_chunk_size):
                    chunk_end = min(stat_end, chunk_start + self.prefill_signature_chunk_size)
                    accum += key_vectors[:, chunk_start:chunk_end, :].float().sum(dim=1)
                mu_slice.copy_(accum / max(stat_end - stat_start, 1))
            if mu is None:
                mu = mu_slice.unsqueeze(1)
        for chunk_start in range(0, valid_length, self.prefill_signature_chunk_size):
            chunk_end = min(valid_length, chunk_start + self.prefill_signature_chunk_size)
            # Keep the prefill fast path numerically aligned with the decode-time signature kernel.
            chunk = key_vectors[:, chunk_start:chunk_end, :].float()
            if mu is not None:
                chunk = chunk - mu
            if self.qcenter != "off":
                qbias[:, chunk_start:chunk_end] = torch.einsum(
                    "rtd,rd->rt", chunk, self.q_mean[layer_idx][row_start:row_end])
            logits = torch.matmul(chunk, projection_t)
            bits = (logits >= 0).to(torch.uint8)
            if pack_n8:
                # asym_n8 layout: first asym_sig_bits sign bits; the spare byte carries the norm code.
                bits[..., self.asym_sig_bits:] = 0
                norms[:, chunk_start:chunk_end] = chunk.norm(dim=-1)
            packed[:, chunk_start:chunk_end, :].copy_(self._pack_signature_bits(bits))
        if pack_n8:
            lo = (self.sig_block_norm_lo[layer_idx][row_start:row_end, block_idx]
                  if new_block else self.sig_norm_lo[layer_idx][row_start:row_end])
            step = (self.sig_block_norm_step[layer_idx][row_start:row_end, block_idx]
                    if new_block else self.sig_norm_step[layer_idx][row_start:row_end])
            log_norms = norms.clamp_min(1e-6).log()
            if freeze_norm_scale or new_block:
                # Each sealed block retains its own scale; frozen mode reuses the prompt scale.
                # Optional margin widens the range on each side before quantization.
                lo_vals = log_norms[:, stat_start:stat_end].min(dim=1).values
                hi_vals = log_norms[:, stat_start:stat_end].max(dim=1).values
                if self.norm_margin > 0:
                    span = (hi_vals - lo_vals).clamp_min(1e-6)
                    lo_vals = lo_vals - self.norm_margin * span
                    hi_vals = hi_vals + self.norm_margin * span
                lo.copy_(lo_vals)
                step.copy_(((hi_vals - lo_vals) / 255.0).clamp_min(1e-8))
            code = torch.round((log_norms - lo.unsqueeze(1)) / step.unsqueeze(1)).clamp(0, 255)
            packed[..., self.sig_bytes - 1] = code.to(torch.uint8)
            if block_idx == 0:
                self.sig_block_norm_lo[layer_idx][row_start:row_end, 0].copy_(lo)
                self.sig_block_norm_step[layer_idx][row_start:row_end, 0].copy_(step)
        if self.qcenter == "8bit":
            q_lo = qbias[:, stat_start:stat_end].min(dim=1, keepdim=True).values
            q_step = ((qbias[:, stat_start:stat_end].max(dim=1, keepdim=True).values - q_lo)
                      / 255.0).clamp_min(1e-8)
            q_code = torch.round((qbias - q_lo) / q_step).clamp(0, 255)
            if self.asym_sig_bits // 8 < self.sig_bytes - 1:
                packed[..., self.asym_sig_bits // 8] = q_code.to(torch.uint8)
            if self.qbias_in_signature:
                # The scorer decodes lo + code * step from the signature byte itself.
                self.sig_block_qbias_lo[layer_idx][row_start:row_end, block_idx].copy_(q_lo.squeeze(1))
                self.sig_block_qbias_step[layer_idx][row_start:row_end, block_idx].copy_(q_step.squeeze(1))
            qbias = q_lo + q_code * q_step       # the value the scorer sees (decoded code)
        # Per-token copy only for the fp32-buffer path (fp mode / frozen statistics).
        self._last_qbias = qbias if self.q_bias is not None else None
        return packed

    def _qbias_from_signatures(self, layer_idx: int) -> torch.Tensor:
        """c values the 16 B/token scorer decodes from the signatures, [rows, tokens] (diagnostics)."""
        if not self.qbias_in_signature:
            raise RuntimeError("c is not stored in the signature (qcenter must be 8bit with block statistics)")
        rows = self.batch_groups
        codes = self.signature_index[layer_idx].view(rows, -1, self.sig_bytes)[..., self.asym_sig_bits // 8]
        tokens = torch.arange(codes.size(1), device=codes.device)
        prompt = int(self.lockstep_prompt_length_host)
        epoch = torch.where(
            tokens < prompt, torch.zeros_like(tokens),
            1 + torch.div(tokens - prompt + self._fixed_prompt_local_recent_overlap(),
                          self._fixed_prompt_local_slide_stride(), rounding_mode="floor"),
        ).clamp_(max=self.sig_block_capacity - 1).expand(rows, -1)
        lo = self.sig_block_qbias_lo[layer_idx].gather(1, epoch)
        step = self.sig_block_qbias_step[layer_idx].gather(1, epoch)
        return lo + codes.float() * step

    def _full_recompute_stats(self, layer_idx: int, token_end: int):
        """Recompute μ, norm range from ALL keys in the retrieval index [0:token_end], then
        rebuild ALL signatures. Called periodically (every full_recompute_interval decode tokens)
        on the eviction path. Returns elapsed ms for profiling."""
        import time
        t0 = time.perf_counter()
        device = self.layer_mapping[str(layer_idx)]
        # Read all keys from the retrieval store: [batch_groups, token_end, head_dim]
        if self.kv_on_gpu:
            all_keys = self.cpu_key_cache[layer_idx][:, :token_end, :].to(torch.float32)
        else:
            # CPU pinned → copy to GPU for fast GEMM. non_blocking=False to ensure data is ready.
            all_keys = self.cpu_key_cache[layer_idx][:, :token_end, :].to(device, non_blocking=False).to(torch.float32)
        if self.quantize_cpu_kv:
            all_keys.mul_(self.cpu_kv_k_scale[layer_idx].unsqueeze(1))
        rows, n_tokens, _ = all_keys.shape

        # --- recompute μ ---
        if self.center_keys and self.sig_key_mean is not None:
            mu = all_keys.mean(dim=1)  # [batch_groups, head_dim]
            self.sig_key_mean[layer_idx].copy_(mu)

        # --- recompute norm range ---
        if self.selector_mode == "asym_n8" and self.sig_norm_lo is not None:
            centered = all_keys - self.sig_key_mean[layer_idx].unsqueeze(1) if self.center_keys else all_keys
            norms = centered.norm(dim=-1).clamp_min(1e-6).log()  # [rows, n_tokens]
            lo_vals = norms.min(dim=1).values
            hi_vals = norms.max(dim=1).values
            if self.norm_margin > 0:
                span = (hi_vals - lo_vals).clamp_min(1e-6)
                lo_vals = lo_vals - self.norm_margin * span
                hi_vals = hi_vals + self.norm_margin * span
            self.sig_norm_lo[layer_idx].copy_(lo_vals)
            self.sig_norm_step[layer_idx].copy_(((hi_vals - lo_vals) / 255.0).clamp_min(1e-8))

        # --- rebuild ALL signatures ---
        packed = self._build_prefill_signatures(
            all_keys, layer_idx, row_start=0, freeze_norm_scale=False,
        )
        self.signature_index[layer_idx][:, :, :token_end, :].copy_(
            packed.view(self.batch_size, self.kv_head, token_end, self.sig_bytes)
        )

        # Synchronize to get accurate timing (eviction path is synchronous anyway)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        self.last_full_recompute_ms = (time.perf_counter() - t0) * 1000.0
        if os.environ.get("COMETKV_DEBUG_RECOMPUTE", "0") == "1" and layer_idx == 0:
            print(f"  [recompute] layer={layer_idx} tokens={n_tokens} "
                  f"μ_norm={float(self.sig_key_mean[layer_idx].norm()):.4f} "
                  f"norm_lo={float(self.sig_norm_lo[layer_idx].mean()):.4f} "
                  f"norm_step={float(self.sig_norm_step[layer_idx].mean()):.6f} "
                  f"ms={self.last_full_recompute_ms:.2f}")

    def _freeze_k_per_channel_scale(self, layer_idx: int, row_start: int, key_rows: torch.Tensor):
        # key_rows: [n_rows, n_tokens, head_dim] full precision. Freeze per-channel K scale from the
        # prompt (one scale per (row, channel), shared across tokens). Decode-evicted tokens reuse it.
        # Chunked over tokens to bound the fp32 transient (a full-prompt .float() is multi-GB at 128k).
        n_rows, n_tokens, _ = key_rows.shape
        amax = torch.full((n_rows, self.head_dim), 1e-8, dtype=torch.float32, device=key_rows.device)
        for chunk_start in range(0, n_tokens, self.prefill_signature_chunk_size):
            chunk_end = min(n_tokens, chunk_start + self.prefill_signature_chunk_size)
            chunk_amax = key_rows[:, chunk_start:chunk_end, :].float().abs().amax(dim=1)
            amax = torch.maximum(amax, chunk_amax)
        self.cpu_kv_k_scale[layer_idx][row_start:row_start + n_rows].copy_(amax / 127.0)

    def _quantize_kv_into_cpu(self, layer_idx: int, row_start: int, token_start: int,
                              key_rows: torch.Tensor, value_rows: torch.Tensor):
        # Quantize [n_rows, n_tokens, head_dim] KV into the pinned int8 store: K per-channel with the
        # frozen scale, V per-token. Chunked over tokens to bound the fp32 transient and the lifetime of
        # the GPU int8 source tensors (returned so the caller can keep async D2H sources alive).
        n_rows, n_tokens, _ = key_rows.shape
        row_end = row_start + n_rows
        k_scale = self.cpu_kv_k_scale[layer_idx][row_start:row_end].unsqueeze(1)  # [n_rows,1,head_dim]
        refs = []
        for chunk_start in range(0, n_tokens, self.prefill_signature_chunk_size):
            chunk_end = min(n_tokens, chunk_start + self.prefill_signature_chunk_size)
            k_chunk = key_rows[:, chunk_start:chunk_end, :].float()
            v_chunk = value_rows[:, chunk_start:chunk_end, :].float()
            k_int8 = torch.clamp(torch.round(k_chunk / k_scale), -127, 127).to(torch.int8)
            v_scale = v_chunk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
            v_int8 = torch.clamp(torch.round(v_chunk / v_scale), -127, 127).to(torch.int8)
            v_scale_2d = v_scale.squeeze(-1).contiguous()
            if self.kv_on_gpu:
                # GPU-resident store: dst is on-device, the D2H int8 helper does not apply. Write the
                # int8 codes + per-token V scale directly (D2D strided) into the interleaved store.
                dst_lo = token_start + chunk_start
                dst_hi = token_start + chunk_end
                self.cpu_kv_cache[layer_idx][row_start:row_end, dst_lo:dst_hi, 0, :].copy_(k_int8, non_blocking=True)
                self.cpu_kv_cache[layer_idx][row_start:row_end, dst_lo:dst_hi, 1, :].copy_(v_int8, non_blocking=True)
                self.cpu_kv_v_scale[layer_idx][row_start:row_end, dst_lo:dst_hi].copy_(v_scale_2d, non_blocking=True)
            else:
                copy_interleaved_int8_kv_to_cpu(
                    k_int8,
                    v_int8,
                    v_scale_2d,
                    self.cpu_kv_cache[layer_idx],
                    self.cpu_kv_v_scale[layer_idx],
                    row_start,
                    token_start + chunk_start,
                )
            refs.append((k_int8, v_int8, v_scale_2d))
        return refs

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        if value <= 1:
            return 1
        return 1 << (value - 1).bit_length()

    def _max_initial_prompt_length(self) -> int:
        valid_start = self.valid_start_list
        if torch.is_tensor(valid_start):
            valid_values = valid_start.detach().cpu().tolist()
        elif hasattr(valid_start, "tolist"):
            valid_values = valid_start.tolist()
        elif isinstance(valid_start, (list, tuple)):
            valid_values = list(valid_start)
        else:
            valid_values = [valid_start]
        if not isinstance(valid_values, list):
            valid_values = [valid_values]
        if len(valid_values) == 0:
            return max(int(self.prompt_capacity), 0)
        min_valid_start = min(int(value) for value in valid_values)
        return max(int(self.prompt_capacity) - min_valid_start, 0)

    def _valid_start_host_list(self):
        valid_start = self.valid_start_list
        if torch.is_tensor(valid_start):
            values = valid_start.detach().cpu().tolist()
        elif hasattr(valid_start, "tolist"):
            values = valid_start.tolist()
        elif isinstance(valid_start, (list, tuple)):
            values = list(valid_start)
        else:
            values = [valid_start]
        if not isinstance(values, list):
            values = [values]
        return [int(value) for value in values]

    def _assert_fast_only_lockstep_supported(self):
        starts = self._valid_start_host_list()
        assert len(starts) == self.batch_size
        assert all(value == starts[0] for value in starts), (
            "CometKV fast-only lockstep requires same valid_start"
        )
        assert starts[0] == 0, (
            "CometKV fast-only lockstep requires valid_start == 0"
        )

    def _static_retrieval_length_for_prompt(self, prompt_length: int) -> int:
        sink_end = min(prompt_length, self.static_pattern_start)
        recent_start = max(sink_end, prompt_length - self.static_pattern_end)
        prompt_retrieval_end = min(prompt_length, recent_start)
        return max(prompt_retrieval_end - sink_end, 0)

    def _static_preserved_length_for_prompt(self, prompt_length: int) -> int:
        sink_end = min(prompt_length, self.static_pattern_start)
        recent_start = max(sink_end, prompt_length - self.static_pattern_end)
        prompt_recent_count = max(prompt_length - recent_start, 0)
        return sink_end + prompt_recent_count

    def _recommended_initial_token_cache_size(self) -> int:
        prompt_length = self._max_initial_prompt_length()
        retrieval_length = self._static_retrieval_length_for_prompt(prompt_length)
        max_topk = self._compute_topk(
            retrieval_length,
            preserved_length=self._static_preserved_length_for_prompt(prompt_length),
            visible_length=prompt_length,
        )
        if max_topk <= self.base_sig_token_cache_size:
            return self.base_sig_token_cache_size
        return max(self._round_up(max_topk, 1024), self.base_sig_token_cache_size)

    def _recommended_initial_token_cache_layer_sizes(self):
        return [self._recommended_initial_token_cache_size() for _ in range(self.layer_num)]

    def _token_cache_bytes_per_slot(self) -> int:
        # ids (int32) + locks (int32) [+ LRU stamps (int32)] + K + V in model dtype, per (row, slot).
        dtype_size = torch.tensor([], dtype=self.dtype).element_size()
        meta_ints = 3 if self.token_cache_lru else 2
        return self.batch_groups * (2 * self.head_dim * dtype_size + meta_ints * 4)

    def _token_cache_capacity_fitting_free_memory(self) -> int:
        """Largest per-layer token-cache capacity that fits in the residual VRAM of every
        device holding layers, after reserving headroom for the decode/concat buffers and
        the CUDA-graph pool that are allocated after the token cache."""
        # 4 GiB default: must cover everything allocated AFTER the token cache — CG concat
        # buffers (~1.3GB at 10% budget), the CUDA-graph pool, flash workspace, and allocator
        # fragmentation headroom (a 24GB 4090 at 10% budget OOMs with a 3 GiB reserve).
        reserve_bytes = int(float(os.environ.get("COMETKV_TOKEN_CACHE_RESERVE_GB", "4.0")) * (1 << 30))
        layers_per_device = {}
        for ldx in range(self.layer_num):
            device = str(self.layer_mapping[str(ldx)])
            layers_per_device[device] = layers_per_device.get(device, 0) + 1
        bytes_per_slot = self._token_cache_bytes_per_slot()
        fit = None
        for device, n_layers in layers_per_device.items():
            total = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated(device)
            usable = total - allocated - reserve_bytes
            cap = max(int(usable // (n_layers * bytes_per_slot)), 0)
            fit = cap if fit is None else min(fit, cap)
        return (fit // 1024) * 1024 if fit else 0

    def _refresh_token_cache_sizes_for_temporal_reuse(self):
        """Size the token cache to mult x K_t records per (row, layer) — paper: 2K_t, the default
        COMETKV_TOKEN_CACHE_MULT=2 — so the cross-step selection overlap turns into UVA-gather
        hits. K_t is the largest length-based budget floor(rho * L_t) of the request (prompt plus
        max_new_length), since the cache is allocated once; the size is rounded up to
        COMETKV_TOKEN_CACHE_ROUND slots. Called at prepare_cache time — after prefill (weights/signatures/retrieval
        store are all allocated, so memory_allocated is an honest floor) and before the decode
        buffers. If residual VRAM cannot hold it, the cache shrinks to what fits (never below
        the init-time ~1x-topk rule) and a warning is printed. COMETKV_TOKEN_CACHE_MULT<=1
        restores the legacy 1024-rounded top-k sizing. COMETKV_TOKEN_CACHE_ROUND sets the rounding
        granularity (default 8 = one set; 1024 = the legacy granularity); COMETKV_TOKEN_CACHE_SLOTS
        fixes the capacity outright."""
        fit = self._token_cache_capacity_fitting_free_memory()
        if self.token_cache_slots is not None:
            size = self.token_cache_slots
            if size > fit:
                print(f"CometKV warning: COMETKV_TOKEN_CACHE_SLOTS={size} exceeds the {fit} slots that fit "
                      f"in residual VRAM", flush=True)
            self.sig_token_cache_layer_sizes = [size for _ in range(self.layer_num)]
            self.sig_token_cache_size = size
            return
        mult = self.token_cache_mult
        if mult <= 1.0:
            return
        visible_length = self._max_initial_prompt_length() + max(int(self.max_new_length), 0)
        max_budget = self._compute_topk(
            self._static_retrieval_length_for_prompt(visible_length),
            preserved_length=self._static_preserved_length_for_prompt(visible_length),
            visible_length=visible_length,
        )
        desired = self._round_up(max(math.ceil(max_budget * mult), 1), self.token_cache_round)
        if desired <= fit:
            size = desired
        else:
            size = max(fit, self._recommended_initial_token_cache_size())
            print(f"CometKV warning: token cache wants {desired} slots ({mult:g} x K_t, K_t={max_budget}) "
                  f"but residual VRAM fits {fit}; using {size}", flush=True)
        self.sig_token_cache_layer_sizes = [size for _ in range(self.layer_num)]
        self.sig_token_cache_size = size

    def _token_cache_size_for_layer(self, layer_idx: int) -> int:
        return self.sig_token_cache_layer_sizes[layer_idx]

    def _allocate_token_cache(self):
        if self.kv_on_gpu or self.disable_token_cache:
            # GPU-resident store gathers directly from device memory; the token cache (a UVA-miss
            # amortiser) is pure overhead, so the GPU gather kernels take no token-cache buffers.
            return
        if self.token_cache_ids is not None:
            return
        self._refresh_token_cache_sizes_for_temporal_reuse()
        self.token_cache_ids = [
            torch.full(
                (self.batch_groups, self._token_cache_size_for_layer(ldx)),
                -1,
                dtype=torch.int32,
                device=self.layer_mapping[str(ldx)],
            ).contiguous()
            for ldx in range(self.layer_num)
        ]
        self.token_cache_locks = [
            torch.zeros(
                (self.batch_groups, self._token_cache_size_for_layer(ldx)),
                dtype=torch.int32,
                device=self.layer_mapping[str(ldx)],
            ).contiguous()
            for ldx in range(self.layer_num)
        ]
        self.token_cache_keys = [
            torch.zeros(
                (self.batch_groups, self._token_cache_size_for_layer(ldx), self.head_dim),
                dtype=self.dtype,
                device=self.layer_mapping[str(ldx)],
            ).contiguous()
            for ldx in range(self.layer_num)
        ]
        self.token_cache_values = [
            torch.zeros(
                (self.batch_groups, self._token_cache_size_for_layer(ldx), self.head_dim),
                dtype=self.dtype,
                device=self.layer_mapping[str(ldx)],
            ).contiguous()
            for ldx in range(self.layer_num)
        ]
        if self.token_cache_lru:
            # Per-slot last-touch stamps (0 = never touched) + a per-device monotonic decode-step
            # counter, both read by the gather kernel to evict the least-recently-used way.
            self.token_cache_stamps = [
                torch.zeros(
                    (self.batch_groups, self._token_cache_size_for_layer(ldx)),
                    dtype=torch.int32,
                    device=self.layer_mapping[str(ldx)],
                ).contiguous()
                for ldx in range(self.layer_num)
            ]
            self._token_cache_step_dev = {
                dev: torch.zeros((1,), dtype=torch.int32, device=dev)
                for dev in {self.layer_mapping[str(ldx)] for ldx in range(self.layer_num)}
            }

    def _token_cache_lru_args(self, layer_idx):
        # (cache_stamps, step_counter, ways) for the gather launcher; (None, None, 2) keeps the
        # legacy 2-way path. In CG mode the driver refreshes the step counter each replay
        # (cg_set_step_inputs); in eager mode the flash-from-indices path fills it per step.
        if not self.token_cache_lru or self.token_cache_stamps is None:
            return None, None, 2
        dev = self.layer_mapping[str(layer_idx)]
        return self.token_cache_stamps[layer_idx], self._token_cache_step_dev[dev], self.token_cache_ways

    def _token_cache_prio_args(self, layer_idx):
        # (prio_buckets, prio_scores) for the gather launcher's "score" replacement policy: the
        # per-(row, token) fp32 priority materialized for THIS layer this step (larger = better;
        # -inf outside candidates). The persistent per-device buffer has a fixed address, so it
        # is CUDA-graph capturable. Selection always runs before the gather in the same step.
        if self.token_cache_policy != "score" or not self.token_cache_lru:
            return None, None
        dev = str(self.layer_mapping[str(layer_idx)])
        return None, self._asym_scores_buffers.get(dev)

    def _allocate_decode_buffers(self):
        if torch.count_nonzero(self.prompt_lengths).item() == 0:
            max_topk = 1
        else:
            max_topk = max(self._static_prompt_fast_topk_capacity_from_tensors(), 1)
        self.range_starts_buffer = torch.zeros((self.batch_groups, 2), dtype=torch.int32, device=self.primary_device)
        self.range_ends_buffer = torch.zeros_like(self.range_starts_buffer)
        self.topk_buffer = torch.zeros((self.batch_groups,), dtype=torch.int32, device=self.primary_device)
        self.selected_indices_buffer = torch.full((self.batch_groups, max_topk), -1, dtype=torch.int32, device=self.primary_device)
        self.sparse_lengths_buffer = torch.zeros((self.batch_groups,), dtype=torch.int32, device=self.primary_device)
        self.active_sparse_len_host = 0
        self.hit_mask_buffer = torch.zeros((self.batch_groups, max_topk), dtype=torch.int32, device=self.primary_device)
        self.static_lengths_buffer = torch.full((self.batch_groups,), self.static_pattern_total, dtype=torch.int32, device=self.primary_device)

        # CUDA-graph fixed-shape concat buffers (layout [static | sparse | recent_capacity], width W).
        # Fixed addresses + fixed shapes so the whole decode step can be captured; the per-step live
        # length is supplied via _cg_cache_seqlens (set eagerly by the replay driver, NOT in-graph).
        max_topk = self.selected_indices_buffer.size(1)
        recent_cap = self._decode_hot_capacity()
        self.cg_concat_width = self.static_pattern_total + max_topk + recent_cap
        self._cg_concat_k = [None for _ in range(self.layer_num)]
        self._cg_concat_v = [None for _ in range(self.layer_num)]
        self._cg_legacy_tmp = None  # lazily allocated by the COMETKV_CG_DIRECT_GATHER=0 A/B path
        self._cg_cache_seqlens = torch.zeros((self.batch_size,), dtype=torch.int32, device=self.primary_device)
        self._cg_decode_step_dev = torch.zeros((1,), dtype=torch.int32, device=self.primary_device)
        self.recall_snapshots = {}
        if int(os.environ.get("COMETKV_RECALL_INTERVAL", "0")) > 0:
            for layer in sorted({0, self.layer_num - 1}):
                device = self.layer_mapping[str(layer)]
                self.recall_snapshots[layer] = (
                    torch.empty((self.batch_size, 1, self.num_heads, self.head_dim), dtype=self.dtype, device=device),
                    torch.empty_like(self.selected_indices_buffer, device=device),
                )

    def _ensure_cg_concat_buffers(self, layer_idx):
        if self._cg_concat_k[layer_idx] is not None:
            return
        device = self.layer_mapping[str(layer_idx)]
        W = self.cg_concat_width
        self._cg_concat_k[layer_idx] = torch.zeros(self.batch_groups * W * self.head_dim, dtype=self.dtype, device=device)
        self._cg_concat_v[layer_idx] = torch.zeros_like(self._cg_concat_k[layer_idx])

    def _sink_ends(self):
        return torch.minimum(
            self.prompt_lengths,
            torch.full_like(self.prompt_lengths, self.static_pattern_start),
        )

    def _recent_starts_for_visible(self, visible_lengths):
        return torch.maximum(self._sink_ends(), visible_lengths - self.static_pattern_end)

    def _refresh_static_recent_state(self, layer_idx: int, visible_lengths):
        if self.static_lengths_buffer is None:
            return
        if (
            torch.is_tensor(visible_lengths)
            and visible_lengths.is_cuda
            and str(visible_lengths.device) == str(self.primary_device)
            and str(self.prompt_lengths.device) == str(self.primary_device)
            # The fused kernel needs every tensor it touches on ONE device: this layer's
            # static/prompt-recent buffers live on layer_mapping's device, so off-primary
            # layers (sharded models) must take the device-safe Python loop below.
            and str(self.layer_mapping[str(layer_idx)]) == str(self.primary_device)
        ):
            # Kernels launch on the CURRENT device, which callers don't pin: with the cache
            # on cuda:1 and ambient device cuda:0 the launch dereferences foreign pointers
            # (illegal memory access). Guard onto the device that owns the buffers.
            with torch.cuda.device(self.primary_device):
                refresh_static_prompt_recent_state(
                    self.prompt_lengths,
                    visible_lengths.to(device=self.primary_device, dtype=torch.int32, non_blocking=True),
                    self.prompt_recent_keys[layer_idx],
                    self.prompt_recent_values[layer_idx],
                    self.static_keys[layer_idx],
                    self.static_values[layer_idx],
                    self.static_lengths_buffer,
                    self.static_pattern_start,
                    self.static_pattern_end,
                )
            return

        for batch_idx in range(self.batch_size):
            prompt_length = int(self.prompt_lengths[batch_idx].item())
            visible_length = int(visible_lengths[batch_idx].item())
            sink_end = min(prompt_length, self.static_pattern_start)
            recent_start = max(sink_end, visible_length - self.static_pattern_end)
            prompt_recent_count = max(min(prompt_length, visible_length) - recent_start, 0)
            prompt_recent_count = min(prompt_recent_count, self.static_pattern_end)
            static_length = sink_end + prompt_recent_count

            row_start = batch_idx * self.kv_head
            row_end = row_start + self.kv_head
            self.static_lengths_buffer[row_start:row_end].fill_(static_length)

            if prompt_recent_count > 0:
                src_start = self.static_pattern_end - prompt_recent_count
                src_end = src_start + prompt_recent_count
                dst_start = self.static_pattern_start
                dst_end = dst_start + prompt_recent_count
                self.static_keys[layer_idx][batch_idx, :, dst_start:dst_end, :].copy_(
                    self.prompt_recent_keys[layer_idx][batch_idx, :, src_start:src_end, :]
                )
                self.static_values[layer_idx][batch_idx, :, dst_start:dst_end, :].copy_(
                    self.prompt_recent_values[layer_idx][batch_idx, :, src_start:src_end, :]
                )

    def _initialize_static_prompt_retrieval(self):
        self.static_prompt_fast_topk_capacity = self._static_prompt_fast_topk_capacity_from_tensors()
        self.static_prompt_fast_recent_capacity = self._static_prompt_fast_recent_capacity()
        self.use_chunked_fixed_prompt_local = bool(
            max(self.max_new_length - 1, 0) > self._fixed_prompt_local_slide_stride()
        )
        self.fixed_prompt_local_prompt_lengths_host = [
            int(length)
            for length in self.prompt_lengths.detach().cpu().tolist()
        ]
        assert all(
            length == self.fixed_prompt_local_prompt_lengths_host[0]
            for length in self.fixed_prompt_local_prompt_lengths_host
        ), "CometKV fast-only lockstep requires same prompt length"
        self.lockstep_prompt_length_host = self.fixed_prompt_local_prompt_lengths_host[0]
        self.lockstep_decode_step_host = max(
            int(self.context) - self.lockstep_prompt_length_host,
            0,
        )
        self.lockstep_evicted_retrieval_end_host = max(
            self.lockstep_prompt_length_host - self.static_pattern_end,
            0,
        )
        self.lockstep_pending_update_tokens = 0
        prompt_length = self.fixed_prompt_local_prompt_lengths_host[0]
        self.fixed_prompt_local_layer_visible_lengths_host = [
            prompt_length
            for _ in range(self.layer_num)
        ]
        self.fixed_prompt_local_static_length_host = self.static_pattern_total
        self._update_retrieval_plan(self.visible_lengths)
        self._refresh_static_recent_state(0, self.visible_lengths)

    def _fixed_prompt_local_recent_capacity(self):
        return self.lockstep_local_capacity

    def _fixed_prompt_local_slide_stride(self):
        return self.lockstep_update_interval

    def _fixed_prompt_local_recent_overlap(self):
        return max(
            self._fixed_prompt_local_recent_capacity()
            - self._fixed_prompt_local_slide_stride(),
            0,
        )

    def _decode_hot_capacity(self):
        return self.decode_hot_keys[0].size(2)

    def _static_prompt_fast_recent_capacity(self):
        return min(
            max(self.max_new_length - 1, 0),
            self._fixed_prompt_local_recent_capacity(),
        )

    def _can_use_static_prompt_retrieval(self):
        if not torch.all(self.prompt_lengths >= self.static_pattern_total).item():
            return False
        max_decode_tokens = max(self.max_new_length - 1, 0)
        standard_capacity = self.static_pattern_end
        if max_decode_tokens <= standard_capacity:
            return True
        return (
            self._fixed_prompt_local_recent_capacity() >= self.static_pattern_end * 2
        )

    def _static_prompt_fast_topk_capacity_from_tensors(self):
        sink_ends = torch.minimum(
            self.prompt_lengths,
            torch.full_like(self.prompt_lengths, self.static_pattern_start),
        )
        # Allocate for the whole planned generation, not just prompt + recent tokens.
        # Candidate/preserved counts below are conservative bounds: the growing local
        # window always preserves at least static_pattern_end recent tokens, and can
        # only reduce k. This covers include-preserved budgets across slides as well.
        max_visible_lengths = self.prompt_lengths + max(self.max_new_length - 1, 0)
        recent_starts = torch.maximum(sink_ends, max_visible_lengths - self.static_pattern_end)
        retrieval_lengths = (recent_starts - sink_ends).clamp(min=0)
        preserved_lengths = sink_ends + max_visible_lengths - recent_starts
        return int(self._compute_topk_for_lengths(
            retrieval_lengths, preserved_lengths, max_visible_lengths,
        ).max().item())

    def _valid_lengths_from_prefill(self, seq_len: int):
        self._assert_fast_only_lockstep_supported()
        lengths = torch.as_tensor(seq_len - self.valid_start_list, dtype=torch.int32, device=self.primary_device)
        lengths_host = [int(value) for value in lengths.detach().cpu().tolist()]
        assert all(value == lengths_host[0] for value in lengths_host), (
            "CometKV fast-only lockstep requires same prompt length"
        )
        self.lockstep_prompt_length_host = lengths_host[0]
        self.lockstep_decode_step_host = 0
        self.lockstep_evicted_retrieval_end_host = max(
            self.lockstep_prompt_length_host - self.static_pattern_end,
            0,
        )
        self.lockstep_pending_update_tokens = 0
        self.prompt_lengths.copy_(lengths)
        self.visible_lengths.copy_(lengths)
        self.layer_visible_lengths.copy_(lengths.unsqueeze(0).expand(self.layer_num, -1))

    def _interleave_kv_into_gpu_store(self, layer_idx, row_start, token_start, key_rows, value_rows):
        # CometKV_GPU: cpu_kv_cache lives in device memory, so the D2H copy_interleaved_kv_to_cpu
        # (cudaMemcpyDeviceToHost) does not apply. Interleave on-device into the [tokens, 2, dim] layout
        # via a contiguous staging tensor + one D2D copy; return it as the copy keep-alive ref.
        n_rows, n_tokens, _ = key_rows.shape
        kv = torch.empty(
            (n_rows, n_tokens, 2, self.head_dim), dtype=self.dtype, device=key_rows.device
        )
        kv[:, :, 0, :].copy_(key_rows)
        kv[:, :, 1, :].copy_(value_rows)
        self.cpu_kv_cache[layer_idx][
            row_start:row_start + n_rows, token_start:token_start + n_tokens, :, :
        ].copy_(kv, non_blocking=True)
        return kv

    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, start_bdx):
        bsz, seq_len, _, _ = key_states.shape
        assert bsz <= self.prefill_bsz, f"Prefilling batch size ({bsz}) should <= {self.prefill_bsz}."
        assert seq_len <= self.max_length, f"Prefilling sequence length ({seq_len}) exceeds max length ({self.max_length})."

        if self.context == 0 and start_bdx == 0 and torch.count_nonzero(self.prompt_lengths).item() == 0:
            self._valid_lengths_from_prefill(seq_len)

        valid_start = int(self.valid_start_list[start_bdx])
        valid_length = seq_len - valid_start
        end_bdx = start_bdx + bsz
        device = self.layer_mapping[str(layer_idx)]
        row_start = start_bdx * self.kv_head
        row_end = end_bdx * self.kv_head
        key_slice = key_states[:, valid_start:valid_start + valid_length].transpose(1, 2)
        value_slice = value_states[:, valid_start:valid_start + valid_length].transpose(1, 2)
        key_vectors = key_slice.reshape(bsz * self.kv_head, valid_length, self.head_dim).contiguous()
        value_vectors = value_slice.reshape(bsz * self.kv_head, valid_length, self.head_dim).contiguous()
        key_vectors_4d = key_vectors.view(bsz, self.kv_head, valid_length, self.head_dim)
        value_vectors_4d = value_vectors.view(bsz, self.kv_head, valid_length, self.head_dim)

        self.prefill_ready_events[device].record()
        with torch.cuda.stream(self.copystream):
            self.prefill_ready_events[device].wait()
            if self.quantize_cpu_kv:
                # Freeze the per-channel K scale from the prompt, then quantize+spill int8 codes.
                # Keep the int8 quant temps alive until the async D2H event completes.
                self._freeze_k_per_channel_scale(layer_idx, row_start, key_vectors)
                quant_refs = self._quantize_kv_into_cpu(
                    layer_idx, row_start, 0, key_vectors, value_vectors
                )
                copy_ref = (key_vectors, value_vectors, quant_refs)
            elif self.kv_on_gpu:
                copy_ref = self._interleave_kv_into_gpu_store(
                    layer_idx, row_start, 0, key_vectors, value_vectors
                )
            else:
                copy_interleaved_kv_to_cpu(
                    key_vectors,
                    value_vectors,
                    self.cpu_kv_cache[layer_idx],
                    row_start,
                    0,
                )
                copy_ref = (key_vectors, value_vectors)
            copy_event = torch.cuda.Event()
            copy_event.record(self.copystream)
            self.prefill_copy_events[layer_idx].append(copy_event)
            self.prefill_copy_refs[layer_idx].append(copy_ref)

        sink_end = min(valid_length, self.static_pattern_start)
        prompt_recent_start = max(sink_end, valid_length - self.static_pattern_end)
        prompt_recent_count = max(valid_length - prompt_recent_start, 0)

        if sink_end > 0:
            self.static_keys[layer_idx][start_bdx:end_bdx, :, :sink_end, :].copy_(
                key_vectors_4d[:, :, :sink_end, :]
            )
            self.static_values[layer_idx][start_bdx:end_bdx, :, :sink_end, :].copy_(
                value_vectors_4d[:, :, :sink_end, :]
            )
        if prompt_recent_count > 0:
            dst_start = self.static_pattern_start
            dst_end = dst_start + prompt_recent_count
            self.prompt_recent_keys[layer_idx][start_bdx:end_bdx, :, :prompt_recent_count, :].copy_(
                key_vectors_4d[:, :, prompt_recent_start:valid_length, :]
            )
            self.prompt_recent_values[layer_idx][start_bdx:end_bdx, :, :prompt_recent_count, :].copy_(
                value_vectors_4d[:, :, prompt_recent_start:valid_length, :]
            )
            self.static_keys[layer_idx][start_bdx:end_bdx, :, dst_start:dst_end, :].copy_(
                key_vectors_4d[:, :, prompt_recent_start:valid_length, :]
            )
            self.static_values[layer_idx][start_bdx:end_bdx, :, dst_start:dst_end, :].copy_(
                value_vectors_4d[:, :, prompt_recent_start:valid_length, :]
            )

        # Block-0 statistics come from the prompt minus the sink (never retrieved).
        stat_span = self._block0_stat_span(valid_length)
        if self.qcenter != "off":
            # One q_sum mean per (layer, row) over the same span, frozen for the whole request.
            stat_count = stat_span[1] - stat_span[0]
            q_valid = query_states[:, valid_start + stat_span[0]:valid_start + stat_span[1]]
            q_sum = q_valid.reshape(bsz, stat_count, self.kv_head, self.group_size, self.head_dim)
            self.q_mean[layer_idx][row_start:row_end].copy_(
                (q_sum.sum(dim=(1, 3), dtype=torch.float32) / max(stat_count, 1))
                .reshape(bsz * self.kv_head, self.head_dim))
        signature_ready_event = torch.cuda.Event()
        signature_ready_event.record()
        with torch.cuda.stream(self.sigstream):
            signature_ready_event.wait()
            packed_rows = self._build_prefill_signatures(
                key_vectors,
                layer_idx,
                row_start=row_start,
                freeze_norm_scale=True,
                stat_span=stat_span,
            )
            self.signature_index[layer_idx][start_bdx:end_bdx, :, :valid_length, :].copy_(
                packed_rows.view(bsz, self.kv_head, valid_length, self.sig_bytes)
            )
            if self.q_bias is not None:
                self.q_bias[layer_idx][row_start:row_end, :valid_length].copy_(self._last_qbias)
            signature_event = torch.cuda.Event()
            signature_event.record(self.sigstream)
            self.prefill_signature_events[layer_idx].append(signature_event)
            self.prefill_signature_refs[layer_idx].append(key_vectors)

        if (layer_idx == self.layer_num - 1) and (end_bdx == self.batch_size):
            self.context += seq_len

        return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]

    def _sync_prefill_layer(self, layer_idx):
        for event in self.prefill_copy_events[layer_idx]:
            event.synchronize()
        self.prefill_copy_events[layer_idx].clear()
        self.prefill_copy_refs[layer_idx].clear()

        for event in self.prefill_signature_events[layer_idx]:
            event.synchronize()
        self.prefill_signature_events[layer_idx].clear()
        self.prefill_signature_refs[layer_idx].clear()

    def sync(self, layer_idx, start_bdx):
        self._sync_prefill_layer(layer_idx)

    def prepare_cache(self):
        for events in self.prefill_copy_events.values():
            for event in events:
                event.synchronize()
        for events in self.prefill_copy_events.values():
            events.clear()
        for refs in self.prefill_copy_refs.values():
            refs.clear()
        for events in self.prefill_signature_events.values():
            for event in events:
                event.synchronize()
        for events in self.prefill_signature_events.values():
            events.clear()
        for refs in self.prefill_signature_refs.values():
            refs.clear()
        for events in self.spill_events.values():
            for event in events:
                event.synchronize()
        self.use_static_prompt_retrieval = self._can_use_static_prompt_retrieval()
        self._assert_fast_only_lockstep_supported()
        assert self.use_static_prompt_retrieval, (
            "CometKV fast-only requires static prompt retrieval"
        )
        self._allocate_token_cache()
        if self.range_starts_buffer is None:
            self._allocate_decode_buffers()
        self._initialize_static_prompt_retrieval()
        self.attn_func = self.sparse_attention

    def _regions_from_lengths(self, prompt_length: int, visible_length: int):
        sink_end = min(prompt_length, self.static_pattern_start)
        recent_start = max(sink_end, visible_length - self.static_pattern_end)

        retrieval_ranges = []
        if recent_start > sink_end:
            retrieval_ranges.append((sink_end, recent_start))

        return {
            "sink_range": (0, sink_end),
            "recent_range": (recent_start, visible_length),
            "retrieval_ranges": retrieval_ranges,
        }

    def get_decode_regions(self, batch_idx: int):
        prompt_length = int(self.prompt_lengths[batch_idx].item())
        visible_length = int(self.visible_lengths[batch_idx].item())
        return self._regions_from_lengths(prompt_length, visible_length)

    def _compute_topk(self, retrieval_length: int, preserved_length: int = 0, visible_length: int | None = None) -> int:
        if retrieval_length <= 0:
            return 0
        if self.sig_topk > 0:
            # Clamp by retrieval_length: the sig_min_retrieval_topk floor must never request
            # more than the available candidates, otherwise the surplus selected slots stay -1
            # and the gather zero-fills them, diluting the softmax (no key-side mask is applied).
            k = min(max(min(self.sig_topk, retrieval_length), self.sig_min_retrieval_topk), retrieval_length)
            return min(k, self.sig_max_retrieval_topk) if self.sig_max_retrieval_topk else k

        if visible_length is None:
            visible_length = retrieval_length + preserved_length
        budget_preserved_length = 0 if self.exclude_preserved_from_budget else int(preserved_length)
        sparse_budget = (int(visible_length * self.retrieval_budget)
                         + self.head_extra_tokens - budget_preserved_length)
        k = min(max(min(max(sparse_budget, 0), retrieval_length), self.sig_min_retrieval_topk), retrieval_length)
        return min(k, self.sig_max_retrieval_topk) if self.sig_max_retrieval_topk else k

    def _compute_topk_for_lengths(self, retrieval_lengths, preserved_lengths, visible_lengths):
        if self.sig_topk > 0:
            topk_batch = torch.minimum(
                torch.full_like(retrieval_lengths, self.sig_topk, dtype=torch.int32),
                retrieval_lengths.to(torch.int32),
            )
            topk_batch = torch.maximum(
                topk_batch,
                torch.full_like(topk_batch, self.sig_min_retrieval_topk),
            )
            # Re-clamp by candidate count so the min-retrieval floor cannot exceed retrieval_len
            # (surplus slots would stay -1 and be zero-filled into the attended window).
            topk_batch = torch.minimum(topk_batch, retrieval_lengths.to(torch.int32))
            if self.sig_max_retrieval_topk:
                topk_batch.clamp_(max=self.sig_max_retrieval_topk)
            return torch.where(retrieval_lengths > 0, topk_batch, torch.zeros_like(topk_batch))

        total_budget = (visible_lengths.to(torch.float32) * self.retrieval_budget).to(torch.int32)
        sparse_budget = total_budget + self.head_extra_tokens
        if not self.exclude_preserved_from_budget:
            sparse_budget = sparse_budget - preserved_lengths.to(torch.int32)
        topk_batch = torch.minimum(
            torch.clamp(sparse_budget, min=0),
            retrieval_lengths.to(torch.int32),
        )
        topk_batch = torch.maximum(
            topk_batch,
            torch.full_like(topk_batch, self.sig_min_retrieval_topk),
        )
        # Re-clamp by candidate count so the min-retrieval floor cannot exceed retrieval_len
        # (surplus slots would stay -1 and be zero-filled into the attended window).
        topk_batch = torch.minimum(topk_batch, retrieval_lengths.to(torch.int32))
        if self.sig_max_retrieval_topk:
            topk_batch.clamp_(max=self.sig_max_retrieval_topk)
        return torch.where(retrieval_lengths > 0, topk_batch, torch.zeros_like(topk_batch))

    def _asym_scores_for(self, layer_idx, device):
        buf = self._asym_scores_buffers.get(str(device))
        if buf is None:
            tokens = self.signature_index[layer_idx].size(2)
            buf = torch.full(
                (self.batch_groups, tokens), float("-inf"),
                dtype=torch.float32, device=device,
            )
            # -inf filled once: decode retrieval ranges only ever grow, and every candidate slot is
            # overwritten by the scoring kernel each step, so non-candidate slots stay -inf forever.
            self._asym_scores_buffers[str(device)] = buf
        return buf

    def _sample_state_for(self, layer_idx, device):
        # Exact-size sampling buffers keyed by device, rebuilt when the plan's m changes (plan
        # changes only happen on eager init/slide steps, so rebuilds never occur inside a graph).
        key = str(device)
        m = self.active_sample_len_host
        st = self._sample_state.get(key)
        if st is None or st["m"] != m:
            rows = self.batch_groups
            tokens = self.signature_index[layer_idx].size(2)
            ch = self._sample_chunk
            tokens_pad = (tokens + ch - 1) // ch * ch
            st = {
                "m": m,
                "owner": -1,   # first layer on this device claims the per-step draw/corr compute
                # padded to the sampling chunk; the pad stays -inf forever (exp -> 0 mass)
                "prop": torch.full((rows, tokens_pad), float("-inf"),
                                   dtype=torch.float32, device=device),
                "noise": torch.zeros(((self.layer_num + self.sample_stride - 1) // self.sample_stride,
                                      rows, m), dtype=torch.float32, device=device),
                "first_layer": min(l for l in range(self.layer_num)
                                   if str(self.layer_mapping[str(l)]) == key),
                "draws": torch.zeros((rows, m), dtype=torch.int32, device=device),
                "corr": torch.zeros((rows, m), dtype=torch.float32, device=device),
                "tail_k": torch.zeros((rows, m, self.head_dim), dtype=self.dtype, device=device),
                "tail_v": torch.zeros((rows, m, self.head_dim), dtype=self.dtype, device=device),
                "tail_hit": torch.zeros((rows, m), dtype=torch.int32, device=device),
                # Batched-window tail path (bf16 CPU store): the resample layer launches ONE
                # gather covering the whole window's layers on the side stream (overlaps the
                # main compute); per-layer consumers slice wk/wv.
                "wk": torch.zeros((self.sample_stride, rows, m, self.head_dim),
                                  dtype=self.dtype, device=device),
                "wv": torch.zeros((self.sample_stride, rows, m, self.head_dim),
                                  dtype=self.dtype, device=device),
                "win_base": -1,
                "win_n": 0,
                "pending_ev": None,
                "stream": torch.cuda.Stream(device=device),
            }
            ptr_list = [(self.cpu_kv_cache[l].data_ptr()
                         if (str(self.layer_mapping[str(l)]) == key
                             and not self.kv_on_gpu and not self.quantize_cpu_kv) else 0)
                        for l in range(self.layer_num)]
            st["kv_ptrs"] = torch.tensor(ptr_list, dtype=torch.int64, device=device)
            # Host-precomputed (capture-safe: no device sync inside the graph): can the window
            # starting at layer L use the batched path?
            st["win_ok"] = [
                all(ptr_list[l] != 0
                    for l in range(L, min(L + self.sample_stride, self.layer_num)))
                for L in range(self.layer_num)
            ]
            # Per-window event pairs: events may be recorded once per capture position, so each
            # resample window gets its own pair (mirrors the _pf per-layer event convention).
            n_windows = (self.layer_num + self.sample_stride - 1) // self.sample_stride
            st["ev_start"] = [torch.cuda.Event() for _ in range(n_windows)]
            st["ev_done"] = [torch.cuda.Event() for _ in range(n_windows)]
            # Per-layer pipelined window fetch (COMETKV_TAIL_PIPELINE=1): one event per layer, so a
            # layer waits only for its own slice while later layers' slices stream under compute.
            st["ev_layer"] = [torch.cuda.Event() for _ in range(self.layer_num)]
            gen = self._sample_gen.get(key)
            if gen is None:
                gen = torch.Generator(device=device)
                gen.manual_seed(self.sample_seed)
                self._sample_gen[key] = gen
            self._sample_state[key] = st
        return st

    def _grouped_score_state_for(self, device, group):
        key = (str(device), group)
        state = self._grouped_score_buffers.get(key)
        if state is None:
            tokens = self.signature_index[0].size(2)
            state = {
                "logits": torch.full((self.batch_groups, group, tokens), float("-inf"),
                                     dtype=torch.float32, device=device),
                "chunk_lse": torch.empty((self.batch_groups, group, (tokens + 255) // 256),
                                         dtype=torch.float32, device=device),
                "head_lse": torch.empty((self.batch_groups, group), dtype=torch.float32, device=device),
            }
            self._grouped_score_buffers[key] = state
        return state

    def _gather_sample_tail(self, layer_idx, st):
        # Fetch the m sampled tokens' KV through the SAME gather kernels as the head (UVA + token
        # cache / GPU-resident), writing into the dedicated tail buffers: static_len=0/recent_len=0
        # so only the sparse section is produced. Tail is an additional independent quota;
        # its gather traffic must be reported separately from the deterministic head.
        rows = self.batch_groups
        dim = self.head_dim
        recent_cap = self._decode_hot_capacity()
        m = st["m"]
        gather_args = (
            self.static_keys[layer_idx].view(rows, self.static_pattern_total, dim),
            self.static_values[layer_idx].view(rows, self.static_pattern_total, dim),
            self.decode_hot_keys[layer_idx].view(rows, recent_cap, dim),
            self.decode_hot_values[layer_idx].view(rows, recent_cap, dim),
            st["draws"], self.cpu_kv_cache[layer_idx],
        )
        if not self.kv_on_gpu and not self.quantize_cpu_kv:
            # bf16 pinned store: cache-bypass UVA read. Sampled tokens are one-shot draws, so
            # routing them through the token cache pollutes the head's working set and pays
            # lock/insert traffic for nothing.
            uva_gather_kv_rows(
                st["draws"], self.cpu_kv_cache[layer_idx], st["tail_k"], st["tail_v"], m,
            )
            return
        tk = st["tail_k"].view(rows, m, 1, dim)
        tv = st["tail_v"].view(rows, m, 1, dim)
        if self.kv_on_gpu:
            if self.quantize_cpu_kv:
                concat_static_recent_gpu_gather_int8(
                    *gather_args, self.cpu_kv_k_scale[layer_idx], self.cpu_kv_v_scale[layer_idx],
                    tk, tv, 0, 0, m,
                )
            else:
                concat_static_recent_gpu_gather(*gather_args, tk, tv, 0, 0, m)
        elif self.quantize_cpu_kv:
            concat_static_recent_lookup_gather_uva_kv_update_cache_int8(
                *gather_args, self.cpu_kv_k_scale[layer_idx], self.cpu_kv_v_scale[layer_idx],
                tk, tv, st["tail_hit"], self.token_cache_ids[layer_idx],
                self.token_cache_locks[layer_idx], self.token_cache_keys[layer_idx],
                self.token_cache_values[layer_idx],
                0, 0, m,
            )
        else:
            stamps, step_dev, ways = self._token_cache_lru_args(layer_idx)
            prio_buckets, prio_scores = self._token_cache_prio_args(layer_idx)
            concat_static_recent_lookup_gather_uva_kv_update_cache(
                *gather_args, tk, tv, st["tail_hit"], self.token_cache_ids[layer_idx],
                self.token_cache_locks[layer_idx], self.token_cache_keys[layer_idx],
                self.token_cache_values[layer_idx],
                0, 0, m,
                stamps, step_dev, ways,
                prio_buckets, prio_scores,
                None,
            )

    @staticmethod
    def _duplicate_leader_slots(draws):
        """[rows, m] int32 -> for each slot, a slot holding the same token (itself for one slot per token)."""
        srt, perm = draws.sort(dim=1, stable=True)
        first = torch.ones_like(srt, dtype=torch.bool)
        first[:, 1:] = srt[:, 1:] != srt[:, :-1]
        pos = torch.arange(srt.size(1), device=draws.device).expand_as(srt)
        lead_pos = torch.where(first, pos, torch.zeros_like(pos)).cummax(dim=1).values
        slots = torch.empty_like(draws)
        slots.scatter_(1, perm, torch.gather(perm, 1, lead_pos).to(draws.dtype))
        return slots

    def _tail_window_gather(self, draws, kv_ptrs, out_k, out_v, m, total_tokens, src_slot=None):
        if self.tail_gather_blocks > 0:
            extra = () if src_slot is None else (src_slot,)   # older bindings take no src_slot
            uva_gather_kv_rows_window_persist(draws, kv_ptrs, out_k, out_v, m, total_tokens,
                                              self.tail_gather_blocks, *extra)
        else:
            uva_gather_kv_rows_window(draws, kv_ptrs, out_k, out_v, m, total_tokens)

    def _tail_traffic_diag(self, path, layer_idx, st, kh, device):
        d = _tail_diag(path, self.layer_num, device)
        draws = st["draws"].long()                                          # [rows, m]
        srt = draws.sort(dim=1).values
        first = torch.ones_like(srt, dtype=torch.bool)
        first[:, 1:] = srt[:, 1:] != srt[:, :-1]                            # one flag per distinct token
        head = self.selected_indices_buffer[:, :kh].long()
        in_head = (draws.unsqueeze(-1) == head.unsqueeze(1)).any(-1)
        u_head = (srt.unsqueeze(-1) == head.unsqueeze(1)).any(-1)
        acc = d["acc"][:, layer_idx]
        acc[0] += draws.numel()
        acc[1] += first.sum()
        acc[2] += in_head.sum()
        acc[3] += (first & ~u_head).sum()
        key = (str(device), layer_idx)
        prev = d["prev"].get(key)
        if prev is not None and prev.shape == srt.shape:
            acc[4] += (first & (srt.unsqueeze(-1) == prev.unsqueeze(1)).any(-1)).sum()
        d["prev"][key] = srt.clone()
        ids = getattr(self, "token_cache_ids", None)
        if ids is not None:
            cache = ids[layer_idx].long()
            acc[5] += (first & (srt.unsqueeze(-1) == cache.unsqueeze(1)).any(-1)).sum()

    def _sampled_tail_attention(self, queries, layer_idx, out_main, lse_main):
        """Importance-corrected micro-attention over m sampled tail tokens, merged IN PLACE.

        Samples m tokens (with replacement) from a full-support candidate proposal
        (recomputed every sample_stride layers), gathers their KV via the
        cache-bypass UVA kernel, and runs the fused tail-attention+LSE-merge kernel with the
        per-slot logit correction -log(m * q_j). The fused merge masks samples belonging to
        the current layer's head without renormalizing their proposal probabilities.
        All shapes are plan-fixed ->
        CUDA-graph capturable; the noise buffer is driver-refreshed per replay
        (cg_set_step_inputs), the eager path refreshes it at each resample layer.
        """
        m = self.active_sample_len_host
        kh = self.active_sparse_len_host
        device = queries.device
        st = self._sample_state_for(layer_idx, device)
        # Keep the owner's head in the proposal: another layer can need those candidates.
        # Draw correction always refers to this actual shared distribution, before masking.
        if st["owner"] < 0 or layer_idx % self.sample_stride == 0:
            st["owner"] = layer_idx
            if not self.use_cuda_graph and layer_idx == st["first_layer"]:
                # One independent noise slice per resampling window. Refresh once
                # per decode step in both eager and CG paths, so RNG consumption
                # and the samples remain identical across capture/replay.
                st["noise"].uniform_(generator=self._sample_gen[str(device)])
            scores = self._asym_scores_for(layer_idx, device)
            # Non-candidate slots are -inf by buffer invariant; proposal transformations
            # leave the shared score buffer intact for token-cache priority reads.
            prop = st["prop"]
            tokens = scores.size(1)
            if self.query_aggregation == "mean_prob" and not self.sample_autoscale:
                # Scores are already log mixture probabilities, not arbitrary
                # projection-space magnitudes. Temperature acts in log space.
                torch.mul(scores, 1.0 / self.sample_tau, out=prop[:, :tokens])
            elif self.sample_autoscale:
                # Selector-agnostic proposal: z-score over the finite (candidate) slots, then
                # scale to sigma/tau nats. -inf non-candidates stay -inf through the affine map.
                finite = torch.isfinite(scores)
                cnt = finite.sum(dim=1, keepdim=True).clamp_min(1)
                sz = torch.where(finite, scores, torch.zeros_like(scores))
                mean = sz.sum(dim=1, keepdim=True) / cnt
                var = (torch.where(finite, scores - mean, torch.zeros_like(scores)) ** 2
                       ).sum(dim=1, keepdim=True) / cnt
                scale = (self.sample_sigma / self.sample_tau) / var.sqrt().clamp_min(1e-6)
                torch.mul(scores - mean, scale, out=prop[:, :tokens])
            else:
                beta = 1.0 / (self.group_size * math.sqrt(self.head_dim) * self.sample_tau)
                torch.mul(scores, beta, out=prop[:, :tokens])
            if self.sample_exclude_head and self.active_sparse_len_host:
                # Owner head = this layer's top-k (selected before the tail merge); -1 = padding.
                sel = self.selected_indices_buffer[:, :self.active_sparse_len_host].long()
                idx = sel.clamp_min(0)
                cur = torch.gather(prop, 1, idx)
                prop.scatter_(1, idx, torch.where(sel >= 0, torch.full_like(cur, float("-inf")), cur))
            # Two-level (chunked) inverse-CDF sampling WITH replacement on the UNNORMALIZED
            # weights: torch's full-width cumsum is a slow path for few long rows (0.131 ms on
            # [8, 117k]); one 512-chunk reduction + tiny cumsums gives the same distribution.
            rows = self.batch_groups
            m_draws = st["draws"].size(1)
            ch = self._sample_chunk
            mx = prop.amax(dim=1, keepdim=True)
            w = (prop - mx).exp()                                    # [rows, tokens_pad]
            # Normalize then mix uniform mass over the real candidate range only. The
            # padded/non-candidate positions remain zero; all candidates have positive mass.
            w.div_(w.sum(dim=1, keepdim=True)).mul_(1.0 - self.sample_uniform_mix)
            w[:, self.plan_range1_start_host:self.plan_range1_end_host].add_(
                self.sample_uniform_mix / self.active_candidates_host)
            wc = w.view(rows, -1, ch)
            chunk_sums = wc.sum(dim=2)                               # [rows, n_chunks]
            cdf_c = torch.cumsum(chunk_sums, dim=1)
            z = cdf_c[:, -1:].clamp_min(1e-30)
            vals = st["noise"][layer_idx // self.sample_stride] * z
            cidx = torch.searchsorted(cdf_c, vals, right=True).clamp_(max=cdf_c.size(1) - 1)
            prev = torch.gather(
                torch.nn.functional.pad(cdf_c, (1, 0)), 1, cidx)    # cdf mass before the chunk
            local = torch.gather(
                wc, 1, cidx.unsqueeze(-1).expand(rows, m_draws, ch))
            local_cdf = torch.cumsum(local, dim=2)                   # [rows, m, ch] tiny
            off = torch.searchsorted(
                local_cdf, (vals - prev).unsqueeze(-1), right=True).squeeze(-1).clamp_(max=ch - 1)
            draws = (cidx * ch + off).clamp_(
                min=self.plan_range1_start_host, max=self.plan_range1_end_host - 1)
            q_sel = (torch.gather(w, 1, draws) / z).clamp_min(1e-30)
            # Truncated-IS clip is applied INSIDE the fused kernel on the full logits
            # (s + corr, per (row,head), capped at block mean + clip): corr-only clipping
            # measured 5 points worse on fwe-32k because the heavy tail is joint.
            st["corr"].copy_(-(math.log(m) + torch.log(q_sel)))      # [rows, m] fp32
            st["draws"].copy_(draws.to(torch.int32))
            src_slot = self._duplicate_leader_slots(st["draws"]) if self.tail_dedup else None
            # Batched window tail gather: one launch fetches the K/V of every layer in this
            # resample window (draws are shared), forked onto the side stream so the PCIe
            # transfer runs under the main compute; consumers below join via the event.
            n_win = min(self.sample_stride, self.layer_num - layer_idx)
            if st["win_ok"][layer_idx] and not self._tail_index_update_step:
                widx = layer_idx // self.sample_stride
                ev_start, ev_done = st["ev_start"][widx], st["ev_done"][widx]
                st["win_base"] = layer_idx
                st["win_n"] = n_win
                ev_start.record()
                st["stream"].wait_event(ev_start)
                if self.tail_pipeline:
                    # Layer by layer, owner first: the owner waits for 1/n_win of the window, the
                    # remaining layers' transfers overlap the owner's and later layers' compute.
                    with torch.cuda.stream(st["stream"]):
                        for i in range(n_win):
                            self._tail_window_gather(
                                st["draws"], st["kv_ptrs"][layer_idx + i:layer_idx + i + 1],
                                st["wk"][i:i + 1], st["wv"][i:i + 1],
                                m, self.cpu_kv_cache[layer_idx + i].size(1), src_slot,
                            )
                            st["ev_layer"][layer_idx + i].record(st["stream"])
                    st["pending_ev"] = "per_layer"
                else:
                    with torch.cuda.stream(st["stream"]):
                        self._tail_window_gather(
                            st["draws"], st["kv_ptrs"][layer_idx:layer_idx + n_win],
                            st["wk"][:n_win], st["wv"][:n_win],
                            m, self.cpu_kv_cache[layer_idx].size(1), src_slot,
                        )
                    ev_done.record(st["stream"])
                    st["pending_ev"] = ev_done
            else:
                # On an index-update step, later layers have not published their
                # newly evicted CPU keys yet. Fetch each layer only after its own
                # update instead of prefetching uninitialized future-layer KV.
                st["win_base"] = -1
        if st["win_base"] >= 0 and st["win_base"] <= layer_idx < st["win_base"] + st["win_n"]:
            if st["pending_ev"] == "per_layer":
                torch.cuda.current_stream().wait_event(st["ev_layer"][layer_idx])
            elif st["pending_ev"] is not None:
                torch.cuda.current_stream().wait_event(st["pending_ev"])
                st["pending_ev"] = None
            slot = layer_idx - st["win_base"]
            tail_k, tail_v = st["wk"][slot], st["wv"][slot]
        else:
            self._gather_sample_tail(layer_idx, st)
            tail_k, tail_v = st["tail_k"], st["tail_v"]
        diag_out = os.environ.get("COMETKV_TAIL_DIAG_OUT")
        if diag_out:
            self._tail_traffic_diag(diag_out, layer_idx, st, kh, device)
        # Fused micro-attention + LSE merge: one kernel computes the importance-corrected
        # softmax over the m tail tokens and merges it IN PLACE into the flash output.
        qg = queries.reshape(self.batch_groups, self.group_size, self.head_dim)
        sampled_tail_attention_merge(
            qg, tail_k, tail_v, st["corr"],
            out_main.reshape(self.batch_groups, self.group_size, self.head_dim),
            lse_main.reshape(self.batch_groups, self.group_size),
            1.0 / math.sqrt(self.head_dim),
            self.sample_clip,
            st["draws"], self.selected_indices_buffer, kh,
        )

    def _grouped_query_asym_topk(self, queries, layer_idx):
        # Continuous asymmetric scores over the packed 16B/token signature index (float query
        # projection x sign-bit keys, norm-weighted), then exact top-k over the active range.
        # No host syncs, fixed shapes -> CUDA-graph capturable (range/k are host plan constants).
        device = queries.device
        bsz, _, n_heads, dim = queries.shape
        group = n_heads // self.kv_head
        rows = self.batch_groups
        start = self.plan_range1_start_host
        end = self.plan_range1_end_host
        max_topk = self.selected_indices_buffer.size(1)
        k = max(0, min(int(self.active_sparse_len_host or 0), max_topk, end - start))
        if self._asym_plan_dirty:
            # k and the -1 padding are regime constants (ranges only grow, k never shrinks within a
            # plan): commit them once per plan commit instead of per step per layer.
            self.selected_indices_buffer.fill_(-1)
            self.sparse_lengths_buffer.fill_(k)
            self._asym_plan_dirty = False
        if k > 0 or self.active_sample_len_host > 0:
            P = self._projection_for(device)                              # [sig_bits, dim] fp32
            bits_used = self.asym_sig_bits
            keys = self.signature_index[layer_idx].view(rows, -1, self.sig_bytes)
            scores = self._asym_scores_for(layer_idx, device)
            normalize_heads = self.query_aggregation == "mean_prob"
            use_lookup = self.signature_score_impl == "lookup" or (
                self.signature_score_impl == "auto" and self.block_stats and not normalize_heads
            )
            grouped_queries = queries.reshape(rows, group, dim)
            qg = (grouped_queries.float() if normalize_heads else
                  grouped_queries.sum(dim=1, keepdim=True, dtype=torch.float32))
            q_proj_in = qg - self.q_mean[layer_idx].unsqueeze(1) if self.qcenter != "off" else qg
            x = torch.matmul(q_proj_in, P.t()).contiguous()
            if self.scorer_mode == "sym":
                # Equal-norm sign/sign scoring: dot product is bits - 2 * Hamming distance.
                x = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
            x_total = x[:, :, :bits_used].sum(dim=-1).contiguous()
            if self.scorer_mode != "sym" and not self.block_stats and not normalize_heads and not use_lookup:
                # Explicit frozen/q_sum ablation preserves the previous kernel.
                asym_signature_score_into(
                    x[:, 0], x_total[:, 0].contiguous(), keys,
                    self.range_starts_buffer, self.range_ends_buffer, scores,
                    bits_used, int(self.active_candidates_host),
                    self.sig_norm_lo[layer_idx], self.sig_norm_step[layer_idx],
                )
            else:
                state = self._grouped_score_state_for(device, qg.size(1))
                center_bias = torch.bmm(qg, self.sig_block_centers[layer_idx].transpose(1, 2))
                residual_scale = self.RSQRT_DIM / self.signature_score_scale if normalize_heads else 1.0
                bias_scale = self.RSQRT_DIM if normalize_heads else self.signature_score_scale
                norm_lo = self.sig_block_norm_lo[layer_idx]
                norm_step = self.sig_block_norm_step[layer_idx]
                if self.scorer_mode == "sym":
                    # Keep GQA probability aggregation identical across the component ladder.
                    norm_lo = torch.zeros_like(norm_lo)
                    norm_step = torch.zeros_like(norm_step)
                    center_bias = torch.zeros_like(center_bias)
                grouped_signature_score_into(
                    x, x_total, keys, self.range_starts_buffer, self.range_ends_buffer,
                    norm_lo, norm_step, center_bias,
                    state["logits"], state["chunk_lse"], state["head_lse"], scores,
                    bits_used, int(self.active_candidates_host),
                    int(self.lockstep_prompt_length_host) if self.block_stats else keys.size(1),
                    end, self._fixed_prompt_local_slide_stride(), self._fixed_prompt_local_recent_overlap(),
                    residual_scale, bias_scale, normalize_heads, use_lookup,
                    *((self.sig_block_qbias_lo[layer_idx], self.sig_block_qbias_step[layer_idx],
                       self.asym_sig_bits // 8) if self.qbias_in_signature else ()),
                )
            if self.q_bias is not None:
                # fp32-buffer path (fp mode / frozen statistics): scores are on the
                # signature_score_scale * (q . k) scale; add the exact mu_q . r term. Non-candidate
                # slots stay -inf; candidate slots were rewritten by the kernel above.
                scores.add_(self.q_bias[layer_idx], alpha=self.signature_score_scale)
            # The torch fallback uses sorted=False: indices are consumed as a set.
            # sorted=True would trip torch's k>4096 sort_outf fallback
            # (cub segmented sort + gather + 2 copies, ~185us/layer at 96k budgets).
            if k > 0:
                use_radix = (self.exact_topk_impl != "torch" and k <= 10240  # native kSelectMaxK
                             and scores.device == self.selected_indices_buffer.device
                             and (self.exact_topk_impl == "radix"
                                  or (end - start >= 4096 and 4 * k <= end - start)))
                if use_radix:
                    key = str(device)
                    workspace = self._exact_topk_workspace.get(key)
                    if workspace is None or workspace[1].shape != scores.shape:
                        workspace = (
                            torch.empty((rows, (scores.size(1) + 1023) // 1024, 288),
                                        dtype=torch.int32, device=device),
                            torch.empty_like(scores, dtype=torch.int32),
                            torch.empty((rows, 7), dtype=torch.int32, device=device),
                        )
                        self._exact_topk_workspace[key] = workspace
                    exact_topk_indices_into(scores, self.selected_indices_buffer, *workspace, k, start, end)
                else:
                    sel = torch.topk(scores, k, dim=1, sorted=False).indices
                    self.selected_indices_buffer[:, :k].copy_(sel.to(torch.int32))

        if layer_idx in self.recall_snapshots:
            query_snapshot, selected_snapshot = self.recall_snapshots[layer_idx]
            query_snapshot.copy_(queries)
            selected_snapshot.copy_(self.selected_indices_buffer)

    def _plan_tail_size(self, budget_min: int, budget_max: int, visible_length: int) -> int:
        """Tail draws m for a plan whose length-based budgets K_t span [budget_min, budget_max]."""
        if (self.sample_mode == "off" or visible_length < self.sample_min_context
                or self.active_candidates_host <= budget_max):
            return 0  # no tail requested, context below threshold, or budget covers every candidate
        if self.sample_mode == "size":
            m = min(self.sample_size, self.sample_max_m)
        else:
            # floor(frac * K_t) over the smallest row budget: the head keeps >= (1 - frac) of K_t.
            m = min(int(self.sample_frac * budget_min), self.sample_max_m)
            if m < self.sample_min_m:
                m = 0
        if self.tail_in_budget and m >= budget_min:
            return 0  # K_t cannot hold both a non-empty head and the tail
        return m

    def _update_retrieval_plan(self, visible_lengths):
        prompt_lengths = self.prompt_lengths
        sink_ends = self._sink_ends()

        if self.use_static_prompt_retrieval:
            prompt_retrieval_end = torch.clamp(
                prompt_lengths - self.static_pattern_end,
                min=0,
            )
            range1_start = sink_ends
            decode_tokens = torch.clamp(visible_lengths - prompt_lengths, min=0)
            capacity = self._fixed_prompt_local_recent_capacity()
            block_size = self._fixed_prompt_local_slide_stride()
            overlap = self._fixed_prompt_local_recent_overlap()
            overflow = torch.clamp(decode_tokens - block_size - 1, min=0)
            window_start = torch.where(
                decode_tokens > block_size,
                (torch.div(overflow, block_size, rounding_mode="floor") + 1) * block_size - overlap,
                torch.zeros_like(decode_tokens),
            )
            # Eviction frontier is schedule-deterministic: once the lockstep window has slid,
            # exactly `window_start` decode tokens have been spilled to cpu_kv (positions
            # [prompt_length, prompt_length + window_start)), so they re-enter retrieval range1.
            # Derived from window_start here (NOT the per-step-lagged lockstep_evicted_retrieval_end_host)
            # so the plan is identical for every layer when committed at layer 0 of a slide step;
            # otherwise layers 0..L-2 would attend a stale range1 and drop the just-evicted band.
            evicted_end = torch.where(
                decode_tokens > block_size,
                prompt_lengths + window_start,
                torch.zeros_like(prompt_lengths),
            )
            range1_end = torch.maximum(
                torch.maximum(prompt_retrieval_end, evicted_end),
                range1_start,
            )
            local_len = torch.clamp(decode_tokens - window_start, min=0, max=capacity)
            static_len = torch.where(
                decode_tokens > block_size,
                sink_ends,
                torch.full_like(sink_ends, self.static_pattern_total),
            )
            preserved_len = static_len + local_len
        else:
            recent_starts = self._recent_starts_for_visible(visible_lengths)
            range1_start = sink_ends
            range1_end = torch.maximum(recent_starts, range1_start)
            recent_len = (visible_lengths - recent_starts).clamp(min=0)
            preserved_len = sink_ends + recent_len

        range2_start = torch.zeros_like(prompt_lengths)
        range2_end = torch.zeros_like(prompt_lengths)

        retrieval_len = (range1_end - range1_start).clamp(min=0)
        range2_len = (range2_end - range2_start).clamp(min=0)
        # Length-based budget K_t per row (head capacity before any tail accounting).
        topk_batch = self._compute_topk_for_lengths(retrieval_len, preserved_len, visible_lengths)
        self.budget_plan_visible_length_host = int(visible_lengths[0].item())
        # Max candidate count across rows (range1 + range2; range2 is empty in decode). Lets the topk
        # kernels launch only the chunks covering real candidates instead of the full padded signature
        # width. Updated only here (slide steps / init), reused by every step's topk launch.
        self.active_candidates_host = int((retrieval_len + range2_len).max().item())
        # Tail size m, committed only at init/slide boundaries so captured shapes stay fixed within
        # a window. With tail_in_budget the head gives m of its K_t slots to the tail.
        m = self._plan_tail_size(int(topk_batch.min().item()), int(topk_batch.max().item()),
                                 int(visible_lengths.max().item()))
        if self.tail_in_budget and m > 0:
            topk_batch = topk_batch - m
        self.requested_sparse_len_host = max(0, int(topk_batch.max().item()))
        if self.requested_sparse_len_host > self.selected_indices_buffer.size(1):
            raise RuntimeError("CometKV head budget exceeds preallocated generation capacity")
        self.active_sparse_len_host = self.requested_sparse_len_host
        self.active_sample_len_host = m
        self.active_retrieval_len_host = self.active_sparse_len_host + m

        self.range_starts_buffer[:, 0].copy_(range1_start.to(torch.int32).repeat_interleave(self.kv_head))
        self.range_ends_buffer[:, 0].copy_(range1_end.to(torch.int32).repeat_interleave(self.kv_head))
        self.range_starts_buffer[:, 1].copy_(range2_start.to(torch.int32).repeat_interleave(self.kv_head))
        self.range_ends_buffer[:, 1].copy_(range2_end.to(torch.int32).repeat_interleave(self.kv_head))
        self.topk_buffer.copy_(topk_batch.repeat_interleave(self.kv_head))
        # Host copies of the (lockstep-uniform) decode plan for selectors that run as plain torch ops:
        # committed only here (init / slide steps), so reading them per step never syncs the device and
        # they are bake-safe for CUDA-graph capture (plan changes always trigger an eager re-capture).
        self.plan_range1_start_host = int(range1_start[0].item())
        self.plan_range1_end_host = int(range1_end[0].item())
        self._asym_plan_dirty = True

    def _fixed_prompt_local_window_start(self, decode_count: int) -> int:
        block_size = self._fixed_prompt_local_slide_stride()
        if decode_count <= block_size:
            return 0
        overlap = self._fixed_prompt_local_recent_overlap()
        return ((int(decode_count) - block_size - 1) // block_size + 1) * block_size - overlap

    def _append_chunked_fixed_prompt_local_kv(self, key_states, value_states, current_positions, prompt_lengths, layer_idx):
        hot_keys = self.decode_hot_keys[layer_idx]
        hot_values = self.decode_hot_values[layer_idx]
        hot_token_ids = self.decode_hot_token_ids[layer_idx]

        for batch_idx in range(self.batch_size):
            absolute_pos = int(current_positions[batch_idx].item())
            prompt_length = int(prompt_lengths[batch_idx].item())
            decode_offset = max(absolute_pos - prompt_length, 0)
            old_start = self._fixed_prompt_local_window_start(decode_offset)
            new_start = self._fixed_prompt_local_window_start(decode_offset + 1)
            if new_start > old_start:
                shift = new_start - old_start
                prev_len = min(
                    decode_offset - old_start,
                    self._fixed_prompt_local_recent_capacity(),
                )
                keep = max(prev_len - shift, 0)
                if keep > 0:
                    hot_keys[batch_idx, :, :keep, :].copy_(
                        hot_keys[batch_idx, :, shift:shift + keep, :].clone()
                    )
                    hot_values[batch_idx, :, :keep, :].copy_(
                        hot_values[batch_idx, :, shift:shift + keep, :].clone()
                    )
                    hot_token_ids[batch_idx, :, :keep].copy_(
                        hot_token_ids[batch_idx, :, shift:shift + keep].clone()
                    )

            slot = decode_offset - new_start
            if slot < 0 or slot >= self._fixed_prompt_local_recent_capacity():
                raise RuntimeError(
                    f"Invalid fixed prompt local decode slot {slot} for offset {decode_offset}."
                )
            hot_keys[batch_idx, :, slot, :].copy_(key_states[batch_idx, 0])
            hot_values[batch_idx, :, slot, :].copy_(value_states[batch_idx, 0])
            hot_token_ids[batch_idx, :, slot].fill_(absolute_pos)

    def _needs_chunked_fixed_prompt_local_append(self, current_positions, prompt_lengths):
        if not (self.use_static_prompt_retrieval and self.use_chunked_fixed_prompt_local):
            return False
        decode_offsets = current_positions - prompt_lengths
        return bool(torch.any(decode_offsets >= self._fixed_prompt_local_recent_capacity()).item())

    def _lockstep_fixed_prompt_local_decode_offset(self):
        if not self.fixed_prompt_local_prompt_lengths_host:
            return None
        decode_offset = int(self.context) - int(self.fixed_prompt_local_prompt_lengths_host[0])
        if decode_offset < 0:
            return None
        return decode_offset

    def _single_batch_fixed_prompt_local_decode_offset(self):
        return self._lockstep_fixed_prompt_local_decode_offset()

    def _needs_chunked_fixed_prompt_local_append_for_context(self):
        decode_offset = self._lockstep_fixed_prompt_local_decode_offset()
        if decode_offset is None:
            return None
        return decode_offset >= self._fixed_prompt_local_recent_capacity()

    def _should_update_fixed_prompt_local_static_lengths(self, current_positions, next_positions, prompt_lengths):
        if not (self.use_static_prompt_retrieval and self.use_chunked_fixed_prompt_local):
            return False
        capacity = self._fixed_prompt_local_slide_stride()
        current_offsets = current_positions - prompt_lengths
        next_offsets = next_positions - prompt_lengths
        crossed_capacity = current_offsets <= capacity
        crossed_capacity &= next_offsets > capacity
        return bool(torch.any(crossed_capacity).item())

    def _should_update_fixed_prompt_local_static_lengths_for_context(self):
        decode_offset = self._lockstep_fixed_prompt_local_decode_offset()
        if decode_offset is None:
            return None
        capacity = self._fixed_prompt_local_slide_stride()
        return decode_offset <= capacity and decode_offset + 1 > capacity

    def _update_fixed_prompt_local_static_lengths(self, visible_lengths):
        decode_tokens = torch.clamp(visible_lengths - self.prompt_lengths, min=0)
        sink_lengths = self._sink_ends()
        prompt_local_limit = self._fixed_prompt_local_slide_stride()
        static_lengths = torch.where(
            decode_tokens > prompt_local_limit,
            sink_lengths,
            torch.full_like(sink_lengths, self.static_pattern_total),
        )
        self.static_lengths_buffer.copy_(static_lengths.to(torch.int32).repeat_interleave(self.kv_head))
        if self.fixed_prompt_local_prompt_lengths_host:
            decode_offset = int(self.context) + 1 - int(self.fixed_prompt_local_prompt_lengths_host[0])
            self.fixed_prompt_local_static_length_host = (
                self.static_pattern_start
                if decode_offset > prompt_local_limit
                else self.static_pattern_total
            )

    def _update_lockstep_evicted_retrieval(self, layer_idx: int, evict_start_step: int, evict_count: int):
        if evict_count <= 0:
            return
        token_start = self.lockstep_prompt_length_host + int(evict_start_step)
        token_end = token_start + int(evict_count)
        if token_end > self.retrieval_capacity:
            raise RuntimeError(
                f"CometKV lockstep retrieval capacity exceeded: {token_end} > {self.retrieval_capacity}"
            )

        local_keys = self.decode_hot_keys[layer_idx][:, :, :evict_count, :]
        local_values = self.decode_hot_values[layer_idx][:, :, :evict_count, :]
        key_rows = local_keys.reshape(self.batch_groups, evict_count, self.head_dim).contiguous()
        value_rows = local_values.reshape(self.batch_groups, evict_count, self.head_dim).contiguous()

        if self.quantize_cpu_kv:
            # Quantize the evicted decode tokens with the frozen per-channel K scale + per-token V.
            self._quantize_kv_into_cpu(layer_idx, 0, token_start, key_rows, value_rows)
        elif self.kv_on_gpu:
            self._interleave_kv_into_gpu_store(layer_idx, 0, token_start, key_rows, value_rows)
        else:
            copy_interleaved_kv_to_cpu(
                key_rows,
                value_rows,
                self.cpu_kv_cache[layer_idx],
                0,
                token_start,
            )

        block_idx = (1 + (int(evict_start_step) + self._fixed_prompt_local_recent_overlap())
                     // self._fixed_prompt_local_slide_stride()) if self.block_stats else 0
        packed = self._build_prefill_signatures(
            key_rows, layer_idx, row_start=0, freeze_norm_scale=False, block_idx=block_idx,
        )
        self.signature_index[layer_idx][:, :, token_start:token_end, :].copy_(
            packed.view(self.batch_size, self.kv_head, evict_count, self.sig_bytes)
        )
        if self.q_bias is not None:
            self.q_bias[layer_idx][:, token_start:token_end].copy_(self._last_qbias)

        self.sig_active_blocks[layer_idx] = max(self.sig_active_blocks[layer_idx], block_idx + 1)
        # Rebuild only AFTER this layer has published its own new keys. Other
        # layers have not necessarily written [token_start:token_end] yet.
        if self.full_recompute_interval > 0:
            self._recompute_decode_counter[layer_idx] += evict_count
            if self._recompute_decode_counter[layer_idx] >= self.full_recompute_interval:
                self._recompute_decode_counter[layer_idx] %= self.full_recompute_interval
                self._full_recompute_stats(layer_idx, token_end)

        if layer_idx == self.layer_num - 1:
            self.lockstep_evicted_retrieval_end_host = token_end

    def _profile_decode_update_call(self, profile, key: str, device, fn):
        if profile is None:
            return fn()
        if str(device).startswith("cuda"):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = fn()
            end_event.record()
            end_event.synchronize()
            profile[key] = profile.get(key, 0.0) + float(start_event.elapsed_time(end_event))
            return result
        start_time = time.perf_counter()
        result = fn()
        profile[key] = profile.get(key, 0.0) + (time.perf_counter() - start_time) * 1000.0
        return result

    def _cg_decode_append(self, key_states, value_states, layer_idx):
        # CUDA-graph decode: append the new token's KV to the lockstep local window using a DEVICE step
        # counter (so the captured kernel writes the advancing slot on every replay). No host bookkeeping
        # and no in-kernel visible-length advance here -- the replay driver advances host counters via
        # cg_advance_host_step() after the step, and sets _cg_decode_step_dev via cg_set_step_inputs().
        append_key_states = key_states if key_states.is_contiguous() else key_states.contiguous()
        append_value_states = value_states if value_states.is_contiguous() else value_states.contiguous()
        append_lockstep_local_kv_cache_dev(
            append_key_states,
            append_value_states,
            self.decode_hot_keys[layer_idx],
            self.decode_hot_values[layer_idx],
            self._cg_decode_step_dev,
            self._fixed_prompt_local_recent_capacity(),
            self._fixed_prompt_local_slide_stride(),
        )

    def _cg_live_seqlen(self):
        # Live attended length for flash cache_seqlens: static_len + sparse_len + recent_len, computed
        # purely from host counters (lockstep, same across layers). decode_count uses context+1 (the
        # token appended this step is included in the recent window), matching the eager path.
        prompt_length = int(self.lockstep_prompt_length_host)
        decode_count = int(self.context) + 1 - prompt_length
        window_start = self._fixed_prompt_local_window_start(decode_count)
        recent_len = min(max(decode_count - window_start, 0), self._fixed_prompt_local_recent_capacity())
        return int(self.fixed_prompt_local_static_length_host) + int(self.active_sparse_len_host) + recent_len

    def cg_set_step_inputs(self):
        # Driver calls this EAGERLY before replaying the captured decode step: set the device step counter
        # (append slot) and the flash cache_seqlens for the current step. No-op-safe outside the graph.
        self._tail_index_update_step = False  # updates themselves always run eager
        decode_step = int(self.context) - int(self.lockstep_prompt_length_host)
        self._cg_decode_step_dev.fill_(decode_step)
        self._cg_cache_seqlens.fill_(self._cg_live_seqlen())
        # Monotonic stamp for the LRU token cache: driver-set each replay so a captured graph still
        # sees an advancing step (the gather kernel reads it to timestamp touched/inserted slots).
        if self._token_cache_step_dev is not None:
            for step_dev in self._token_cache_step_dev.values():
                step_dev.fill_(decode_step)
        # Fresh sampling noise per replay: the captured graph consumes the buffer via
        # searchsorted, the driver re-randomizes it here (same pattern as the LRU stamp).
        if self.active_sample_len_host > 0:
            for key, st in self._sample_state.items():
                st["noise"].uniform_(generator=self._sample_gen[key])

    def cg_advance_host_step(self):
        # Driver calls this EAGERLY after replaying the captured decode step to advance host counters
        # (mirrors the eager last-layer bookkeeping). visible_lengths device tensor is kept in sync for
        # the next eager (window-slide) step.
        self.visible_lengths.add_(1)
        self.context += 1
        self.lockstep_decode_step_host = int(self.context) - int(self.lockstep_prompt_length_host)
        if self.fixed_prompt_local_layer_visible_lengths_host is not None:
            next_visible_host = int(self.context)
            for host_layer_idx in range(self.layer_num):
                self.fixed_prompt_local_layer_visible_lengths_host[host_layer_idx] = next_visible_host

    def cg_is_slide_step(self):
        # Host-only (no GPU sync) check: does the lockstep window slide on the NEXT decode step?
        step = int(self.context) - int(self.lockstep_prompt_length_host)
        return self._fixed_prompt_local_window_start(step + 1) > self._fixed_prompt_local_window_start(step)

    def decode_update_kv_cache(self, key_states, value_states, layer_idx):
        if self.use_cuda_graph:
            with self._cuda_device_guard(self.layer_mapping[str(layer_idx)]):
                self._cg_decode_append(key_states, value_states, layer_idx)
            return None, None
        with self._cuda_device_guard(self.layer_mapping[str(layer_idx)]):
            _event_t = self._event_profile_start()
            bsz, seq_len, group_num, _ = key_states.shape
            assert seq_len == 1, f"CometKV decode expects seq_len == 1, got {seq_len}."
            assert bsz == self.batch_size, f"CometKV decode expects batch_size == {self.batch_size}, got {bsz}."
            assert group_num == self.kv_head, f"CometKV decode expects kv_head == {self.kv_head}, got {group_num}."
            decode_update_profile = None
            if self.profile_decode_update:
                self.last_decode_update_profile = {
                    "hash_lockstep_append_ms": 0.0,
                    "hash_evicted_decode_kv_index_update_ms": 0.0,
                    "hash_lockstep_retrieval_plan_update_ms": 0.0,
                }
                decode_update_profile = self.last_decode_update_profile
            assert self.use_static_prompt_retrieval, (
                "CometKV fast-only decode_update_kv_cache requires prepare_cache() "
                "to enable static prompt retrieval before decode."
            )

            if key_states.device == self.visible_lengths.device:
                current_positions = self.visible_lengths
                prompt_lengths = self.prompt_lengths
            else:
                current_positions = self.visible_lengths.clone()
                prompt_lengths = self.prompt_lengths.clone()
            lockstep_decode_step_host = None
            lockstep_evict_count = 0
            advanced_visible_lengths_in_append = False

            lockstep_decode_step_host = max(
                int(self.context) - int(self.lockstep_prompt_length_host),
                int(self.lockstep_decode_step_host),
                0,
            )
            old_window_start = self._fixed_prompt_local_window_start(lockstep_decode_step_host)
            new_window_start = self._fixed_prompt_local_window_start(lockstep_decode_step_host + 1)
            evict_count = max(new_window_start - old_window_start, 0)
            has_future_decode = (lockstep_decode_step_host + 1) < max(self.max_new_length - 1, 0)
            lockstep_evict_count = evict_count if has_future_decode else 0
            if layer_idx == 0:
                self._tail_index_update_step = lockstep_evict_count > 0
            if lockstep_evict_count > 0:
                if decode_update_profile is None:
                    self._update_lockstep_evicted_retrieval(
                        layer_idx,
                        old_window_start,
                        lockstep_evict_count,
                    )
                else:
                    self._profile_decode_update_call(
                        decode_update_profile,
                        "hash_evicted_decode_kv_index_update_ms",
                        key_states.device,
                        lambda: self._update_lockstep_evicted_retrieval(
                            layer_idx,
                            old_window_start,
                            lockstep_evict_count,
                        ),
                    )
            append_key_states = key_states if key_states.is_contiguous() else key_states.contiguous()
            append_value_states = value_states if value_states.is_contiguous() else value_states.contiguous()
            advance_visible_lengths_in_append = (
                layer_idx == self.layer_num - 1
                and key_states.device == self.visible_lengths.device
            )
            if decode_update_profile is None:
                if advance_visible_lengths_in_append:
                    append_lockstep_local_kv_cache_and_advance(
                        append_key_states,
                        append_value_states,
                        self.decode_hot_keys[layer_idx],
                        self.decode_hot_values[layer_idx],
                        self.visible_lengths,
                        lockstep_decode_step_host,
                        self._fixed_prompt_local_recent_capacity(),
                        self._fixed_prompt_local_slide_stride(),
                    )
                    advanced_visible_lengths_in_append = True
                else:
                    append_lockstep_local_kv_cache(
                        append_key_states,
                        append_value_states,
                        self.decode_hot_keys[layer_idx],
                        self.decode_hot_values[layer_idx],
                        lockstep_decode_step_host,
                        self._fixed_prompt_local_recent_capacity(),
                        self._fixed_prompt_local_slide_stride(),
                    )
            else:
                if advance_visible_lengths_in_append:
                    self._profile_decode_update_call(
                        decode_update_profile,
                        "hash_lockstep_append_ms",
                        key_states.device,
                        lambda: append_lockstep_local_kv_cache_and_advance(
                            append_key_states,
                            append_value_states,
                            self.decode_hot_keys[layer_idx],
                            self.decode_hot_values[layer_idx],
                            self.visible_lengths,
                            lockstep_decode_step_host,
                            self._fixed_prompt_local_recent_capacity(),
                            self._fixed_prompt_local_slide_stride(),
                        ),
                    )
                    advanced_visible_lengths_in_append = True
                else:
                    self._profile_decode_update_call(
                        decode_update_profile,
                        "hash_lockstep_append_ms",
                        key_states.device,
                        lambda: append_lockstep_local_kv_cache(
                            append_key_states,
                            append_value_states,
                            self.decode_hot_keys[layer_idx],
                            self.decode_hot_values[layer_idx],
                            lockstep_decode_step_host,
                            self._fixed_prompt_local_recent_capacity(),
                            self._fixed_prompt_local_slide_stride(),
                        ),
                    )

            next_positions = None
            should_update_static_lengths = self._should_update_fixed_prompt_local_static_lengths_for_context()
            if should_update_static_lengths is None:
                next_positions = current_positions + 1
                should_update_static_lengths = self._should_update_fixed_prompt_local_static_lengths(
                    current_positions,
                    next_positions,
                    prompt_lengths,
                )
            if (
                layer_idx == 0
                and should_update_static_lengths
            ):
                if next_positions is None:
                    next_positions = (
                        self.visible_lengths
                        if advanced_visible_lengths_in_append
                        else current_positions + 1
                    )
                self._update_fixed_prompt_local_static_lengths(next_positions)
            if layer_idx == 0 and lockstep_evict_count > 0:
                # Commit the retrieval plan for the WHOLE step here at layer 0 (consistent with the
                # static-length flip above) using the post-step decode count, so EVERY layer attends
                # over the band the window slide just evicted. Previously the plan was committed at the
                # last layer, so layers 0..L-2 attended a stale range1 and dropped a contiguous
                # slide_stride-sized band of live tokens every slide step. range1_end is now
                # schedule-deterministic, so this layer-0 commit matches the per-layer recent window.
                # Use self.visible_lengths (primary device, not yet advanced at layer 0) + 1 so the
                # plan is computed against self.prompt_lengths on the same device.
                post_step_positions = self.visible_lengths + 1
                if decode_update_profile is None:
                    self._update_retrieval_plan(post_step_positions)
                else:
                    self._profile_decode_update_call(
                        decode_update_profile,
                        "hash_lockstep_retrieval_plan_update_ms",
                        key_states.device,
                        lambda: self._update_retrieval_plan(post_step_positions),
                    )
            if self.fixed_prompt_local_layer_visible_lengths_host is not None:
                self.fixed_prompt_local_layer_visible_lengths_host[layer_idx] = int(self.context) + 1

            if layer_idx == self.layer_num - 1:
                if advanced_visible_lengths_in_append:
                    next_positions = self.visible_lengths
                else:
                    if next_positions is None:
                        next_positions = current_positions + 1
                    self.visible_lengths.copy_(next_positions)
                if lockstep_decode_step_host is not None:
                    # Plan is now committed at layer 0 (above) so every layer in a slide step sees a
                    # consistent post-slide range1; only advance the lockstep step counter here.
                    self.lockstep_decode_step_host = lockstep_decode_step_host + 1
                if self.fixed_prompt_local_layer_visible_lengths_host is not None:
                    next_visible_host = int(self.context) + 1
                    for host_layer_idx in range(self.layer_num):
                        self.fixed_prompt_local_layer_visible_lengths_host[host_layer_idx] = next_visible_host
                self.context += 1

            self._event_profile_end("decode_update_kv_cache", _event_t)
        return None, None

    def _can_use_static_fixed_concat_flash_attention(self):
        return (
            self.use_static_prompt_retrieval
            and self.fixed_prompt_local_layer_visible_lengths_host is not None
            and self.fixed_prompt_local_prompt_lengths_host is not None
        )

    def _can_use_static_fixed_direct_concat_gather(self):
        return self._can_use_static_fixed_concat_flash_attention()

    def _static_fixed_concat_rows_view(self, storage, total_len):
        return torch.as_strided(
            storage,
            (self.batch_groups, total_len, 1, self.head_dim),
            (total_len * self.head_dim, self.head_dim, self.head_dim, 1),
        )

    def _static_fixed_concat_kvcache_view(self, storage, total_len):
        return torch.as_strided(
            storage,
            (self.batch_size, total_len, self.kv_head, self.head_dim),
            (
                self.kv_head * total_len * self.head_dim,
                self.head_dim,
                total_len * self.head_dim,
                1,
            ),
        )

    def _ensure_static_fixed_concat_flash_storage(self, layer_idx, total_len):
        if self.static_fixed_concat_flash_capacity[layer_idx] >= total_len:
            return
        capacity = max(
            total_len,
            self.static_pattern_total
            + self._fixed_prompt_local_recent_capacity()
            + self.concat_sparse_capacity_bound,
        )
        device = self.layer_mapping[str(layer_idx)]
        numel = self.batch_groups * capacity * self.head_dim
        self.static_fixed_concat_flash_key_storage[layer_idx] = torch.empty(numel, dtype=self.dtype, device=device)
        self.static_fixed_concat_flash_value_storage[layer_idx] = torch.empty(numel, dtype=self.dtype, device=device)
        self.static_fixed_concat_flash_capacity[layer_idx] = capacity

    def _lockstep_fixed_prompt_recent_length_host(self, layer_idx):
        if not self.fixed_prompt_local_layer_visible_lengths_host:
            return 0
        prompt_length = int(self.fixed_prompt_local_prompt_lengths_host[0])
        visible_length = int(self.fixed_prompt_local_layer_visible_lengths_host[layer_idx])
        decode_count = max(visible_length - prompt_length, 0)
        window_start = self._fixed_prompt_local_window_start(decode_count)
        return min(
            max(decode_count - window_start, 0),
            self._fixed_prompt_local_recent_capacity(),
        )

    def _single_batch_fixed_prompt_recent_length_host(self, layer_idx):
        return self._lockstep_fixed_prompt_recent_length_host(layer_idx)

    def _active_sparse_len_for_selected_indices(self, selected_indices):
        if selected_indices.size(1) == 0:
            return 0
        if (
            selected_indices is self.selected_indices_buffer
            and self.active_sparse_len_host is not None
        ):
            return max(0, min(int(self.active_sparse_len_host), selected_indices.size(1)))
        if self.sparse_lengths_buffer is None:
            return selected_indices.size(1)
        return max(
            0,
            min(
                int(self.sparse_lengths_buffer.max().item()),
                selected_indices.size(1),
            ),
        )

    def _uncached_gather(self, layer_idx, selected_indices, sparse_len):
        if self.quantize_cpu_kv or self.dtype != torch.bfloat16:
            raise ValueError("Cache-free UVA ablation currently requires unquantized BF16")
        indices = selected_indices[:, :sparse_len].contiguous()
        shape = (self.batch_groups, sparse_len, self.head_dim)
        keys = torch.empty(shape, dtype=self.dtype, device=selected_indices.device)
        values = torch.empty_like(keys)
        uva_gather_kv_rows(indices, self.cpu_kv_cache[layer_idx], keys, values, sparse_len)
        self.hit_mask_buffer[:, :sparse_len].zero_()
        return keys, values

    def _build_static_fixed_concat_flash_kv_from_indices(self, layer_idx, selected_indices):
        static_len = int(self.fixed_prompt_local_static_length_host)
        recent_len = self._lockstep_fixed_prompt_recent_length_host(layer_idx)
        sparse_len = self._active_sparse_len_for_selected_indices(selected_indices)
        selected_indices = selected_indices[:, :sparse_len].contiguous()
        total_len = static_len + recent_len + sparse_len
        self._ensure_static_fixed_concat_flash_storage(layer_idx, total_len)

        concat_keys_rows = self._static_fixed_concat_rows_view(
            self.static_fixed_concat_flash_key_storage[layer_idx],
            total_len,
        )
        concat_values_rows = self._static_fixed_concat_rows_view(
            self.static_fixed_concat_flash_value_storage[layer_idx],
            total_len,
        )
        static_keys_rows = self.static_keys[layer_idx].view(self.batch_groups, self.static_pattern_total, self.head_dim)
        static_values_rows = self.static_values[layer_idx].view(self.batch_groups, self.static_pattern_total, self.head_dim)
        recent_keys_rows = self.decode_hot_keys[layer_idx].view(self.batch_groups, self._decode_hot_capacity(), self.head_dim)
        recent_values_rows = self.decode_hot_values[layer_idx].view(self.batch_groups, self._decode_hot_capacity(), self.head_dim)
        if self.kv_on_gpu:
            # GPU-resident store: dedicated token-cache-free gather reading straight from device memory.
            if self.quantize_cpu_kv:
                concat_static_recent_gpu_gather_int8(
                    static_keys_rows, static_values_rows, recent_keys_rows, recent_values_rows,
                    selected_indices, self.cpu_kv_cache[layer_idx],
                    self.cpu_kv_k_scale[layer_idx], self.cpu_kv_v_scale[layer_idx],
                    concat_keys_rows, concat_values_rows, static_len, recent_len, sparse_len,
                )
            else:
                concat_static_recent_gpu_gather(
                    static_keys_rows, static_values_rows, recent_keys_rows, recent_values_rows,
                    selected_indices, self.cpu_kv_cache[layer_idx],
                    concat_keys_rows, concat_values_rows, static_len, recent_len, sparse_len,
                )
            return (
                self._static_fixed_concat_kvcache_view(
                    self.static_fixed_concat_flash_key_storage[layer_idx], total_len,
                ),
                self._static_fixed_concat_kvcache_view(
                    self.static_fixed_concat_flash_value_storage[layer_idx], total_len,
                ),
                total_len,
            )
        hit_mask_view = self.hit_mask_buffer[:, :sparse_len]
        hit_mask = hit_mask_view if hit_mask_view.is_contiguous() else hit_mask_view.contiguous()
        if self.disable_token_cache:
            gathered_k, gathered_v = self._uncached_gather(layer_idx, selected_indices, sparse_len)
            for out, static, recent, sparse in (
                (concat_keys_rows, static_keys_rows, recent_keys_rows, gathered_k),
                (concat_values_rows, static_values_rows, recent_values_rows, gathered_v),
            ):
                out[:, :static_len, 0].copy_(static[:, :static_len])
                out[:, static_len:static_len + recent_len, 0].copy_(recent[:, :recent_len])
                out[:, static_len + recent_len:, 0].copy_(sparse)
            hit_mask.zero_()
        elif self.quantize_cpu_kv:
            concat_static_recent_lookup_gather_uva_kv_update_cache_int8(
                static_keys_rows, static_values_rows, recent_keys_rows, recent_values_rows,
                selected_indices,
                self.cpu_kv_cache[layer_idx],
                self.cpu_kv_k_scale[layer_idx],
                self.cpu_kv_v_scale[layer_idx],
                concat_keys_rows,
                concat_values_rows,
                hit_mask,
                self.token_cache_ids[layer_idx],
                self.token_cache_locks[layer_idx],
                self.token_cache_keys[layer_idx],
                self.token_cache_values[layer_idx],
                static_len,
                recent_len,
                sparse_len,
            )
        else:
            stamps, step_dev, ways = self._token_cache_lru_args(layer_idx)
            if step_dev is not None:
                step_dev.fill_(int(self.lockstep_decode_step_host))
            prio_buckets, prio_scores = self._token_cache_prio_args(layer_idx)
            cache_prio = None
            concat_static_recent_lookup_gather_uva_kv_update_cache(
                static_keys_rows, static_values_rows, recent_keys_rows, recent_values_rows,
                selected_indices,
                self.cpu_kv_cache[layer_idx],
                concat_keys_rows,
                concat_values_rows,
                hit_mask,
                self.token_cache_ids[layer_idx],
                self.token_cache_locks[layer_idx],
                self.token_cache_keys[layer_idx],
                self.token_cache_values[layer_idx],
                static_len,
                recent_len,
                sparse_len,
                stamps,
                step_dev,
                ways,
                prio_buckets,
                prio_scores,
                cache_prio,
            )
        if hit_mask.data_ptr() != hit_mask_view.data_ptr():
            hit_mask_view.copy_(hit_mask)
        return (
            self._static_fixed_concat_kvcache_view(
                self.static_fixed_concat_flash_key_storage[layer_idx],
                total_len,
            ),
            self._static_fixed_concat_kvcache_view(
                self.static_fixed_concat_flash_value_storage[layer_idx],
                total_len,
            ),
            total_len,
        )

    def _static_fixed_concat_flash_attention_from_indices(self, queries, layer_idx, selected_indices,
                                                          return_lse=False):
        _pack_event_t = self._event_profile_start()
        lengths = None
        if self.num_gpus == 1:
            # Use exactly the CG key order and padded shape in eager mode too.
            # Different orders/split reductions perturb BF16 logits enough to
            # change sampled-tail generations even with identical random draws.
            self._ensure_cg_concat_buffers(layer_idx)
            live = (int(self.fixed_prompt_local_static_length_host)
                    + self._active_sparse_len_for_selected_indices(selected_indices)
                    + self._lockstep_fixed_prompt_recent_length_host(layer_idx))
            self._cg_cache_seqlens.fill_(live)
            _, step_dev, _ = self._token_cache_lru_args(layer_idx)
            if step_dev is not None:
                step_dev.fill_(int(self.lockstep_decode_step_host))
            flash_keys, flash_values = self._cg_build_concat(layer_idx, selected_indices)
            lengths = self._cg_cache_seqlens
        else:
            flash_keys, flash_values, _ = self._build_static_fixed_concat_flash_kv_from_indices(
                layer_idx, selected_indices)
        self._event_profile_end("decode_direct_concat_gather_pack", _pack_event_t)
        _flash_event_t = self._event_profile_start()
        try:
            return flash_attn_with_kvcache(
                q=queries,
                k_cache=flash_keys,
                v_cache=flash_values,
                cache_seqlens=lengths,
                causal=False,
                return_softmax_lse=return_lse,
            )
        finally:
            self._event_profile_end("decode_concat_flash_attn", _flash_event_t)

    def _cg_build_concat(self, layer_idx, selected_indices):
        # Build a FIXED-shape concat [static(static_len) | sparse(sparse_len) | recent(recent_capacity)]
        # at fixed width cg_concat_width, all at fixed offsets/addresses so it is CUDA-graph capturable.
        # static_len/sparse_len are constant within an inter-slide regime (baked at capture); the growing
        # recent window + sparse-pad are masked at flash time via self._cg_cache_seqlens (driver-set).
        self._ensure_cg_concat_buffers(layer_idx)
        rows = self.batch_groups
        dim = self.head_dim
        W = self.cg_concat_width
        recent_cap = self._decode_hot_capacity()
        static_len = int(self.fixed_prompt_local_static_length_host)
        sparse_len = self._active_sparse_len_for_selected_indices(selected_indices)
        max_topk = selected_indices.size(1)
        # Gather only [static(static_len) | sparse(sparse_len)] into the fixed-W buffer.
        # Preallocating for the full generation must not cause padded capacity slots to be
        # gathered on every step. Indices/hit-mask retain their capacity stride.
        # the gather kernels take out rows wider than the produced prefix (out_row_len = W derived
        # from out.size(1)), so the old tmp-then-copy_ assembly (an extra read+write of the whole
        # sparse region, ~40-60MB/layer/step) is gone. W >= static_len + max_topk + recent_cap
        # always holds (W is sized with static_pattern_total >= static_len).
        # A/B escape hatch: COMETKV_CG_DIRECT_GATHER=0 restores the legacy tmp+copy_ assembly.
        direct = os.environ.get("COMETKV_CG_DIRECT_GATHER", "1") != "0"
        if direct:
            tk = self._cg_concat_k[layer_idx].view(rows, W, 1, dim)
            tv = self._cg_concat_v[layer_idx].view(rows, W, 1, dim)
        else:
            need = static_len + max_topk
            if self._cg_legacy_tmp is None:
                self._cg_legacy_tmp = [None for _ in range(self.layer_num)]
            if self._cg_legacy_tmp[layer_idx] is None or self._cg_legacy_tmp[layer_idx][0].shape[1] != need:
                device = self._cg_concat_k[layer_idx].device
                self._cg_legacy_tmp[layer_idx] = (
                    torch.zeros((rows, need, 1, dim), dtype=self.dtype, device=device),
                    torch.zeros((rows, need, 1, dim), dtype=self.dtype, device=device),
                )
            tk, tv = self._cg_legacy_tmp[layer_idx]
        hit = self.hit_mask_buffer[:, :max_topk]
        gather_args = (
            self.static_keys[layer_idx].view(rows, self.static_pattern_total, dim),
            self.static_values[layer_idx].view(rows, self.static_pattern_total, dim),
            self.decode_hot_keys[layer_idx].view(rows, recent_cap, dim),
            self.decode_hot_values[layer_idx].view(rows, recent_cap, dim),
            selected_indices, self.cpu_kv_cache[layer_idx],
        )
        if self.kv_on_gpu:
            # GPU-resident store: dedicated token-cache-free gather reading straight from device memory.
            if self.quantize_cpu_kv:
                concat_static_recent_gpu_gather_int8(
                    *gather_args, self.cpu_kv_k_scale[layer_idx], self.cpu_kv_v_scale[layer_idx],
                    tk, tv, static_len, 0, sparse_len,
                )
            else:
                concat_static_recent_gpu_gather(
                    *gather_args, tk, tv, static_len, 0, sparse_len,
                )
        elif self.disable_token_cache:
            gathered_k, gathered_v = self._uncached_gather(layer_idx, selected_indices, sparse_len)
            for out, static, sparse in (
                (tk, gather_args[0], gathered_k), (tv, gather_args[1], gathered_v),
            ):
                out[:, :static_len, 0].copy_(static[:, :static_len])
                out[:, static_len:static_len + sparse_len, 0].copy_(sparse)
            hit.zero_()
        elif self.quantize_cpu_kv:
            concat_static_recent_lookup_gather_uva_kv_update_cache_int8(
                *gather_args, self.cpu_kv_k_scale[layer_idx], self.cpu_kv_v_scale[layer_idx],
                tk, tv, hit, self.token_cache_ids[layer_idx], self.token_cache_locks[layer_idx],
                self.token_cache_keys[layer_idx], self.token_cache_values[layer_idx],
                static_len, 0, sparse_len,
            )
        else:
            stamps, step_dev, ways = self._token_cache_lru_args(layer_idx)
            prio_buckets, prio_scores = self._token_cache_prio_args(layer_idx)
            cache_prio = None
            concat_static_recent_lookup_gather_uva_kv_update_cache(
                *gather_args, tk, tv, hit, self.token_cache_ids[layer_idx], self.token_cache_locks[layer_idx],
                self.token_cache_keys[layer_idx], self.token_cache_values[layer_idx],
                static_len, 0, sparse_len,
                stamps, step_dev, ways,
                prio_buckets, prio_scores,
                cache_prio,
            )
        # only the recent tail still needs assembly: it is packed right after the valid sparse
        # prefix (the kernel's [static|recent|sparse] region order cannot express recent-last).
        ck = self._cg_concat_k[layer_idx].view(rows, W, dim)
        cv = self._cg_concat_v[layer_idx].view(rows, W, dim)
        if not direct:
            tkk = tk.view(rows, static_len + max_topk, dim)
            tvv = tv.view(rows, static_len + max_topk, dim)
            ck[:, :static_len, :].copy_(tkk[:, :static_len, :])
            cv[:, :static_len, :].copy_(tvv[:, :static_len, :])
            ck[:, static_len:static_len + sparse_len, :].copy_(tkk[:, static_len:static_len + sparse_len, :])
            cv[:, static_len:static_len + sparse_len, :].copy_(tvv[:, static_len:static_len + sparse_len, :])
        lo = static_len + sparse_len
        ck[:, lo:lo + recent_cap, :].copy_(self.decode_hot_keys[layer_idx].view(rows, recent_cap, dim))
        cv[:, lo:lo + recent_cap, :].copy_(self.decode_hot_values[layer_idx].view(rows, recent_cap, dim))
        # flash views [batch, W, kv_head, dim]
        fk = self._static_fixed_concat_kvcache_view(self._cg_concat_k[layer_idx], W)
        fv = self._static_fixed_concat_kvcache_view(self._cg_concat_v[layer_idx], W)
        return fk, fv

    def _cg_sparse_attention(self, queries, layer_idx):
        self._grouped_query_asym_topk(queries, layer_idx)
        fk, fv = self._cg_build_concat(layer_idx, self.selected_indices_buffer)
        # cache_seqlens (device, driver-set) supplies the per-step live length, masking the padded tail.
        # num_splits is left auto: flash derives it on the host from the FIXED k_cache width W (not from
        # the device cache_seqlens), so it is constant across replays -> graph-safe AND split-kv-fast.
        if self.active_sample_len_host > 0:
            out, lse_main = flash_attn_with_kvcache(
                q=queries, k_cache=fk, v_cache=fv, cache_seqlens=self._cg_cache_seqlens,
                causal=False, return_softmax_lse=True,
            )
            self._sampled_tail_attention(queries, layer_idx, out, lse_main)
        else:
            out = flash_attn_with_kvcache(
                q=queries, k_cache=fk, v_cache=fv, cache_seqlens=self._cg_cache_seqlens, causal=False,
            )
        return out

    def sparse_attention(self, queries, layer_idx, static_len=None):
        if self.use_cuda_graph:
            with self._cuda_device_guard(queries.device):
                return self._cg_sparse_attention(queries, layer_idx)
        with self._cuda_device_guard(queries.device):
            _compute_event_t = self._event_profile_start()
            assert self.use_static_prompt_retrieval, (
                "CometKV fast-only sparse attention requires static prompt retrieval"
            )
            assert self._can_use_static_fixed_direct_concat_gather(), (
                "CometKV fast-only sparse attention requires direct concat gather"
            )
            visible_lengths = self.visible_lengths
            if self.range_starts_buffer is None:
                self._allocate_decode_buffers()
                self._refresh_static_recent_state(layer_idx, visible_lengths)

            _retrieval_event_t = self._event_profile_start()

            max_topk = self.selected_indices_buffer.size(1)
            if max_topk > 0:
                if self.selected_indices_buffer is None or self.selected_indices_buffer.size(1) < max_topk:
                    self._allocate_decode_buffers()
                    self._refresh_static_recent_state(layer_idx, visible_lengths)
                _fused_topk_event_t = self._event_profile_start()
                self._grouped_query_asym_topk(queries, layer_idx)
                self._event_profile_end("decode_query_asym_topk", _fused_topk_event_t)
            self._event_profile_end("decode_retrieval_topk_total", _retrieval_event_t)
            if self.active_sample_len_host > 0:
                out, lse_main = self._static_fixed_concat_flash_attention_from_indices(
                    queries,
                    layer_idx,
                    self.selected_indices_buffer,
                    return_lse=True,
                )
                _tail_event_t = self._event_profile_start()
                self._sampled_tail_attention(queries, layer_idx, out, lse_main)
                self._event_profile_end("decode_sampled_tail", _tail_event_t)
            else:
                out = self._static_fixed_concat_flash_attention_from_indices(
                    queries,
                    layer_idx,
                    self.selected_indices_buffer,
                )
            self._event_profile_end("decode_compute_total", _compute_event_t)
            return out
