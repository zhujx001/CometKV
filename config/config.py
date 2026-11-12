import os, json
from pathlib import Path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _default_cometkv_config():
    return {
        "static_pattern_start": 4,
        "static_pattern_end": 32,
        "core": get_numa_node_core_count(0),
        "retrieval_budget": 0.02,
        "sig_bits": 128,
        "sig_seed": 1234,
        "sig_mode": "random_orth",
        "sig_topk": 0,
        "sig_chunk_size": 131072,
        "sig_token_cache_size": 1024,
        "sig_min_retrieval_topk": 0,
        "exclude_preserved_from_budget": True,
        "cpu_kv_quant": "none",
        "stats_mode": "block",
        "query_aggregation": "q_sum",
        "mean_update_alpha": 0.0,
        "norm_margin": 0.0,
        "full_recompute_interval": 0,
    }


def add_config_args(parser):
    parser.add_argument("--attn_type", type=str, default="CometKV",
                        choices=["Full_Flash_Attn", "CometKV", "CometKV_GPU", "Exact_TopK"],
                        help="Attention method. CometKV_GPU keeps the retrieval KV store on GPU "
                             "(no UVA) for an apples-to-apples speed comparison vs Full_Flash_Attn. "
                             "Exact_TopK is the oracle baseline: pure exact q.k top-k retrieval at "
                             "decode (GQA group-mean query scoring), k frozen from the prompt length "
                             "as int(retrieval_budget * prompt_len).")
    parser.add_argument("--retrieval_budget", type=float, default=0.02, help="Retrieval budget")
    parser.add_argument("--sig_bits", type=int, default=128, help="Signature bit width for CometKV")
    parser.add_argument("--sig_seed", type=int, default=1234, help="Random projection seed for CometKV")
    parser.add_argument("--sig_chunk_size", type=int, default=131072, help="Chunk size for CometKV signature retrieval")
    parser.add_argument("--cometkv_min_retrieval_topk", type=int, default=0,
                        help="Minimum CometKV retrieval topk per row when candidates exist. "
                             "0 (default, paper) uses K_t = floor(rho * L_t) without a floor.")
    parser.add_argument("--cometkv_token_cache_size", type=int, default=1024,
                        help="Minimum CometKV token cache slots per row. Runtime may raise it to the retrieval topk.")
    parser.add_argument("--cometkv_static_pattern_start", type=int, default=None,
                        help="Override CometKV sink tokens. Defaults to the model config value.")
    parser.add_argument("--cometkv_static_pattern_end", type=int, default=None,
                        help="Override CometKV recent/local tokens. Defaults to the model config value.")
    parser.add_argument("--cometkv_exclude_preserved_from_budget",
                        dest="cometkv_exclude_preserved_from_budget",
                        action="store_true",
                        default=True,
                        help="Do not count CometKV sink/recent direct-attention tokens against the retrieval budget.")
    parser.add_argument("--cometkv_include_preserved_in_budget",
                        dest="cometkv_exclude_preserved_from_budget",
                        action="store_false",
                        help="Count CometKV sink/recent direct-attention tokens against the retrieval budget.")
    parser.add_argument("--cometkv_cpu_kv_quant", type=str, default="none", choices=["none", "int8"],
                        help="Quantize the CPU retrieval KV store ('int8' = K per-channel / V per-token, "
                             "dequantized to model dtype on gather) to cut UVA/PCIe gather traffic.")
    parser.add_argument("--cometkv_selector", type=str, default="asym_n8",
                        choices=["asym_n8"],
                        help="CometKV retrieval selector. 'asym_n8' (default): asymmetric scoring over "
                             "a 16B/token signature (112 sign bits + 8-bit query-centering code + "
                             "8-bit log-norm) plus per-block statistics and shared scoring workspace.")
    parser.add_argument("--cometkv_stats_mode", choices=["block", "mu_frozen", "frozen"], default="block",
                        help="block: update mean and norm scale per block; mu_frozen: freeze prompt mean "
                             "and update norm scale per block; frozen: freeze both prompt statistics.")
    parser.add_argument("--cometkv_query_aggregation", choices=["q_sum", "mean_prob"], default="q_sum",
                        help="Sum GQA queries before scoring (default); mean_prob is an explicit alternative.")
    parser.add_argument("--cometkv_mean_update_alpha", type=float, default=0.0,
                        help="Deprecated forward-only EMA. Must be 0; use --cometkv_stats_mode block.")
    parser.add_argument("--cometkv_norm_margin", type=float, default=0.0,
                        help="Widen each newly sealed block's log-norm range by this fraction. "
                             "0.0 = observed range (default). 0.3 = 30%% margin each side.")
    parser.add_argument("--cometkv_full_recompute_interval", type=int, default=0,
                        help="Frozen-mode ablation: rebuild all statistics/signatures on eviction "
                             "boundaries after N evicted tokens. 0 = disabled; incompatible with block mode.")
    return parser


def get_numa_node_core_count(node_id=0):
    path = Path(f"/sys/devices/system/node/node{node_id}/cpulist")
    if not path.exists():
        count = os.cpu_count()
        print(f"NUMA node{node_id} not found, set core to #total_cpu_core: {count}")
        return max(count - 2, 1)    # reserve 2 cores for system
    # get NUMA node core count
    cpulist = path.read_text().strip()
    count = 0
    for part in cpulist.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            count += end - start + 1
        else:
            count += 1
    return max(count - 2, 1)  # reserve 2 cores for system


def generate_config(
    model_name, context_len, attn_type, 
    retrieval_budget=0.02, sig_bits=128, sig_seed=1234, sig_chunk_size=131072,
    cometkv_min_retrieval_topk=0,
    cometkv_token_cache_size=1024,
    cometkv_static_pattern_start=None,
    cometkv_static_pattern_end=None,
    cometkv_exclude_preserved_from_budget=True,
    cometkv_cpu_kv_quant="none",
    cometkv_selector="asym_n8",
    cometkv_mean_update_alpha=0.0,
    cometkv_norm_margin=0.0,
    cometkv_full_recompute_interval=0,
    cometkv_stats_mode="block",
    cometkv_query_aggregation="q_sum",
    cometkv_head_extra_tokens=0,
    cometkv_tail_in_budget=True,
    cometkv_sample_min_context=32768,
    cometkv_sample_mode="fraction",
    cometkv_sample_frac=0.25,
    cometkv_sample_size=256,
    cometkv_sample_max_m=256,
    cometkv_sample_min_m=64,
    cometkv_token_cache_mult=2.0,
    cometkv_token_cache_round=8,
):
    CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
    MODEL_NAME = model_name.split("/")[-1]+'.json'
    CONFIG_FILE = os.path.join(CONFIG_DIR, MODEL_NAME)
    with open(CONFIG_FILE, "r") as f:
        _config = json.load(f)
    
    if attn_type in ("CometKV", "CometKV_GPU"):
        # CometKV_GPU reuses the CometKV config under its own key (so init_kv_cache, which looks the
        # config up by attention_type, finds it) and only flips the retrieval store onto the GPU.
        template = _config.get("CometKV", _default_cometkv_config())
        _config.setdefault(attn_type, dict(template))
        _config[attn_type]["core"] = get_numa_node_core_count(0)
        if cometkv_static_pattern_start is not None:
            _config[attn_type]["static_pattern_start"] = int(cometkv_static_pattern_start)
        if cometkv_static_pattern_end is not None:
            _config[attn_type]["static_pattern_end"] = int(cometkv_static_pattern_end)
        _config[attn_type]["retrieval_budget"] = retrieval_budget
        _config[attn_type]["head_extra_tokens"] = int(cometkv_head_extra_tokens)
        _config[attn_type]["sig_bits"] = int(sig_bits)
        _config[attn_type]["sig_seed"] = int(sig_seed)
        _config[attn_type]["sig_chunk_size"] = int(sig_chunk_size)
        _config[attn_type]["sig_min_retrieval_topk"] = max(int(cometkv_min_retrieval_topk), 0)
        _config[attn_type]["sig_max_retrieval_topk"] = int(os.environ.get(
            "COMETKV_MAX_RETRIEVAL_TOPK", _config[attn_type].get("sig_max_retrieval_topk", 0)))
        _config[attn_type]["sig_token_cache_size"] = int(cometkv_token_cache_size)
        _config[attn_type]["exclude_preserved_from_budget"] = bool(cometkv_exclude_preserved_from_budget)
        # Environment override lets benchmark scripts toggle int8 without adding wrapper flags.
        _config[attn_type]["cpu_kv_quant"] = str(
            os.environ.get("COMETKV_CPU_KV_QUANT", cometkv_cpu_kv_quant)
        )
        # Retrieval selector: "asym_n8" (asymmetric scoring over a 16B/token signature:
        # 112 sign bits + 8-bit query-centering code + 8-bit log-norm).
        _config[attn_type]["sig_selector"] = str(
            os.environ.get("COMETKV_SELECTOR", _config[attn_type].get("sig_selector", cometkv_selector))
        )
        _config[attn_type]["stats_mode"] = os.environ.get("COMETKV_STATS_MODE", cometkv_stats_mode).lower()
        _config[attn_type]["query_aggregation"] = os.environ.get(
            "COMETKV_QUERY_AGG", cometkv_query_aggregation
        ).lower()
        # Kept for configuration compatibility. Nonzero forward-only EMA is rejected by
        # the cache because old signatures require their original center and norm scale.
        _config[attn_type]["mean_update_alpha"] = float(
            os.environ.get("COMETKV_MEAN_UPDATE_ALPHA",
                           _config[attn_type].get("mean_update_alpha", cometkv_mean_update_alpha))
        )
        # Widen each newly sealed block's log-norm range (prompt only in frozen mode).
        _config[attn_type]["norm_margin"] = float(
            os.environ.get("COMETKV_NORM_MARGIN",
                           _config[attn_type].get("norm_margin", cometkv_norm_margin))
        )
        _config[attn_type]["full_recompute_interval"] = int(
            os.environ.get("COMETKV_FULL_RECOMPUTE_INTERVAL",
                           _config[attn_type].get("full_recompute_interval", cometkv_full_recompute_interval))
        )
        # Sampled tail (paper defaults; every key has an environment override):
        # sample_mode (COMETKV_SAMPLE_MODE): fraction -> m = min(floor(sample_frac * K_t), sample_max_m),
        #   no tail if m < sample_min_m; size -> m = min(sample_size, sample_max_m); off -> no tail.
        # tail_in_budget (env COMETKV_TAIL_IN_BUDGET=0/1, default 1 = paper): the m draws count
        # against rho (head = K_t - m); 0 = head K_t plus m additional draws.
        # sample_min_context (env COMETKV_SAMPLE_MIN_CONTEXT, default 32768 = paper): no tail
        # while the visible context L_t is shorter; 0 = always.
        _config[attn_type]["tail_in_budget"] = os.environ.get(
            "COMETKV_TAIL_IN_BUDGET",
            "1" if _config[attn_type].get("tail_in_budget", cometkv_tail_in_budget) else "0",
        ) == "1"
        _config[attn_type]["sample_min_context"] = int(os.environ.get(
            "COMETKV_SAMPLE_MIN_CONTEXT",
            _config[attn_type].get("sample_min_context", cometkv_sample_min_context),
        ))
        if "COMETKV_SAMPLE_MODE" not in os.environ and "COMETKV_SAMPLE_SIZE" in os.environ:
            raise ValueError("COMETKV_SAMPLE_SIZE is set without COMETKV_SAMPLE_MODE; set COMETKV_SAMPLE_MODE=size "
                             "for a fixed draw count (the default mode is fraction)")
        for key, env, default, cast in (
            ("sample_mode", "COMETKV_SAMPLE_MODE", cometkv_sample_mode, lambda v: str(v).lower()),
            ("sample_frac", "COMETKV_SAMPLE_FRAC", cometkv_sample_frac, float),
            ("sample_size", "COMETKV_SAMPLE_SIZE", cometkv_sample_size, int),
            ("sample_max_m", "COMETKV_SAMPLE_MAX_M", cometkv_sample_max_m, int),
            ("sample_min_m", "COMETKV_SAMPLE_MIN_M", cometkv_sample_min_m, int),
            ("token_cache_mult", "COMETKV_TOKEN_CACHE_MULT", cometkv_token_cache_mult, float),
            ("token_cache_round", "COMETKV_TOKEN_CACHE_ROUND", cometkv_token_cache_round, int),
        ):
            _config[attn_type][key] = cast(os.environ.get(env, _config[attn_type].get(key, default)))
        _config[attn_type]["sample_tau"] = float(
            os.environ.get("COMETKV_SAMPLE_TAU", _config[attn_type].get("sample_tau", 1.0))
        )
        _config[attn_type]["sample_seed"] = int(
            os.environ.get("COMETKV_SAMPLE_SEED", _config[attn_type].get("sample_seed", 1234))
        )
        _config[attn_type]["kv_store_device"] = "gpu" if attn_type == "CometKV_GPU" else "cpu"
    elif attn_type == "Exact_TopK":
        # Oracle exact top-k retrieval baseline. Pure retrieval by default; env overrides let
        # protocol scripts pin sink/recent rows outside the budget without new CLI flags:
        #   EXACT_TOPK_FORCE_SINK=4 EXACT_TOPK_FORCE_RECENT=32 bash scripts/run_longbench.sh
        template = _config.get("Exact_TopK", {})
        _config["Exact_TopK"] = {
            "retrieval_budget": retrieval_budget,
            "force_sink": int(os.environ.get("EXACT_TOPK_FORCE_SINK", template.get("force_sink", cometkv_static_pattern_start if cometkv_static_pattern_start is not None else 4))),
            "force_recent": int(os.environ.get("EXACT_TOPK_FORCE_RECENT", template.get("force_recent", cometkv_static_pattern_end if cometkv_static_pattern_end is not None else 32))),
            "min_retrieval_topk": int(os.environ.get("EXACT_TOPK_MIN_TOPK", template.get("min_retrieval_topk", 1))),
            # Hybrid sampled-tail estimator (0.0 = pure top-k, bit-identical
            # legacy behavior): sample_frac of the SAME budget k is drawn from the tail with
            # importance-weighted logits instead of taking the next-best exact tokens.
            #   EXACT_TOPK_SAMPLE_FRAC=0.5 EXACT_TOPK_SAMPLE_TAU=1.0 bash scripts/run_ruler.sh
            "sample_frac": float(os.environ.get("EXACT_TOPK_SAMPLE_FRAC", template.get("sample_frac", 0.0))),
            "sample_tau": float(os.environ.get("EXACT_TOPK_SAMPLE_TAU", template.get("sample_tau", 1.0))),
            "sample_seed": int(os.environ.get("EXACT_TOPK_SAMPLE_SEED", template.get("sample_seed", 1234))),
        }
    elif attn_type != "Full_Flash_Attn":
        raise ValueError(f"Unsupported attention type: {attn_type}")
    
    if attn_type != "Full_Flash_Attn":
        print(_config[attn_type])
    
    return _config
