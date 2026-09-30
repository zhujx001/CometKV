# CometKV

## Overview

CometKV accelerates long-context LLM decoding on one GPU by keeping the large retrieval portion of the KV cache in pinned CPU memory and fetching only a small attention working set through CUDA Unified Virtual Addressing (UVA). A compact hash signature selects the working set. Optional importance-corrected samples estimate contributions outside that selection using a separate, additional quota.

The repository contains the paper implementation, CUDA extensions, single-GPU inference path, LongBench and RULER pipelines, latency and throughput harnesses, and focused regression tests.

## Method sketch

CometKV separates each sequence into always-visible sink/recent regions and a retrieval region. Prefill uses dense attention, writes retrieval KV to pinned host memory, and builds a 16-byte `asym_n8` signature for each key: 120 sign bits plus an 8-bit log-norm code.

At each decode step:

1. The query scores the GPU-resident signatures and selects a deterministic head within the configured head budget.
2. An independent quota supplies additional samples. A shared proposal covers all retrieval candidates, mixing score probabilities with uniform probability. Each layer excludes samples already in its own head when merging the importance-corrected tail; head size is never reduced. Duplicate draws retain their multiplicity. Clipping ignores excluded slots, and an empty sampled tail leaves the main output unchanged.
3. A set-associative GPU token cache serves recurring selections. Misses are gathered from pinned host memory over UVA and packed with the sink/recent KV for FlashAttention.
4. The complete lockstep decode step can be captured and replayed with a CUDA graph.

The current defaults extend the original frozen-statistics / summed-query selector:
`--cometkv_stats_mode block --cometkv_query_aggregation mean_prob`. Prompt signatures
keep their original statistics. At each 128-token decode-window slide, newly evicted
keys receive a separate mean and log-norm range (the first block has 96 tokens with
the default 32-token overlap). Historical signatures and their statistics stay fixed.
Scoring adds the query-dependent offset between each block mean and the prompt mean,
so centering does not introduce an uncompensated bias between blocks.

Each GQA query head scores candidates separately. Its calibrated logits are normalized
with a softmax over retrieval candidates, and the head probabilities are averaged
before top-k. The kernel stores the log of this average, which preserves the ranking.
Simply summing linear per-head scores would be identical to summing the queries.
This is a candidate-conditional selector, not exact full-attention probabilities.
Quality gains from the new aggregation require task evaluation; they are not guaranteed.

Four query heads share signature loads in the CUDA scorer. Its scratch space is shared
between layers; block metadata costs `(head_dim + 2) * 4` bytes per block per KV head
per layer (520 bytes at dimension 128), in addition to the 16-byte token signatures.
Use `--cometkv_stats_mode frozen --cometkv_query_aggregation q_sum` to reproduce the
previous scoring rule. The head budget is refreshed on window slides using visible
length; buffers now cover the entire planned generation. Within each window k stays
fixed, and `COMETKV_MAX_RETRIEVAL_TOPK` can explicitly cap it. Preallocation does not
cause unused capacity slots to be gathered. This is length-based allocation, without
an online quality controller. See [tail and budget repair](report/cometkv_tail_budget_fix.md).

The default head retrieval budget is `0.02`. Tail defaults are `SIZE=256`, `TAU=1.0`,
`CLIP=4`, `STRIDE=8` (layers), `UNIFORM_MIX=0.1`, and `MAX_M=256`. Set
`COMETKV_SAMPLE_SIZE=0` for pure top-k, or `32`/`64`/`128`/`256` for an explicit additional
draw count per layer and KV head. These sizes are starting configurations, not measured
quality optima. The old `COMETKV_SAMPLE_FRAC` override still derives a sample count
from k with `MIN_M=64`, but now also adds that count outside the head budget. Explicit
`COMETKV_SAMPLE_SIZE` takes precedence; `COMETKV_SAMPLE_FRAC=0` still disables sampling
when SIZE is unset. Neither tail nor preserved-token traffic should be hidden in the
reported head retrieval ratio.

The sampled-tail merge uses an exact shared-memory head set and vectorized, parallel
reductions for aligned 128-dimensional KV. See the [NCU optimization report](report/cometkv_ncu_optimization.md)
for hardware counters, numerical checks, and kernel/model latency comparisons.

The selector now uses exact radix top-k for long sparse ranges and FP32 nibble
lookup scoring for block/q_sum. See the [selector optimization report](report/cometkv_selector_optimization.md)
for the Tensor Core experiments and controlled comparisons. Large k falls back
to torch top-k without reducing the retrieval budget.

## Backends

| `attn_type` | Retrieval KV | Selection and purpose |
|---|---|---|
| `CometKV` | Pinned CPU memory, gathered through UVA | `asym_n8`, block compensation, GQA probability aggregation, sampled tail, and GPU token cache |
| `CometKV_GPU` | GPU memory | Same retrieval method without host-transfer cost; useful for controlled speed comparisons and moderate contexts |
| `Full_Flash_Attn` | Dense GPU KV | Full-attention baseline |
| `Exact_TopK` | Dense GPU KV | Exact query-key top-k oracle; eager decode only and supported for Llama/Mistral, not Qwen |

`CometKV` and `CometKV_GPU` require same-length, unpadded prompts within a batch. Use batch size 1 or construct a batch whose tokenized prompts have identical lengths.

## Install

Python 3.10 and a local CUDA 12.x toolkit are recommended. The toolkit used to compile the extension must be compatible with the installed PyTorch CUDA stack.

```bash
conda create -n cometkv python=3.10 -y
conda activate cometkv
conda install -y mkl
python -m pip install pip==25.0

python -m pip install -r requirements.txt
python -m pip install -r requirements-cu128.txt

export CUDA_HOME=/path/to/your/cuda-12.x
export PATH="$CUDA_HOME/bin:$PATH"

cd library/cometkv
python -m pip install .
cd ../..
```

Set `CUDA_HOME` to the CUDA 12.x toolkit installed on your machine; do not assume a fixed `/usr/local/cuda-*` location.

The default model ids (`meta-llama/...`, `mistralai/...`) are gated on HuggingFace:
either run `huggingface-cli login` (or set `HF_TOKEN`) after accepting the model
licenses, or point `--model_name` / `MODEL_PATH` at a local model directory.

## Quick start

Model and dataset bindings live in `config/paths.json`. Override the whole file with `COMETKV_PATH_CONFIG=/path/to/paths.json`, or pass `--model_name` directly.

```bash
conda activate cometkv
export PYTHONPATH="$PWD:$PWD/library/cometkv${PYTHONPATH:+:$PYTHONPATH}"
export COMETKV_EVENT_PROFILE=0

python -u simple_test.py \
  --model_name /path/to/Llama-3.1-8B-Instruct \
  --attn_type CometKV \
  --retrieval_budget 0.02 \
  --use_cuda_graph \
  --profile_timing
```

Useful alternatives are `--attn_type Full_Flash_Attn`, `--attn_type CometKV_GPU`, and `--cometkv_cpu_kv_quant int8`. Use `COMETKV_SAMPLE_SIZE=0` to disable the sampled tail while retaining `asym_n8` retrieval.

## Data preparation (one-click)

`scripts/prepare_data.sh` downloads LongBench, fetches the RULER source corpora
(Paul Graham essays, SQuAD, HotpotQA), and pre-generates the RULER task data in
exactly the layout the benchmark wrappers expect. Every step is idempotent —
existing files are kept, only missing pieces are fetched or generated.

```bash
# everything (LongBench + 13 RULER tasks x 32k/64k/96k, 50 samples each)
DATA_ROOT=/path/to/data bash scripts/prepare_data.sh

# RULER only, a subset
PREPARE_LONGBENCH=0 TASKS="fwe niah_single_1 vt" RULER_LENGTHS="32768" \
  DATA_ROOT=/path/to/data bash scripts/prepare_data.sh

# then run the benchmarks against the same root
DATA_ROOT=/path/to/data bash scripts/run_longbench.sh
DATA_ROOT=/path/to/data bash scripts/run_ruler.sh
```

Knobs: `MODEL_PATH` (tokenizer used to calibrate RULER lengths; must match the
model you evaluate), `NUM_SAMPLES` (default 50), `HF_ENDPOINT` (LongBench mirror),
`DRY_RUN=1` (print commands only). Downloads honor `HTTP_PROXY`/`HTTPS_PROXY`.

## Benchmarks

The top-level wrappers activate the `cometkv` conda environment unless it is already active. All defaults can be overridden through their documented environment variables or command-line options.

```bash
# LongBench: DATASETS=paper (default) expands to the 16-task paper set
bash scripts/run_longbench.sh
DATASETS="qasper 2wikimqa" MAX_SAMPLES=50 GPU_ID=0 bash scripts/run_longbench.sh

# RULER: needs pre-generated data at $RULER_DATA_DIR/<len>/<task>/validation.jsonl
bash scripts/run_ruler.sh
TASKS="fwe niah_single_1 vt" RULER_LENGTHS="32768 65536" MAX_SAMPLES=25 bash scripts/run_ruler.sh

# FWE latency / TPOT sweep
bash scripts/run_latency.sh

# Same-length batch throughput sweep
bash throughput_eval/run.sh
```

Shared wrapper environment knobs (defaults in parentheses): `MODEL_NAME` (`llama-3.1`; also `qwen`/`mistral`) or `MODEL_PATH`, `ATTN_TYPE` (`CometKV`), `BUDGET` (`0.02`), `SINK`/`RECENT` (`4`/`32`, counted inside the budget via `--cometkv_include_preserved_in_budget`), `COMETKV_SELECTOR` (`asym_n8`), `MAX_SAMPLES`, `GPU_ID`, `CONDA_ENV` (`cometkv`), `DATA_ROOT` (repo-local `./data`). Each run writes `run_config.json` plus tee'd logs under `results/{longbench,ruler,latency}/<model_tag>/<run_name>/`, and finishes with per-task evaluation (`summary.csv` / `result.json`).

Use `DRY_RUN=1` with the three `scripts/run_*.sh` wrappers to inspect generated commands. The lower-level LongBench and RULER drivers use `LONG_BENCH_DRY_RUN=1` and `RULER_DRY_RUN=1`, respectively.

Expected data layouts:

```text
$DATA_ROOT/longbench-jsonl/
  qasper.jsonl
  qasper_e.jsonl          # optional LongBench-E form
  ...

$DATA_ROOT/RULER/
  32768/
    fwe/validation.jsonl
    niah_single_1/validation.jsonl
  65536/
    ...
  98304/
    ...

test_data/
  fwe.json                # latency wrapper input

throughput_eval/test_data/
  NIAH_30000.json
  NIAH_60000.json
  NIAH_120000.json
  fwe.json
  vt.json
  qa1.json
  AIME.json
```

The wrappers write generated outputs below `results/`; throughput logs go to `throughput_eval/cometkv_logs/`. `throughput_eval/run.sh` requires `numactl` (NUMA-pinned measurement).

Bundled test JSON provenance: `throughput_eval/test_data/qa1.json` embeds SQuAD
passages (Wikipedia-derived, CC BY-SA 4.0); the `NIAH_*` haystacks are public-domain
King James Bible text; the remaining bundled inputs are synthetic RULER-style content.

## Configuration

Command-line arguments configure the stable public interface; environment variables are convenient for benchmark sweeps and low-level runtime controls.

| Setting | Default | Effect |
|---|---:|---|
| `--retrieval_budget` | `0.02` | Fraction of the visible sequence allocated to retrieval |
| `--cometkv_static_pattern_start` / `_end` | `4` / `32` | Always-visible sink / prompt-recent window sizes (unified across model templates and wrappers) |
| `--cometkv_selector` / `COMETKV_SELECTOR` | `asym_n8` | The supported 16-byte hash-signature selector |
| `--cometkv_stats_mode` / `COMETKV_STATS_MODE` | `block` | Per-eviction-block mean and norm scale with score compensation; `frozen` uses prompt statistics |
| `--cometkv_query_aggregation` / `COMETKV_QUERY_AGG` | `mean_prob` | Average per-query candidate softmax probabilities before top-k; `q_sum` restores summed-query scoring |
| `COMETKV_SCORE_IMPL` | `auto` | FP32 lookup for block/q_sum, scalar otherwise; `scalar` / `lookup` select an explicit implementation |
| `COMETKV_TOPK_IMPL` | `auto` | Exact radix selection for ≥4096 candidates and K≤min(4096,candidates/4); `torch` / `radix` select a comparison path; K>4096 retains the full budget via torch |
| `--cometkv_min_retrieval_topk` / `COMETKV_MIN_RETRIEVAL_TOPK` | `16` | Minimum retrieved tokens per row; the environment form is consumed by wrappers |
| `--cometkv_token_cache_size` / `COMETKV_TOKEN_CACHE_SIZE` | `1024` | Initial per-row token-cache floor; the environment form is consumed by wrappers |
| `COMETKV_MAX_RETRIEVAL_TOPK` | `0` | Explicit upper bound on head k; `0` means no extra cap beyond length and candidates |
| `COMETKV_SAMPLE_SIZE` | `256` in generated configs | Additional sample slots per layer/KV head; `0` disables; overrides FRAC |
| `COMETKV_SAMPLE_FRAC` | unset | Compatibility override: derive additional sample count from head k; never reduces head |
| `COMETKV_SAMPLE_TAU` / `COMETKV_SAMPLE_SEED` | `1.0` / `1234` | Proposal temperature and deterministic sampling seed |
| `COMETKV_SAMPLE_CLIP` | `4.0` | Cap for corrected tail logits in nats; `0` disables clipping |
| `COMETKV_SAMPLE_STRIDE` | `8` | Resample cadence across layers |
| `COMETKV_SAMPLE_MIN_M` / `COMETKV_SAMPLE_MAX_M` | `64` / `256` | Minimum for fraction-derived counts only; hard cap applies to all sample counts |
| `COMETKV_SAMPLE_UNIFORM_MIX` | `0.1` | Uniform mixture weight in `(0,1]`, ensuring support across all retrieval candidates |
| `COMETKV_SAMPLE_AUTOSCALE` / `COMETKV_SAMPLE_SIGMA` | `0` for `mean_prob`, `1` for `q_sum` / `2.0` | Optional proposal standardization; `mean_prob` normally uses its log probabilities directly |
| `COMETKV_TOKEN_CACHE_WAYS` | `8` | Set associativity; values above 2 enable stamped W-way replacement, `2` restores the legacy stamp-free path |
| `COMETKV_TOKEN_CACHE_POLICY` | `score` | `score` evicts the worst-selector-priority way (paper default); `lru` evicts the least recently used |
| `COMETKV_TOKEN_CACHE_MULT` | `4.0` | Target cache capacity relative to retrieval top-k |
| `COMETKV_TOKEN_CACHE_RESERVE_GB` | `4.0` | VRAM reserved after token-cache sizing |
| `COMETKV_NO_KEY_CENTER` | `0` | Set to `1` to disable key centering and mean compensation; block norm ranges still update |
| `--cometkv_cpu_kv_quant` / `COMETKV_CPU_KV_QUANT` | `none` | `int8` stores retrieval K per channel and V per token, then dequantizes on gather |
| `COMETKV_MEAN_UPDATE_ALPHA` | `0.0` | Nonzero forward-only EMA is rejected because old signatures require their original statistics |
| `COMETKV_NORM_MARGIN` | `0.0` | Margin around each newly sealed block's log-norm range |
| `COMETKV_FULL_RECOMPUTE_INTERVAL` | `0` | Frozen-mode ablation: periodically rebuild all statistics/signatures on eviction boundaries; incompatible with block mode |
| `COMETKV_EXCLUDE_PRESERVED_FROM_BUDGET` / `COMETKV_INCLUDE_PRESERVED_IN_BUDGET` | exclude | LongBench wrapper budget accounting for sink/recent tokens |
| `COMETKV_CG_DIRECT_GATHER` | `1` | Directly gather into fixed CUDA-graph concat buffers |
| `COMETKV_CG_EAGER` | `0` | Debug escape hatch that runs the graph-compatible path eagerly |
| `COMETKV_EVENT_PROFILE` | `0` | Enable per-phase CUDA event profiling |
| `COMETKV_DEBUG_RECOMPUTE` | `0` | Print full-recompute diagnostics |
| `COMETKV_PATH_CONFIG` | `config/paths.json` | Alternate path registry |
| `EXACT_TOPK_FORCE_SINK` / `EXACT_TOPK_FORCE_RECENT` | `0` / `0` | Exact_TopK baseline: pin sink/recent rows into the oracle selection (counted inside its budget) |
| `EXACT_TOPK_MIN_TOPK` | `1` | Exact_TopK: floor on the frozen top-k |
| `EXACT_TOPK_SAMPLE_FRAC` / `_TAU` / `_SEED` | `0` / `1.0` / `1234` | Exact_TopK oracle sampled-tail prototype (0 = pure top-k, bit-identical) |

The corresponding CLI flags for sink/recent sizes, budget accounting, quantization, mean updates, norm margin, and recompute cadence are defined by `python simple_test.py --help`.

## Supported models

The checked-in model templates and path aliases cover:

- Llama 3.1 8B Instruct
- Llama 3 8B Instruct Gradient 1048k
- Qwen2.5 7B Instruct
- Mistral 7B Instruct v0.2
- DeepSeek-R1-Distill-Llama-8B
- DeepSeek-R1-Distill-Qwen-7B

Llama-family models route through `model_hub/llama.py`, Qwen-family models through `model_hub/qwen.py`, and Mistral through `model_hub/mistral.py`. `Exact_TopK` is intentionally available only on the Llama/Mistral path.

## Tests

Run lightweight repository and benchmark tests:

```bash
python -m pytest -q \
  test/test_paths.py \
  test/test_optional_backend_imports.py \
  test/test_environment_metadata.py \
  benchmark/longbench \
  benchmark/ruler \
  scripts/test_run_cometkv_benchmarks.py
```

After building the extension on a CUDA host, run the renamed kernel/runtime suites:

```bash
python -m pytest -q library/cometkv/test/test_cometkv_*.py
python test/cg_stage1_test.py
python test/cg_bisect_cache.py
python test/int8_gather_test.py
python test/int8_e2e_compare.py
```

The CUDA tests require a compatible GPU and a freshly built `cometkv` extension.

## License

Apache License 2.0 — see [LICENSE](LICENSE). The RULER data-generation code under
`benchmark/ruler/data/` derives from NVIDIA's [RULER](https://github.com/NVIDIA/RULER)
(Apache-2.0); LongBench evaluation code derives from
[THUDM/LongBench](https://github.com/THUDM/LongBench) (MIT).

## Citation

If you use CometKV in your research, please cite the paper (citation entry to be
added upon publication).
