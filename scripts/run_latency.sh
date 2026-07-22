#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-cometkv}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
RULER_DATA_DIR="${RULER_DATA_DIR:-${DATA_ROOT}/RULER}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
DEVICE="${DEVICE:-cuda:0}"

MODEL_NAME="${MODEL_NAME:-llama-3.1}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_TAG="${MODEL_TAG:-}"
MODEL_KEY="${MODEL_KEY:-}"
CONFIG_MODEL_NAME="${CONFIG_MODEL_NAME:-}"

BUDGET="${BUDGET:-0.02}"
SINK="${SINK:-4}"
RECENT="${RECENT:-32}"
ATTN_TYPE="${ATTN_TYPE:-CometKV}"
COMETKV_SELECTOR="${COMETKV_SELECTOR:-asym_n8}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
DRY_RUN="${DRY_RUN:-0}"
LENGTHS="${LENGTHS:-32768 98304}"
DTYPE="${DTYPE:-bf16}"
BATCH_SIZES="${BATCH_SIZES:-1}"
MAX_NEW_LENGTH="${MAX_NEW_LENGTH:-256}"
IGNORE_FIRST_STEPS="${IGNORE_FIRST_STEPS:-1}"
PREFILL_BSZ="${PREFILL_BSZ:-1}"
PREFILL_METHOD="${PREFILL_METHOD:-full}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/test_data/fwe.json}"
COMETKV_MIN_RETRIEVAL_TOPK="${COMETKV_MIN_RETRIEVAL_TOPK:-16}"
COMETKV_TOKEN_CACHE_SIZE="${COMETKV_TOKEN_CACHE_SIZE:-1024}"
SIG_BITS="${SIG_BITS:-128}"
SIG_SEED="${SIG_SEED:-1234}"
SIG_CHUNK_SIZE="${SIG_CHUNK_SIZE:-131072}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DO_SAMPLE="${DO_SAMPLE:-false}"

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_latency.sh

Environment overrides:
  MODEL_NAME=llama|llama-3.1 MODEL_PATH=/path/to/model GPU_ID=0 DEVICE=cuda:0
  LENGTHS="32768 98304" BATCH_SIZES=1 MAX_NEW_LENGTH=256
  BUDGET=0.02 SINK=4 RECENT=32 COMETKV_SELECTOR=asym_n8
  DRY_RUN=1 CONDA_ENV=cometkv
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --model-name|--model) MODEL_NAME="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --lengths) LENGTHS="$2"; shift 2 ;;
    --gpu-id) GPU_ID="$2"; CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --budget) BUDGET="$2"; shift 2 ;;
    --sink) SINK="$2"; shift 2 ;;
    --recent) RECENT="$2"; shift 2 ;;
    --attn-type) ATTN_TYPE="$2"; shift 2 ;;
    --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
    --max-new-length) MAX_NEW_LENGTH="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --result-root) RESULT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

is_truthy() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

sanitize_component() {
  local value="$1"
  value="${value//\//_}"
  value="${value// /_}"
  value="${value//:/_}"
  printf '%s' "${value}"
}

resolve_model() {
  local key="${MODEL_NAME,,}"
  case "${key}" in
    llama|llama3|llama-3|llama-3.1|llama-3.1-8b|llama-3.1-8b-instruct|meta-llama-3.1-8b-instruct|meta-llama/meta-llama-3.1-8b-instruct)
      MODEL_KEY="${MODEL_KEY:-llama-3.1-8b}"
      MODEL_TAG="${MODEL_TAG:-llama-3.1}"
      CONFIG_MODEL_NAME="${CONFIG_MODEL_NAME:-Llama-3.1-8B-Instruct}"
      MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
      ;;
    qwen|qwen2.5|qwen-2.5|qwen2.5-7b|qwen2.5-7b-instruct|qwen-2.5-7b-instruct|qwen/qwen2.5-7b-instruct)
      MODEL_KEY="${MODEL_KEY:-qwen2.5-7b}"
      MODEL_TAG="${MODEL_TAG:-qwen2.5-7b}"
      CONFIG_MODEL_NAME="${CONFIG_MODEL_NAME:-Qwen2.5-7B-Instruct}"
      MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
      ;;
    mistral|mistral-v0.2|mistral-7b|mistral-7b-instruct|mistral-7b-instruct-v0.2|mistralai/mistral-7b-instruct-v0.2)
      MODEL_KEY="${MODEL_KEY:-mistral-7b-instruct-v0.2}"
      MODEL_TAG="${MODEL_TAG:-mistral-7b-v0.2}"
      CONFIG_MODEL_NAME="${CONFIG_MODEL_NAME:-Mistral-7B-Instruct-v0.2}"
      MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-7B-Instruct-v0.2}"
      ;;
    *)
      echo "Unsupported MODEL_NAME=${MODEL_NAME}" >&2
      exit 2
      ;;
  esac
}

space_list_to_csv() {
  local value="$1"
  value="${value//, /,}"
  value="${value// /,}"
  printf '%s' "${value}"
}

print_command() {
  printf 'RUN:'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

run_or_echo() {
  print_command "$@"
  if ! is_truthy "${DRY_RUN}"; then
    "$@"
  fi
}

activate_conda() {
  if is_truthy "${DRY_RUN}" || [[ -z "${CONDA_ENV}" || "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    return
  fi
  if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "Cannot find conda. Set CONDA_ROOT or activate ${CONDA_ENV} manually." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV}"
}

resolve_model
ATTN_TAG="$(sanitize_component "${ATTN_TYPE}")"
SELECTOR_TAG="$(sanitize_component "${COMETKV_SELECTOR}")"
RUN_NAME="${RUN_NAME:-budget${BUDGET}_sink${SINK}_recent${RECENT}_${ATTN_TAG}_selector${SELECTOR_TAG}_fwe_latency}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/latency}"
RUN_DIR="${RUN_DIR:-${RESULT_ROOT}/${MODEL_TAG}/${RUN_NAME}}"
OUTPUT_DIR="${RUN_DIR}/raw"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

LENGTHS_CSV="$(space_list_to_csv "${LENGTHS}")"
BATCH_SIZES_CSV="$(space_list_to_csv "${BATCH_SIZES}")"

write_run_config() {
  export RUN_CONFIG_JSON="${RUN_DIR}/run_config.json"
  export REPO_ROOT CONDA_ENV DATA_ROOT RULER_DATA_DIR GPU_ID CUDA_VISIBLE_DEVICES DEVICE
  export MODEL_NAME MODEL_KEY MODEL_TAG CONFIG_MODEL_NAME MODEL_PATH
  export BUDGET SINK RECENT ATTN_TYPE COMETKV_SELECTOR MAX_SAMPLES LENGTHS DTYPE
  export BATCH_SIZES MAX_NEW_LENGTH IGNORE_FIRST_STEPS PREFILL_BSZ PREFILL_METHOD DATA_PATH
  export COMETKV_MIN_RETRIEVAL_TOPK COMETKV_TOKEN_CACHE_SIZE SIG_BITS SIG_SEED SIG_CHUNK_SIZE
  export TEMPERATURE DO_SAMPLE RUN_NAME RUN_DIR OUTPUT_DIR LOG_FILE DRY_RUN
  python3 - <<'PY'
import json
import os
from datetime import datetime, timezone

keys = [
    "REPO_ROOT", "CONDA_ENV", "DATA_ROOT", "RULER_DATA_DIR", "GPU_ID",
    "CUDA_VISIBLE_DEVICES", "DEVICE", "MODEL_NAME", "MODEL_KEY", "MODEL_TAG",
    "CONFIG_MODEL_NAME", "MODEL_PATH", "BUDGET", "SINK", "RECENT", "ATTN_TYPE",
    "COMETKV_SELECTOR", "MAX_SAMPLES", "LENGTHS", "DTYPE", "BATCH_SIZES",
    "MAX_NEW_LENGTH", "IGNORE_FIRST_STEPS", "PREFILL_BSZ", "PREFILL_METHOD",
    "DATA_PATH", "COMETKV_MIN_RETRIEVAL_TOPK", "COMETKV_TOKEN_CACHE_SIZE",
    "SIG_BITS", "SIG_SEED", "SIG_CHUNK_SIZE", "TEMPERATURE", "DO_SAMPLE",
    "RUN_NAME", "RUN_DIR", "OUTPUT_DIR", "LOG_FILE", "DRY_RUN",
]
config = {key.lower(): os.environ.get(key, "") for key in keys}
config.update(
    {
        "benchmark": "latency",
        "task": "fwe",
        "budget_includes_sink_recent": True,
        "greedy": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
)
with open(os.environ["RUN_CONFIG_JSON"], "w", encoding="utf-8") as handle:
    json.dump(config, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
}

# MODEL_PATH may be a local directory or a HuggingFace hub id.
if [[ "${MODEL_PATH}" == /* && ! -d "${MODEL_PATH}" ]]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Latency data path not found: ${DATA_PATH}" >&2
  exit 1
fi

write_run_config
activate_conda

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export COMETKV_SELECTOR
export COMETKV_EVENT_PROFILE=0
PYTHONPATH_VALUE="${REPO_ROOT}:${REPO_ROOT}/library/cometkv:${PYTHONPATH:-}"

if [[ "${ATTN_TYPE}" != "CometKV" ]]; then
  echo "[WARN] benchmark/ruler/bench_cometkv_fwe_sweep.py is CometKV-specific; ATTN_TYPE=${ATTN_TYPE} is recorded but not used by this latency harness." >&2
fi

echo "[INFO] CometKV latency"
echo "[INFO] run_dir=${RUN_DIR}"
echo "[INFO] model=${MODEL_KEY} model_path=${MODEL_PATH}"
echo "[INFO] budget=${BUDGET} sink=${SINK} recent=${RECENT} includes_sink_recent=true"
echo "[INFO] lengths=${LENGTHS} batch_sizes=${BATCH_SIZES} selector=${COMETKV_SELECTOR}"

LATENCY_CMD=(
  env
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "PYTHONPATH=${PYTHONPATH_VALUE}"
  "TOKENIZERS_PARALLELISM=false"
  "COMETKV_SELECTOR=${COMETKV_SELECTOR}"
  "COMETKV_EVENT_PROFILE=0"
  python -u benchmark/ruler/bench_cometkv_fwe_sweep.py
  --data_path "${DATA_PATH}"
  --model_path "${MODEL_PATH}"
  --config_model_name "${CONFIG_MODEL_NAME}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --lengths "${LENGTHS_CSV}"
  --batch_sizes "${BATCH_SIZES_CSV}"
  --max_new_length "${MAX_NEW_LENGTH}"
  --ignore_first_steps "${IGNORE_FIRST_STEPS}"
  --prefill_bsz "${PREFILL_BSZ}"
  --prefill_method "${PREFILL_METHOD}"
  --retrieval_budget "${BUDGET}"
  --sig_bits "${SIG_BITS}"
  --sig_seed "${SIG_SEED}"
  --sig_chunk_size "${SIG_CHUNK_SIZE}"
  --cometkv_min_retrieval_topk "${COMETKV_MIN_RETRIEVAL_TOPK}"
  --cometkv_token_cache_size "${COMETKV_TOKEN_CACHE_SIZE}"
  --cometkv_static_pattern_start "${SINK}"
  --cometkv_static_pattern_end "${RECENT}"
  --cometkv_include_preserved_in_budget
  --output_dir "${OUTPUT_DIR}"
  --csv_name raw_latency.csv
  --jsonl_name raw_latency.jsonl
)
run_or_echo "${LATENCY_CMD[@]}"

echo "[INFO] Latency results: ${RUN_DIR}"
