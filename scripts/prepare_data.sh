#!/usr/bin/env bash
set -euo pipefail

# One-click dataset preparation for the CometKV benchmarks.
#
#   bash scripts/prepare_data.sh                       # LongBench + RULER into $DATA_ROOT
#   DATA_ROOT=/my/data bash scripts/prepare_data.sh    # custom target root
#   PREPARE_LONGBENCH=0 TASKS="fwe vt" RULER_LENGTHS="32768" bash scripts/prepare_data.sh
#
# Produces the exact layouts scripts/run_longbench.sh and scripts/run_ruler.sh expect:
#   $DATA_ROOT/longbench-jsonl/<task>.jsonl
#   $DATA_ROOT/RULER/<len>/<task>/validation.jsonl
# Every step is idempotent: existing files are kept, only missing pieces are fetched
# or generated. Network downloads honor HTTP_PROXY/HTTPS_PROXY and HF_ENDPOINT.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
LONGBENCH_DIR="${LONGBENCH_DIR:-${DATA_ROOT}/longbench-jsonl}"
RULER_DATA_DIR="${RULER_DATA_DIR:-${DATA_ROOT}/RULER}"
PREPARE_LONGBENCH="${PREPARE_LONGBENCH:-1}"
PREPARE_RULER="${PREPARE_RULER:-1}"

# Tokenizer used to calibrate RULER sequence lengths — must match the model you evaluate.
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_TEMPLATE_TYPE="${MODEL_TEMPLATE_TYPE:-meta-chat}"
RULER_LENGTHS="${RULER_LENGTHS:-32768 65536 98304}"
TASKS="${TASKS:-niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"

CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-cometkv}"
DRY_RUN="${DRY_RUN:-0}"

HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
LONGBENCH_URL="${LONGBENCH_URL:-${HF_ENDPOINT}/datasets/THUDM/LongBench/resolve/main/data.zip}"

log() { echo "[prepare_data] $*"; }
run() {
  if [[ "${DRY_RUN}" == "1" ]]; then echo "RUN: $*"; else "$@"; fi
}

if [[ "${DRY_RUN}" != "1" && "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
  if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
  fi
fi

# ---------- 1. LongBench ----------
if [[ "${PREPARE_LONGBENCH}" == "1" ]]; then
  if compgen -G "${LONGBENCH_DIR}/*.jsonl" > /dev/null; then
    log "LongBench: ${LONGBENCH_DIR} already populated — skip."
  else
    log "LongBench: downloading ${LONGBENCH_URL}"
    run mkdir -p "${LONGBENCH_DIR}"
    TMP_DIR="$(mktemp -d)"
    run wget -q --show-progress -O "${TMP_DIR}/data.zip" "${LONGBENCH_URL}"
    run unzip -q "${TMP_DIR}/data.zip" -d "${TMP_DIR}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      find "${TMP_DIR}" -name '*.jsonl' -exec mv {} "${LONGBENCH_DIR}/" \;
      rm -rf "${TMP_DIR}"
      log "LongBench: $(ls "${LONGBENCH_DIR}" | wc -l) jsonl files in ${LONGBENCH_DIR}"
    fi
  fi
fi

# ---------- 2. RULER ----------
if [[ "${PREPARE_RULER}" == "1" ]]; then
  JSON_DIR="${REPO_ROOT}/benchmark/ruler/data/synthetic/json"
  SQUAD_URL="${SQUAD_URL:-https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json}"
  # NOTE: the upstream CMU server is flaky; override HOTPOTQA_URL with a mirror if it is down.
  HOTPOTQA_URL="${HOTPOTQA_URL:-http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json}"

  # fetch <url> <dest>: download via a temp file so failures never leave empty artifacts.
  fetch() {
    local url="$1" dest="$2"
    if [[ "${DRY_RUN}" == "1" ]]; then echo "RUN: wget -O ${dest} ${url}"; return 0; fi
    if ! wget -q --show-progress -O "${dest}.tmp" "${url}" || [[ ! -s "${dest}.tmp" ]]; then
      rm -f "${dest}.tmp"
      echo "[prepare_data] FAILED to download ${url} — check network/proxy or override the *_URL env." >&2
      return 1
    fi
    mv "${dest}.tmp" "${dest}"
  }

  # 2a. source corpora — downloaded only if a requested task needs them.
  if [[ " ${TASKS} " == *"niah"* && ! -s "${JSON_DIR}/PaulGrahamEssays.json" ]]; then
    log "RULER: downloading Paul Graham essays (niah haystack)"
    ( cd "${JSON_DIR}" && run python -u download_paulgraham_essay.py )
  fi
  if [[ " ${TASKS} " == *" qa_1 "* && ! -s "${JSON_DIR}/squad.json" ]]; then
    log "RULER: downloading SQuAD (qa_1)"
    fetch "${SQUAD_URL}" "${JSON_DIR}/squad.json"
  fi
  if [[ " ${TASKS} " == *" qa_2 "* && ! -s "${JSON_DIR}/hotpotqa.json" ]]; then
    log "RULER: downloading HotpotQA (qa_2)"
    fetch "${HOTPOTQA_URL}" "${JSON_DIR}/hotpotqa.json"
  fi
  run python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

  # 2b. generate task data: $RULER_DATA_DIR/<len>/<task>/validation.jsonl
  # MODEL_PATH may be a local directory or a HuggingFace hub id (used as tokenizer only).
  if [[ "${MODEL_PATH}" == /* && ! -d "${MODEL_PATH}" && "${DRY_RUN}" != "1" ]]; then
    echo "[prepare_data] MODEL_PATH not found: ${MODEL_PATH} (needed as RULER tokenizer)" >&2
    exit 1
  fi
  for seq_len in ${RULER_LENGTHS}; do
    for task in ${TASKS}; do
      out="${RULER_DATA_DIR}/${seq_len}/${task}/validation.jsonl"
      if [[ -f "${out}" ]]; then
        log "RULER: ${seq_len}/${task} exists — skip."
        continue
      fi
      log "RULER: generating ${task} @ ${seq_len} (${NUM_SAMPLES} samples)"
      ( cd "${REPO_ROOT}/benchmark/ruler" && run python -u data/prepare.py \
          --save_dir "${RULER_DATA_DIR}/${seq_len}" \
          --benchmark synthetic \
          --task "${task}" \
          --tokenizer_path "${MODEL_PATH}" \
          --tokenizer_type hf \
          --max_seq_length "${seq_len}" \
          --model_template_type "${MODEL_TEMPLATE_TYPE}" \
          --num_samples "${NUM_SAMPLES}" )
    done
  done
fi

log "Done. Run the benchmarks with:"
log "  DATA_ROOT=${DATA_ROOT} bash scripts/run_longbench.sh"
log "  DATA_ROOT=${DATA_ROOT} bash scripts/run_ruler.sh"
