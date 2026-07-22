#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-cometkv}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
CONFIG_MODEL_NAME="${CONFIG_MODEL_NAME:-Llama-3.1-8B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LENGTHS="${LENGTHS:-32k,64k,96k}"
MAX_NEW_LENGTH="${MAX_NEW_LENGTH:-256}"
IGNORE_FIRST_STEPS="${IGNORE_FIRST_STEPS:-1}"
RETRIEVAL_BUDGET="${RETRIEVAL_BUDGET:-0.02}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmark/ruler/speed_results/latency}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/test_data/fwe.json}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/library/cometkv:${PYTHONPATH:-}"
export COMETKV_EVENT_PROFILE=0

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -u benchmark/ruler/bench_cometkv_fwe_latency.py \
  --data_path "${DATA_PATH}" \
  --model_path "${MODEL_PATH}" \
  --config_model_name "${CONFIG_MODEL_NAME}" \
  --lengths "${LENGTHS}" \
  --batch_sizes "${BATCH_SIZE}" \
  --max_new_length "${MAX_NEW_LENGTH}" \
  --ignore_first_steps "${IGNORE_FIRST_STEPS}" \
  --retrieval_budget "${RETRIEVAL_BUDGET}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
