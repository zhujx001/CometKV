#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-cometkv}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_NAME="${MODEL_NAME:-llama-3.1-8b}"
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
DTYPE="${DTYPE:-bf16}"
HASH_DEVICE="${HASH_DEVICE:-cuda:0}"
PREFILL_METHOD="${PREFILL_METHOD:-full}"

LONG_BENCH_DATA_DIR="${LONG_BENCH_DATA_DIR:-${DATA_ROOT}/longbench-jsonl}"
RULER_DATA_ROOT="${RULER_DATA_ROOT:-${DATA_ROOT}/RULER}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/benchmark/results/cometkv_grid_$(date +%Y%m%d_%H%M%S)}"

SINK_VALUES="${SINK_VALUES:-32}"
RECENT_VALUES="${RECENT_VALUES:-64}"
BUDGETS="${BUDGETS:-0.05 0.10}"
EXCLUDE_PRESERVED_FROM_BUDGET="${EXCLUDE_PRESERVED_FROM_BUDGET:-1}"

RUN_LONGBENCH="${RUN_LONGBENCH:-1}"
RUN_RULER="${RUN_RULER:-1}"
LONG_BENCH_TASKS="${LONG_BENCH_TASKS:-qasper}"
LONG_BENCH_NUM_EXAMPLES="${LONG_BENCH_NUM_EXAMPLES:--1}"
RULER_TASKS="${RULER_TASKS:-fwe}"
RULER_CONTEXT_LENGTHS="${RULER_CONTEXT_LENGTHS:-32768}"
RULER_NUM_SAMPLES="${RULER_NUM_SAMPLES:-50}"
RESUME="${RESUME:-0}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/library/cometkv:${PYTHONPATH:-}"
export COMETKV_EVENT_PROFILE=0

budget_label() {
  local value="$1"
  value="${value/#0./0p}"
  value="${value//./p}"
  printf '%s' "${value}"
}

print_command() {
  printf 'RUN:'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

run_or_echo_longbench() {
  print_command "$@"
  if [ "${LONG_BENCH_DRY_RUN:-0}" != "1" ]; then
    "$@"
  fi
}

run_or_echo_ruler() {
  print_command "$@"
  if [ "${RULER_DRY_RUN:-0}" != "1" ]; then
    "$@"
  fi
}

run_longbench_task() {
  local task="$1"
  local budget="$2"
  local exp_dir="$3"
  local extra_args=(
    --cometkv_static_pattern_start "${sink}"
    --cometkv_static_pattern_end "${recent}"
    --cometkv_min_retrieval_topk 16
  )
  if [ "${EXCLUDE_PRESERVED_FROM_BUDGET}" = "1" ]; then
    extra_args+=(--cometkv_exclude_preserved_from_budget)
  else
    extra_args+=(--cometkv_include_preserved_in_budget)
  fi

  local pred_args=(
    --task "${task}"
    --attn_type CometKV
    --model "${MODEL_NAME}"
    --model_path "${MODEL_PATH}"
    --data_dir "${LONG_BENCH_DATA_DIR}"
    --dtype "${DTYPE}"
    --device "${HASH_DEVICE}"
    --retrieval_budget "${budget}"
    --num_examples "${LONG_BENCH_NUM_EXAMPLES}"
    "${extra_args[@]}"
  )
  local eval_args=(
    --attn_type CometKV
    --model "${MODEL_NAME}"
    --task "${task}"
  )

  mkdir -p "${exp_dir}/longbench"
  (
    cd "${REPO_ROOT}/benchmark/longbench"
    run_or_echo_longbench "${PYTHON_BIN}" -u pred.py "${pred_args[@]}"
    run_or_echo_longbench "${PYTHON_BIN}" -u eval.py "${eval_args[@]}"
    if [ "${LONG_BENCH_DRY_RUN:-0}" != "1" ]; then
      local result_dir="${REPO_ROOT}/benchmark/longbench/results/pred/${MODEL_NAME}/CometKV"
      cp -f "${result_dir}/result.json" "${exp_dir}/longbench/result_${task}.json" 2>/dev/null || true
      cp -f "${result_dir}/${task}.jsonl" "${exp_dir}/longbench/" 2>/dev/null || true
    fi
  )
}

run_ruler_task() {
  local task="$1"
  local context_len="$2"
  local budget="$3"
  local exp_dir="$4"
  local ruler_data_dir="${RULER_DATA_ROOT%/}/${context_len}"
  local extra_args=(
    --cometkv_static_pattern_start "${sink}"
    --cometkv_static_pattern_end "${recent}"
    --cometkv_min_retrieval_topk 16
  )
  if [ "${EXCLUDE_PRESERVED_FROM_BUDGET}" = "1" ]; then
    extra_args+=(--cometkv_exclude_preserved_from_budget)
  else
    extra_args+=(--cometkv_include_preserved_in_budget)
  fi

  mkdir -p "${exp_dir}/ruler"
  (
    cd "${REPO_ROOT}/benchmark/ruler"
    run_or_echo_ruler env \
      RULER_USE_EXISTING_DATA=1 \
      RULER_DATA_DIR="${ruler_data_dir}" \
      ROOT_DIR="${exp_dir}/ruler" \
      NUM_SAMPLES="${RULER_NUM_SAMPLES}" \
      RULER_EXTRA_PRED_ARGS="${extra_args[*]}" \
      bash ruler_run.sh \
      "${MODEL_NAME}" "${PREFILL_METHOD}" CometKV "${context_len}" "${task}" "${DTYPE}" "${budget}"
  )
}

mkdir -p "${RESULT_ROOT}"
echo "RESULT_ROOT=${RESULT_ROOT}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "LONG_BENCH_DATA_DIR=${LONG_BENCH_DATA_DIR}"
echo "RULER_DATA_ROOT=${RULER_DATA_ROOT}"
echo "Preserved tokens excluded from budget: ${EXCLUDE_PRESERVED_FROM_BUDGET}"

for sink in ${SINK_VALUES}; do
  for recent in ${RECENT_VALUES}; do
    for budget in ${BUDGETS}; do
      label="sink${sink}_recent${recent}_budget$(budget_label "${budget}")"
      exp_dir="${RESULT_ROOT}/${label}"
      mkdir -p "${exp_dir}"
      cat > "${exp_dir}/params.txt" <<PARAMS
model_name=${MODEL_NAME}
model_path=${MODEL_PATH}
sink=${sink}
recent=${recent}
retrieval_budget=${budget}
exclude_preserved_from_budget=${EXCLUDE_PRESERVED_FROM_BUDGET}
longbench_data_dir=${LONG_BENCH_DATA_DIR}
ruler_data_root=${RULER_DATA_ROOT}
PARAMS
      echo "BEGIN_EXPERIMENT ${label}"

      if [ "${RUN_LONGBENCH}" = "1" ]; then
        for task in ${LONG_BENCH_TASKS}; do
          echo "LONG_BENCH ${label} task=${task}"
          run_longbench_task "${task}" "${budget}" "${exp_dir}"
        done
      fi

      if [ "${RUN_RULER}" = "1" ]; then
        for context_len in ${RULER_CONTEXT_LENGTHS}; do
          for task in ${RULER_TASKS}; do
            echo "RULER ${label} context=${context_len} task=${task}"
            run_ruler_task "${task}" "${context_len}" "${budget}" "${exp_dir}"
          done
        done
      fi

      echo "END_EXPERIMENT ${label}"
    done
  done
done

echo "All benchmark results under ${RESULT_ROOT}"
