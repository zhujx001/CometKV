#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [ $# -ne 7 ]; then
    echo "Usage: $0 <model_name> <prefill_method> <attn_type> <context_length> <task> <dtype> <budget_ratio>"
    echo "Optional env: CUDA_VISIBLE_DEVICES, RULER_EXTRA_PRED_ARGS"
    exit 1
fi

print_command() {
    printf 'RUN:'
    for arg in "$@"; do
        printf ' %q' "$arg"
    done
    printf '\n'
}

run_or_echo() {
    if [ "${RULER_DRY_RUN:-0}" = "1" ]; then
        print_command "$@"
    else
        "$@"
    fi
}

clean_task_predictions() {
    local pred_dir=$1
    local task_name=$2
    local task_file="${pred_dir}/${task_name}.jsonl"
    local task_chunk_glob="${pred_dir}/${task_name}-*.jsonl"

    if [ "${RULER_DRY_RUN:-0}" = "1" ]; then
        echo "CLEAN_PRED_FILE: ${task_file}"
        echo "CLEAN_PRED_GLOB: ${task_chunk_glob}"
    else
        rm -f "${task_file}"
        rm -f "${pred_dir}/${task_name}-"*.jsonl
    fi
}

run_single_task() {
    local task_name=$1
    local task_index=$2
    local task_total=$3

    echo "===== [${task_index}/${task_total}] RUN ${task_name} ====="

    local task_dir="${DATA_DIR}/${task_name}"
    local data_file="${task_dir}/${SUBSET}.jsonl"

    if [ "${RULER_USE_EXISTING_DATA:-0}" = "1" ]; then
        if [ -f "${data_file}" ]; then
            echo "DATA_HIT: ${data_file}"
            echo "SKIP_PREPARE"
        else
            echo "DATA_MISS: ${data_file}" >&2
            echo "RULER_USE_EXISTING_DATA=1 forbids regenerating RULER data." >&2
            exit 1
        fi
    else
        CACHE_CHECK_CMD=(
            python3 -u data/cache_metadata.py
            --task_dir "${task_dir}"
            --benchmark "${BENCHMARK}"
            --task "${task_name}"
            --subset "${SUBSET}"
            --num_samples "${NUM_SAMPLES}"
            --max_seq_length "${MAX_SEQ_LENGTH}"
            --tokenizer_path "${TOKENIZER_PATH}"
            --tokenizer_type "${TOKENIZER_TYPE}"
            --model_template_type "${MODEL_TEMPLATE_TYPE}"
        )

        if "${CACHE_CHECK_CMD[@]}"; then
        echo "CACHE_HIT: ${task_dir}"
        echo "SKIP_PREPARE"
        else
            echo "CACHE_MISS: ${task_dir}"
            if [ "${RULER_DRY_RUN:-0}" = "1" ]; then
                echo "CLEAN_TASK_DIR: ${task_dir}"
            else
                rm -rf "${task_dir}"
            fi

            PREPARE_CMD=(
                python -u data/prepare.py
                --save_dir "${DATA_DIR}"
                --benchmark "${BENCHMARK}"
                --task "${task_name}"
                --tokenizer_path "${TOKENIZER_PATH}"
                --tokenizer_type "${TOKENIZER_TYPE}"
                --max_seq_length "${MAX_SEQ_LENGTH}"
                --model_template_type "${MODEL_TEMPLATE_TYPE}"
                --num_samples "${NUM_SAMPLES}"
            )
            if [ -n "${REMOVE_NEWLINE_TAB}" ]; then
                PREPARE_CMD+=("${REMOVE_NEWLINE_TAB}")
            fi
            run_or_echo "${PREPARE_CMD[@]}"
        fi
    fi

    if [ "${RULER_DRY_RUN:-0}" = "1" ]; then
        echo "ENSURE_PRED_DIR: ${PRED_DIR}"
    else
        mkdir -p "${PRED_DIR}"
    fi

    PREDICT_CMD=(
        python -u pred/call_api.py
        --model_name "${MODEL_NAME}"
        --attn_type "${ATTN_TYPE}"
        --max_len "${MAX_SEQ_LENGTH}"
        --batch_size 1
        --max_samples "${NUM_SAMPLES}"
        --data_dir "${DATA_DIR}"
        --save_dir "${PRED_DIR}"
        --benchmark "${BENCHMARK}"
        --task "${task_name}"
        --dtype "${DTYPE}"
        --server_type "${MODEL_FRAMEWORK}"
        --device "${DEVICE}"
        --retrieval_budget "${BUDGET_RATIO}"
        --synthetic_len "${MAX_SEQ_LENGTH}"
        --prefill_method "${PREFILL_METHOD}"
    )
    if [ -n "${RULER_EXTRA_PRED_ARGS:-}" ]; then
        read -r -a EXTRA_PRED_ARGS <<< "${RULER_EXTRA_PRED_ARGS}"
        PREDICT_CMD+=("${EXTRA_PRED_ARGS[@]}")
    fi

    local PRED_FILE="${PRED_DIR}/${task_name}.jsonl"
    local TASK_FILE="${data_file}"
    if [ -f "${PRED_FILE}" ]; then
        local PRED_LINES
        local TASK_LINES
        PRED_LINES=$(wc -l < "${PRED_FILE}")
        TASK_LINES=$(wc -l < "${TASK_FILE}")
        if [ "${PRED_LINES}" -ge "${TASK_LINES}" ]; then
            echo "SKIP_PRED: ${task_name} already complete (${PRED_LINES}/${TASK_LINES})"
        else
            echo "RESUME_PRED: ${task_name} incomplete (${PRED_LINES}/${TASK_LINES}), re-running"
            clean_task_predictions "${PRED_DIR}" "${task_name}"
            run_or_echo "${PREDICT_CMD[@]}"
        fi
    else
        clean_task_predictions "${PRED_DIR}" "${task_name}"
        run_or_echo "${PREDICT_CMD[@]}"
    fi

    EVAL_CMD=(
        python -u eval/evaluate.py
        --data_dir "${PRED_DIR}"
        --benchmark "${BENCHMARK}"
    )
    run_or_echo "${EVAL_CMD[@]}"
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Root directories
ROOT_DIR=${ROOT_DIR:-"./ruler_eval_result"}
DATA_CACHE_DIR=${DATA_CACHE_DIR:-"./ruler_dataset_cache"}

NUM_SAMPLES=${NUM_SAMPLES:-50}
MAX_SEQ_LENGTH=${4}
ATTN_TYPE=${3}
if [ "${ATTN_TYPE}" != "CometKV" ] && [ "${ATTN_TYPE}" != "CometKV_GPU" ] && [ "${ATTN_TYPE}" != "Full_Flash_Attn" ]; then
    echo "Unsupported attention type: ${ATTN_TYPE}" >&2
    exit 1
fi
DEVICE=auto
BUDGET_RATIO=${7}
PREFILL_METHOD=${2}
SUBSET=validation

# Model and tokenizer
source ruler_config_models.sh
INPUT_MODEL_NAME=${1}
MODEL_CONFIG=$(MODEL_SELECT "${INPUT_MODEL_NAME}")
IFS=":" read -r MODEL_NAME MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE <<< "${MODEL_CONFIG}"
if [ -z "${MODEL_NAME}" ]; then
    echo "Model: ${INPUT_MODEL_NAME} is not supported"
    exit 1
fi

# Benchmark and tasks
source ruler_config_tasks.sh
BENCHMARK=synthetic
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

TASK=${5}
DATA_DIR="${RULER_DATA_DIR:-${DATA_CACHE_DIR}/${BENCHMARK}/${MAX_SEQ_LENGTH}}"
RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}/${ATTN_TYPE}"
PRED_DIR="${RESULTS_DIR}/pred"

DTYPE=${6}

if [ "${TASK}" = "ALL" ]; then
    TASK_LIST=("${TASKS[@]}")
else
    TASK_LIST=("${TASK}")
fi

TASK_TOTAL=${#TASK_LIST[@]}
for ((task_idx=0; task_idx<TASK_TOTAL; task_idx++)); do
    run_single_task "${TASK_LIST[$task_idx]}" "$((task_idx + 1))" "${TASK_TOTAL}"
done
