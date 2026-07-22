#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATH_CONFIG_PYTHON="${PATH_CONFIG_PYTHON:-python3}"
LLAMA_GRADIENT_MODEL="${LLAMA_GRADIENT_MODEL:-$("${PATH_CONFIG_PYTHON}" "${PROJECT_ROOT}/config/paths.py" model-path llama-3-8b-1048k)}"

mkdir -p "${SCRIPT_DIR}/cometkv_logs"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-30000 60000 120000}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4}"
RETRIEVAL_BUDGET="${RETRIEVAL_BUDGET:-0.02}"

for context_len in ${CONTEXT_LENGTHS}; do
    for bsz in ${BATCH_SIZES}; do
        echo "CometKV throughput: context=${context_len}, batch=${bsz}"
        numactl --cpunodebind=0 --membind=0 python -u test.py \
            --model_name "${LLAMA_GRADIENT_MODEL}" \
            --attn_type CometKV \
            --retrieval_budget "${RETRIEVAL_BUDGET}" \
            --context_len "${context_len}" \
            --task_name NIAH \
            --batch_size "${bsz}" \
            > "cometkv_logs/context${context_len}_bsz${bsz}.log" 2>&1
    done
done

echo "Done. Logs are in ${SCRIPT_DIR}/cometkv_logs"
