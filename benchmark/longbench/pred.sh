# !/bin/bash

if [ $# -ne 5 ]; then
    echo "Usage: $0 <model_name> <task_name> <attn_type> <dtype> <budget_ratio>"
    exit 1
fi

NUM_EXAMPLES=-1
MODEL=${1}
TASK=${2}
ATTN_TYPE=${3}
DTYPE=${4}
BUDGET_RATIO=${5}

if [ "${ATTN_TYPE}" != "CometKV" ] && [ "${ATTN_TYPE}" != "CometKV_GPU" ] && [ "${ATTN_TYPE}" != "Full_Flash_Attn" ]; then
    echo "Unsupported attention type: ${ATTN_TYPE}" >&2
    exit 1
fi

RESULT_DIR="./results/pred/${MODEL}/${ATTN_TYPE}"
RESULT_DIR_E="./results/pred_e/${MODEL}/${ATTN_TYPE}"

MODEL_PATH_ARGS=()
DATA_DIR_ARGS=()
if [ -n "${MODEL_PATH:-}" ]; then
    MODEL_PATH_ARGS=(--model_path "${MODEL_PATH}")
fi
if [ -n "${DATA_DIR:-}" ]; then
    DATA_DIR_ARGS=(--data_dir "${DATA_DIR}")
fi

DEVICE_ARG="${DEVICE:-}"
if [ -z "${DEVICE_ARG}" ]; then
    # CometKV / CometKV_GPU are single-GPU (same-length lockstep); dense baseline can shard with auto.
    if [ "${ATTN_TYPE}" == "CometKV" ] || [ "${ATTN_TYPE}" == "CometKV_GPU" ]; then
        DEVICE_ARG="cuda:0"
    else
        DEVICE_ARG="auto"
    fi
fi

COMETKV_BUDGET_ARGS=()
if [ "${ATTN_TYPE}" != "Full_Flash_Attn" ] && [ "${COMETKV_INCLUDE_PRESERVED_IN_BUDGET:-0}" = "1" ]; then
    COMETKV_BUDGET_ARGS=(--cometkv_include_preserved_in_budget)
elif [ "${ATTN_TYPE}" != "Full_Flash_Attn" ] && [ "${COMETKV_EXCLUDE_PRESERVED_FROM_BUDGET:-0}" = "1" ]; then
    COMETKV_BUDGET_ARGS=(--cometkv_exclude_preserved_from_budget)
fi

echo "remove previous result file..."
rm -f "${RESULT_DIR}/${TASK}.jsonl"
rm -f "${RESULT_DIR_E}/${TASK}.jsonl"

echo "Start to predict..."
PRED_ARGS=(
    --task ${TASK}
    --attn_type ${ATTN_TYPE}
    --model ${MODEL}
    --dtype ${DTYPE}
    --device ${DEVICE_ARG}
    --retrieval_budget ${BUDGET_RATIO}
    --num_examples ${NUM_EXAMPLES}
    "${COMETKV_BUDGET_ARGS[@]}"
    "${MODEL_PATH_ARGS[@]}"
    "${DATA_DIR_ARGS[@]}"
)

if [ "${LONG_BENCH_DRY_RUN:-0}" = "1" ]; then
    printf 'python -u pred.py'
    for arg in "${PRED_ARGS[@]}"; do
        printf ' %s' "${arg}"
    done
    printf '\n'
    exit 0
fi

# numactl --cpunodebind=0,1 python -u pred.py \
python -u pred.py "${PRED_ARGS[@]}"
