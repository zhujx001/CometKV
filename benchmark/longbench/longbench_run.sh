# !/bin/bash

if [ $# -ne 5 ]; then
    echo "Usage: $0 <model> <attn_type> <budget_ratio> <dtype> <category>"
    exit 1
fi

MODEL=${1}
ATTN_TYPE=${2}
BUDGET_RATIO=${3}
DTYPE=${4}
CATEGORY=${5}

RESULT_DIR="./results/pred/${MODEL}/${ATTN_TYPE}"

ALL_TASKS=(
  qasper multifieldqa_en narrativeqa
  hotpotqa 2wikimqa musique dureader
  gov_report qmsum multi_news vcsum
  trec lsht samsum triviaqa
  passage_retrieval_en passage_count passage_retrieval_zh
  repobench-p lcc
)

if [ "$CATEGORY" == "SQA" ]; then
  tasks=(qasper multifieldqa_en narrativeqa)
elif [ "$CATEGORY" == "MQA" ]; then
  tasks=(hotpotqa 2wikimqa musique dureader)
elif [ "$CATEGORY" == "SUM" ]; then
  tasks=(gov_report qmsum multi_news vcsum)
elif [ "$CATEGORY" == "FSL" ]; then
  tasks=(trec lsht samsum triviaqa)
elif [ "$CATEGORY" == "ST" ]; then
  tasks=(passage_retrieval_en passage_count passage_retrieval_zh)
elif [ "$CATEGORY" == "CC" ]; then
  tasks=(repobench-p lcc)
elif [ "$CATEGORY" == "ALL" ]; then
  tasks=("${ALL_TASKS[@]}")
elif [[ " ${ALL_TASKS[*]} " == *" ${CATEGORY} "* ]]; then
  tasks=(${CATEGORY})
else
  echo "Unknown CATEGORY: $CATEGORY"
  tasks=()
fi

if [ ${#tasks[@]} -eq 0 ]; then
  exit 1
fi

if [ "${LONG_BENCH_DRY_RUN:-0}" = "1" ]; then
  for task in "${tasks[@]}"; do
    echo "Parameters: ${MODEL} ${task} ${ATTN_TYPE} ${DTYPE} ${BUDGET_RATIO}"
  done
  echo "DRY RUN: skip prediction and evaluation"
  exit 0
fi

for task in "${tasks[@]}"; do
    echo "Parameters: ${MODEL} ${task} ${ATTN_TYPE} ${DTYPE} ${BUDGET_RATIO}"
    bash pred.sh ${MODEL} ${task} ${ATTN_TYPE} ${DTYPE} ${BUDGET_RATIO}
done

echo "Start to evaluate..."
EVAL_ARGS=(
  --attn_type ${ATTN_TYPE}
  --model ${MODEL}
)

if [ ${#tasks[@]} -eq 1 ]; then
  EVAL_ARGS+=(--task "${tasks[0]}")
fi

python -u eval.py \
    "${EVAL_ARGS[@]}"

echo "Results:"
cat "${RESULT_DIR}/result.json"
