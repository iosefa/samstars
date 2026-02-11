#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 SIZE [RUNS] [WARMUP]" >&2
  exit 1
fi

SIZE="$1"
RUNS="${2:-3}"
WARMUP="${3:-1}"

VAL_PROBS="${VAL_PROBS:-data/sam_probs_fixed_val/${SIZE}/val_probs.json}"
MODEL_DIR="${MODEL_DIR:-data/stardist_index/${SIZE}}"
OUT_DIR="${OUT_DIR:-data/stardist_eval/${SIZE}}"
INPUT_KEY="${INPUT_KEY:-index}"
SCALE_STATS="${SCALE_STATS:-data/chips_128_100/sam_scale_2_98.json}"
USE_AUX="${USE_AUX:-1}"
OUT_LOG_DIR="${OUT_LOG_DIR:-data/timing}"
LOG_PREFIX="${LOG_PREFIX:-stardist_eval_${SIZE}}"

mkdir -p "${OUT_LOG_DIR}"

aux_args=()
if [[ "${USE_AUX}" -eq 1 ]]; then
  aux_args+=(--use-aux --scale-stats "${SCALE_STATS}")
fi

echo "[stardist] size=${SIZE} runs=${RUNS} warmup=${WARMUP} input_key=${INPUT_KEY}"

for i in $(seq 0 "${WARMUP}"); do
  if [[ "${i}" -eq 0 && "${WARMUP}" -gt 0 ]]; then
    log="${OUT_LOG_DIR}/${LOG_PREFIX}_run0.log"
  else
    break
  fi
  /usr/bin/time -p python pipelines/stardist/eval_stardist_on_probs.py \
    --val-probs "${VAL_PROBS}" \
    --model-dir "${MODEL_DIR}" \
    --out-dir "${OUT_DIR}" \
    --input-key "${INPUT_KEY}" \
    "${aux_args[@]}" \
    2>&1 | tee "${log}"
done

for i in $(seq 1 "${RUNS}"); do
  log="${OUT_LOG_DIR}/${LOG_PREFIX}_run${i}.log"
  /usr/bin/time -p python pipelines/stardist/eval_stardist_on_probs.py \
    --val-probs "${VAL_PROBS}" \
    --model-dir "${MODEL_DIR}" \
    --out-dir "${OUT_DIR}" \
    --input-key "${INPUT_KEY}" \
    "${aux_args[@]}" \
    2>&1 | tee "${log}"
done
