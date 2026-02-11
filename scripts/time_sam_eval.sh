#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 SIZE [RUNS] [WARMUP]" >&2
  exit 1
fi

SIZE="$1"
RUNS="${2:-3}"
WARMUP="${3:-1}"

MANIFESTS_ROOT="${MANIFESTS_ROOT:-data/sam_manifests_fixed_val}"
CKPT_ROOT="${CKPT_ROOT:-data/chips_128_100}"
OUT_ROOT="${OUT_ROOT:-data/sam_probs_fixed_val}"
SCALE_STATS="${SCALE_STATS:-data/chips_128_100/sam_scale_2_98.json}"
DEVICE="${DEVICE:-mps}"
SPLITS="${SPLITS:-val}"
OUT_DIR="${OUT_DIR:-data/timing}"

FAST="${FAST:-1}"
FAST_BATCH="${FAST_BATCH:-64}"
LOG_PREFIX="${LOG_PREFIX:-sam_eval_${SIZE}_fast}"

mkdir -p "${OUT_DIR}"

fast_args=()
if [[ "${FAST}" -eq 1 ]]; then
  fast_args+=(--fast --fast-batch-size "${FAST_BATCH}")
fi

echo "[sam] size=${SIZE} runs=${RUNS} warmup=${WARMUP} fast=${FAST} batch=${FAST_BATCH}"

ckpt="${CKPT_ROOT}/${SIZE}/sam/sam_crowns_decoder.pth"
if [[ ! -f "${ckpt}" ]]; then
  echo "ERROR: missing SAM decoder checkpoint: ${ckpt}" >&2
  exit 2
fi

manifest="${MANIFESTS_ROOT}/${SIZE}/val_manifest.json"
if [[ ! -f "${manifest}" ]]; then
  echo "ERROR: missing manifest: ${manifest}" >&2
  exit 2
fi

for i in $(seq 0 "${WARMUP}"); do
  if [[ "${i}" -eq 0 && "${WARMUP}" -gt 0 ]]; then
    log="${OUT_DIR}/${LOG_PREFIX}_run0.log"
  else
    break
  fi
  /usr/bin/time -p python pipelines/sam/predict_sam_from_manifests.py \
    --manifests-root "${MANIFESTS_ROOT}" \
    --ckpt-root "${CKPT_ROOT}" \
    --out-root "${OUT_ROOT}" \
    --device "${DEVICE}" \
    --sizes "${SIZE}" \
    --splits "${SPLITS}" \
    --scale-stats "${SCALE_STATS}" \
    "${fast_args[@]}" \
    2>&1 | tee "${log}"
done

for i in $(seq 1 "${RUNS}"); do
  log="${OUT_DIR}/${LOG_PREFIX}_run${i}.log"
  /usr/bin/time -p python pipelines/sam/predict_sam_from_manifests.py \
    --manifests-root "${MANIFESTS_ROOT}" \
    --ckpt-root "${CKPT_ROOT}" \
    --out-root "${OUT_ROOT}" \
    --device "${DEVICE}" \
    --sizes "${SIZE}" \
    --splits "${SPLITS}" \
    --scale-stats "${SCALE_STATS}" \
    "${fast_args[@]}" \
    2>&1 | tee "${log}"
done
