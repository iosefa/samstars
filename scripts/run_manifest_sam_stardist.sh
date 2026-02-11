#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_manifest_sam_stardist.sh \
    --tiles-dir /abs/path/run/tiles \
    --out-dir /abs/path/run_manifest \
    [options]

Required:
  --tiles-dir PATH            Tile directory (must contain tile_*/img.tif,chm.tif,cost.tif,seeds.gpkg)
  --out-dir PATH              Output root for manifests + SAM probs + StarDist preds

Optional:
  --python BIN                Python executable (default: python3)
  --size-id ID                Size id folder (default: 0474)
  --device DEV                SAM device (default: mps)
  --ckpt-root PATH            SAM ckpt root (default: data/chips_128_100_full)
  --stardist-model-dir PATH   StarDist model dir (default: data/stardist_index/0474)
  --stardist-model-name NAME  StarDist model name (default: stardist)
  --stardist-input-key KEY    StarDist manifest input key: index|file_name|logit (default: index)
  --stardist-out-subdir NAME  Output subdir under --out-dir (default: stardist_preds)
  --stardist-preds-name NAME  Output GPKG filename (default: preds.gpkg)
  --scale-stats PATH          Scale stats for CHM aux (default: data/pld-chips/chips_128_100/scale_2_98.json)
  --sam-only                  Run SAM only
  --stardist-only             Run StarDist only (assumes SAM probs already exist)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="python3"
TILES_DIR=""
OUT_DIR=""
SIZE_ID="0474"
DEVICE="mps"
CKPT_ROOT="$ROOT_DIR/data/chips_128_100_full"
STARDIST_MODEL_DIR="$ROOT_DIR/data/stardist_index/0474"
STARDIST_MODEL_NAME="stardist"
STARDIST_INPUT_KEY="index"
STARDIST_OUT_SUBDIR="stardist_preds"
STARDIST_PREDS_NAME="preds.gpkg"
SCALE_STATS="$ROOT_DIR/data/pld-chips/chips_128_100/scale_2_98.json"
SAM_ONLY="false"
STARDIST_ONLY="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tiles-dir) TILES_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --size-id) SIZE_ID="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
    --stardist-model-dir) STARDIST_MODEL_DIR="$2"; shift 2 ;;
    --stardist-model-name) STARDIST_MODEL_NAME="$2"; shift 2 ;;
    --stardist-input-key) STARDIST_INPUT_KEY="$2"; shift 2 ;;
    --stardist-out-subdir) STARDIST_OUT_SUBDIR="$2"; shift 2 ;;
    --stardist-preds-name) STARDIST_PREDS_NAME="$2"; shift 2 ;;
    --scale-stats) SCALE_STATS="$2"; shift 2 ;;
    --sam-only) SAM_ONLY="true"; shift ;;
    --stardist-only) STARDIST_ONLY="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$TILES_DIR" || -z "$OUT_DIR" ]]; then
  usage
  exit 1
fi
if [[ "$SAM_ONLY" == "true" && "$STARDIST_ONLY" == "true" ]]; then
  echo "Use only one of --sam-only or --stardist-only." >&2
  exit 1
fi

MANIFESTS_ROOT="$OUT_DIR/manifests"
SAM_OUT_ROOT="$OUT_DIR/sam_probs"
STARDIST_OUT_DIR="$OUT_DIR/$STARDIST_OUT_SUBDIR"
PROBS_JSON="$SAM_OUT_ROOT/$SIZE_ID/train_probs.json"
TRAIN_PROBS_DIR="$SAM_OUT_ROOT/$SIZE_ID/train_probs"

if [[ "$STARDIST_ONLY" != "true" ]]; then
  echo "==> Build manifests"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/build_inference_manifests_from_tiles.py" \
    --tiles-dir "$TILES_DIR" \
    --out-root "$MANIFESTS_ROOT" \
    --size-id "$SIZE_ID" \
    --split train

  echo "==> SAM manifest inference"
  "$PYTHON_BIN" "$ROOT_DIR/pipelines/sam/predict_sam_from_manifests.py" \
    --manifests-root "$MANIFESTS_ROOT" \
    --ckpt-root "$CKPT_ROOT" \
    --out-root "$SAM_OUT_ROOT" \
    --sizes "$SIZE_ID" \
    --splits train \
    --device "$DEVICE" \
    --fast
fi

if [[ "$SAM_ONLY" == "true" ]]; then
  echo "Done (SAM only): $SAM_OUT_ROOT/$SIZE_ID/train_probs.json"
  exit 0
fi

if [[ ! -f "$PROBS_JSON" ]]; then
  if [[ -d "$TRAIN_PROBS_DIR" ]]; then
    echo "==> train_probs.json missing; building partial manifest from completed SAM outputs"
    "$PYTHON_BIN" - "$TRAIN_PROBS_DIR" "$PROBS_JSON" <<'PY'
import json, sys
from pathlib import Path

train_dir = Path(sys.argv[1])
out_json = Path(sys.argv[2])

recs = []
for idx in sorted(train_dir.glob("*_sam_index.tif")):
    stem = idx.stem
    if not stem.endswith("_sam_index"):
        continue
    base = stem[:-10]  # drop "_sam_index"
    prob = train_dir / f"{base}_sam_prob.tif"
    chm = train_dir / f"{base}_chm.tif"
    cost = train_dir / f"{base}_cost.tif"
    logit = train_dir / f"{base}_logit.tif"
    if not (prob.exists() and chm.exists() and cost.exists() and logit.exists()):
        continue
    recs.append(
        {
            "file_name": str(prob),
            "chm": str(chm),
            "cost": str(cost),
            "logit": str(logit),
            "index": str(idx),
            "mask": "",
            "meta": {},
        }
    )

if not recs:
    raise SystemExit(f"No complete SAM records found under {train_dir}")

out_json.parent.mkdir(parents=True, exist_ok=True)
json.dump(recs, open(out_json, "w"))
print(f"✓ wrote partial probs manifest: {out_json} ({len(recs)} records)")
PY
  else
    echo "Missing probs manifest: $PROBS_JSON" >&2
    exit 1
  fi
fi

echo "==> StarDist manifest inference"
"$PYTHON_BIN" "$ROOT_DIR/pipelines/stardist/predict_stardist_from_probs.py" \
  --probs-json "$PROBS_JSON" \
  --model-dir "$STARDIST_MODEL_DIR" \
  --model-name "$STARDIST_MODEL_NAME" \
  --out-dir "$STARDIST_OUT_DIR" \
  --input-key "$STARDIST_INPUT_KEY" \
  --use-aux \
  --scale-stats "$SCALE_STATS" \
  --preds-name "$STARDIST_PREDS_NAME" \
  --write-labels

echo "Done."
echo "  SAM probs:   $SAM_OUT_ROOT/$SIZE_ID"
echo "  StarDist:    $STARDIST_OUT_DIR/preds.gpkg"
echo "  Labels dir:  $STARDIST_OUT_DIR/labels"
