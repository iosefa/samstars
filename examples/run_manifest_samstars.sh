#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash examples/run_manifest_samstars.sh \
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
  --skip-existing-sam         Run SAM only on chips without existing sam_index/sam_prob/logit (default)
  --no-skip-existing-sam      Recompute SAM for every chip in manifest
  --crop-smooth-sigma FLOAT   Per-crop SAM logit blur sigma (default: 2)
  --smooth-sigma FLOAT        Global SAM probability mosaic blur sigma (default: 4.0)
  --post-smooth-sigma FLOAT   Optional post-fusion index blur sigma (default: 0.0)
  --stardist-model-dir PATH   StarDist model dir (default: data/stardist_index/0474)
  --stardist-model-name NAME  StarDist model name (default: stardist)
  --stardist-input-key KEY    StarDist manifest input key: index|file_name|logit (default: index)
  --stardist-out-subdir NAME  Output subdir under --out-dir (default: stardist_preds)
  --stardist-preds-name NAME  Output GPKG filename (default: preds.gpkg)
  --scale-stats PATH          Scale stats for CHM aux (default: data/pld-chips/chips_128_100/scale_2_98.json)
  --stardist-cpu              Force StarDist to CPU (sets CUDA_VISIBLE_DEVICES="")
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
SKIP_EXISTING_SAM="true"
STARDIST_CPU="false"
CROP_SMOOTH_SIGMA="2"
SMOOTH_SIGMA="4.0"
POST_SMOOTH_SIGMA="0.0"

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
    --skip-existing-sam) SKIP_EXISTING_SAM="true"; shift ;;
    --no-skip-existing-sam) SKIP_EXISTING_SAM="false"; shift ;;
    --crop-smooth-sigma) CROP_SMOOTH_SIGMA="$2"; shift 2 ;;
    --smooth-sigma) SMOOTH_SIGMA="$2"; shift 2 ;;
    --post-smooth-sigma) POST_SMOOTH_SIGMA="$2"; shift 2 ;;
    --stardist-cpu) STARDIST_CPU="true"; shift ;;
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
TRAIN_MANIFEST="$MANIFESTS_ROOT/$SIZE_ID/train_manifest.json"
PENDING_MANIFEST="$MANIFESTS_ROOT/$SIZE_ID/train_manifest.pending.json"
PENDING_MANIFEST_ROOT="$OUT_DIR/.pending_manifests"
PENDING_COUNT=0

if [[ "$STARDIST_ONLY" != "true" ]]; then
  echo "==> Build manifests"
  "$PYTHON_BIN" "$ROOT_DIR/examples/build_inference_manifests_from_tiles.py" \
    --tiles-dir "$TILES_DIR" \
    --out-root "$MANIFESTS_ROOT" \
    --size-id "$SIZE_ID" \
    --split train

  SAM_MANIFEST_ROOT="$MANIFESTS_ROOT"
  if [[ "$SKIP_EXISTING_SAM" == "true" ]]; then
    mkdir -p "$TRAIN_PROBS_DIR"
    echo "==> Build pending-only SAM manifest (skip existing outputs)"
    "$PYTHON_BIN" "$ROOT_DIR/examples/build_pending_manifest_from_sam_outputs.py" \
      --manifest "$TRAIN_MANIFEST" \
      --sam-out-dir "$TRAIN_PROBS_DIR" \
      --out "$PENDING_MANIFEST"

    PENDING_COUNT="$("$PYTHON_BIN" - "$PENDING_MANIFEST" <<'PY'
import json, sys
recs = json.load(open(sys.argv[1]))
print(len(recs))
PY
)"

    if [[ "$PENDING_COUNT" == "0" ]]; then
      echo "==> SAM up-to-date for size $SIZE_ID (nothing pending)."
    else
      mkdir -p "$PENDING_MANIFEST_ROOT/$SIZE_ID"
      cp "$PENDING_MANIFEST" "$PENDING_MANIFEST_ROOT/$SIZE_ID/train_manifest.json"
      SAM_MANIFEST_ROOT="$PENDING_MANIFEST_ROOT"
      echo "==> Pending chips: $PENDING_COUNT"
    fi
  fi

  if [[ "$SKIP_EXISTING_SAM" != "true" || "$PENDING_COUNT" != "0" ]]; then
    echo "==> SAM manifest inference"
    "$PYTHON_BIN" "$ROOT_DIR/samstars/segmentation/proposals.py" \
      --manifests-root "$SAM_MANIFEST_ROOT" \
      --ckpt-root "$CKPT_ROOT" \
      --out-root "$SAM_OUT_ROOT" \
      --sizes "$SIZE_ID" \
      --splits train \
      --device "$DEVICE" \
      --fast \
      --crop-smooth-sigma "$CROP_SMOOTH_SIGMA" \
      --smooth-sigma "$SMOOTH_SIGMA" \
      --post-smooth-sigma "$POST_SMOOTH_SIGMA"
  fi
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

if [[ ! -f "$STARDIST_MODEL_DIR/$STARDIST_MODEL_NAME/config.json" ]]; then
  if [[ -f "$ROOT_DIR/models/stardist/config.json" ]]; then
    echo "==> StarDist config missing at '$STARDIST_MODEL_DIR/$STARDIST_MODEL_NAME/config.json'; using '$ROOT_DIR/models/stardist'."
    STARDIST_MODEL_DIR="$ROOT_DIR/models"
    STARDIST_MODEL_NAME="stardist"
  else
    echo "Missing StarDist config: $STARDIST_MODEL_DIR/$STARDIST_MODEL_NAME/config.json" >&2
    exit 1
  fi
fi

if [[ ! -f "$SCALE_STATS" ]]; then
  SCALE_STATS="$OUT_DIR/scale_2_98.json"
  echo "==> Scale stats not found; building from probs manifest: $SCALE_STATS"
  "$PYTHON_BIN" "$ROOT_DIR/examples/build_scale_stats_from_probs_manifest.py" \
    --probs-json "$PROBS_JSON" \
    --out "$SCALE_STATS" \
    --nodata-values "-9999"
fi

echo "==> StarDist manifest inference"
STARDIST_CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/samstars/segmentation/refinement.py"
  --probs-json "$PROBS_JSON"
  --model-dir "$STARDIST_MODEL_DIR"
  --model-name "$STARDIST_MODEL_NAME"
  --out-dir "$STARDIST_OUT_DIR"
  --input-key "$STARDIST_INPUT_KEY"
  --use-aux
  --scale-stats "$SCALE_STATS"
  --preds-name "$STARDIST_PREDS_NAME"
  --write-labels
)
if [[ "$STARDIST_CPU" == "true" ]]; then
  CUDA_VISIBLE_DEVICES="" "${STARDIST_CMD[@]}"
else
  "${STARDIST_CMD[@]}"
fi

echo "Done."
echo "  SAM probs:   $SAM_OUT_ROOT/$SIZE_ID"
echo "  StarDist:    $STARDIST_OUT_DIR/preds.gpkg"
echo "  Labels dir:  $STARDIST_OUT_DIR/labels"
