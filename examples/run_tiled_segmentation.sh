#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash examples/run_tiled_segmentation.sh \
    --tiles-dir /abs/path/run/tiles \
    --out-dir /abs/path/run_dir \
    --sam-ckpt /abs/path/models/sam_crowns_decoder.pth \
    --stardist-model /abs/path/models/stardist \
    [options]

Required:
  --tiles-dir PATH            Tile directory (must contain tile_*/img.tif,chm.tif,cost.tif,seeds.gpkg)
  --out-dir PATH              Output root for run records, proposals, and crowns
  --sam-ckpt PATH             Direct path to SAM decoder checkpoint
  --stardist-model PATH       Direct path to StarDist model directory containing config.json

Optional:
  --python BIN                Python executable (default: python3)
  --device DEV                SAM device (default: mps)
  --skip-existing-sam         Run SAM only on chips without existing proposal rasters (default)
  --no-skip-existing-sam      Recompute SAM for every tile record
  --crop-smooth-sigma FLOAT   Per-crop SAM logit blur sigma (default: 2)
  --smooth-sigma FLOAT        Global SAM probability mosaic blur sigma (default: 4.0)
  --post-smooth-sigma FLOAT   Optional post-fusion index blur sigma (default: 0.0)
  --stardist-input-key KEY    StarDist input key: index|file_name|logit (default: index)
  --crowns-name NAME          Output GeoPackage filename under crowns/ (default: crowns.gpkg)
  --scale-stats PATH          Scale stats for CHM aux (default: build under out-dir if missing)
  --sam-only                  Run SAM only
  --stardist-only             Run StarDist only (assumes proposal records already exist)
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="python3"
TILES_DIR=""
OUT_DIR=""
DEVICE="mps"
SAM_CKPT=""
STARDIST_MODEL=""
STARDIST_INPUT_KEY="index"
CROWNS_NAME="crowns.gpkg"
SCALE_STATS=""
SAM_ONLY="false"
STARDIST_ONLY="false"
SKIP_EXISTING_SAM="true"
CROP_SMOOTH_SIGMA="2"
SMOOTH_SIGMA="4.0"
POST_SMOOTH_SIGMA="0.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tiles-dir) TILES_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --sam-ckpt) SAM_CKPT="$2"; shift 2 ;;
    --stardist-model) STARDIST_MODEL="$2"; shift 2 ;;
    --stardist-input-key) STARDIST_INPUT_KEY="$2"; shift 2 ;;
    --crowns-name) CROWNS_NAME="$2"; shift 2 ;;
    --scale-stats) SCALE_STATS="$2"; shift 2 ;;
    --skip-existing-sam) SKIP_EXISTING_SAM="true"; shift ;;
    --no-skip-existing-sam) SKIP_EXISTING_SAM="false"; shift ;;
    --crop-smooth-sigma) CROP_SMOOTH_SIGMA="$2"; shift 2 ;;
    --smooth-sigma) SMOOTH_SIGMA="$2"; shift 2 ;;
    --post-smooth-sigma) POST_SMOOTH_SIGMA="$2"; shift 2 ;;
    --sam-only) SAM_ONLY="true"; shift ;;
    --stardist-only) STARDIST_ONLY="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$TILES_DIR" || -z "$OUT_DIR" || -z "$SAM_CKPT" || -z "$STARDIST_MODEL" ]]; then
  usage
  exit 1
fi
if [[ "$SAM_ONLY" == "true" && "$STARDIST_ONLY" == "true" ]]; then
  echo "Use only one of --sam-only or --stardist-only." >&2
  exit 1
fi

RECORDS_ROOT="$OUT_DIR/run_records"
PROPOSALS_ROOT="$OUT_DIR/proposals"
CROWNS_DIR="$OUT_DIR/crowns"
PROPOSAL_RECORDS="$PROPOSALS_ROOT/proposal_records.json"
PROPOSAL_RASTERS_DIR="$PROPOSALS_ROOT/rasters"
TILE_RECORDS="$RECORDS_ROOT/tiles.json"
PENDING_RECORDS="$RECORDS_ROOT/tiles.pending.json"
ACTIVE_TILE_RECORDS="$TILE_RECORDS"
PENDING_COUNT=0

if [[ "$STARDIST_ONLY" != "true" ]]; then
  echo "==> Build tile records"
  "$PYTHON_BIN" "$ROOT_DIR/examples/build_tile_records.py" \
    --tiles-dir "$TILES_DIR" \
    --out-root "$RECORDS_ROOT"

  if [[ "$SKIP_EXISTING_SAM" == "true" ]]; then
    mkdir -p "$PROPOSAL_RASTERS_DIR"
    echo "==> Build pending tile records (skip existing proposal rasters)"
    "$PYTHON_BIN" "$ROOT_DIR/examples/build_pending_tile_records.py" \
      --records "$TILE_RECORDS" \
      --proposal-rasters "$PROPOSAL_RASTERS_DIR" \
      --out "$PENDING_RECORDS"

    PENDING_COUNT="$("$PYTHON_BIN" - "$PENDING_RECORDS" <<'PY'
import json, sys
recs = json.load(open(sys.argv[1]))
print(len(recs))
PY
)"

    if [[ "$PENDING_COUNT" == "0" ]]; then
      echo "==> Proposal rasters are up to date."
    else
      ACTIVE_TILE_RECORDS="$PENDING_RECORDS"
      echo "==> Pending tiles: $PENDING_COUNT"
    fi
  fi

  if [[ "$SKIP_EXISTING_SAM" != "true" || "$PENDING_COUNT" != "0" ]]; then
    echo "==> Run SAM proposals"
    "$PYTHON_BIN" "$ROOT_DIR/samstars/segmentation/proposals.py" \
      --tile-records "$ACTIVE_TILE_RECORDS" \
      --ckpt-path "$SAM_CKPT" \
      --out-root "$PROPOSALS_ROOT" \
      --device "$DEVICE" \
      --fast \
      --crop-smooth-sigma "$CROP_SMOOTH_SIGMA" \
      --smooth-sigma "$SMOOTH_SIGMA" \
      --post-smooth-sigma "$POST_SMOOTH_SIGMA"
  fi
fi

if [[ "$SAM_ONLY" == "true" ]]; then
  echo "Done (SAM only): $PROPOSAL_RECORDS"
  exit 0
fi

if [[ ! -f "$PROPOSAL_RECORDS" ]]; then
  if [[ -d "$PROPOSAL_RASTERS_DIR" ]]; then
    echo "==> Proposal records missing; rebuilding from completed proposal rasters"
    "$PYTHON_BIN" - "$PROPOSAL_RASTERS_DIR" "$PROPOSAL_RECORDS" <<'PY'
import json, sys
from pathlib import Path

rasters_dir = Path(sys.argv[1])
out_json = Path(sys.argv[2])

recs = []
for idx in sorted(rasters_dir.glob("*_sam_index.tif")):
    stem = idx.stem
    if not stem.endswith("_sam_index"):
        continue
    base = stem[:-10]
    prob = rasters_dir / f"{base}_sam_prob.tif"
    chm = rasters_dir / f"{base}_chm.tif"
    cost = rasters_dir / f"{base}_cost.tif"
    logit = rasters_dir / f"{base}_logit.tif"
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
    raise SystemExit(f"No complete proposal records found under {rasters_dir}")

out_json.parent.mkdir(parents=True, exist_ok=True)
json.dump(recs, open(out_json, "w"))
print(f"✓ wrote proposal records: {out_json} ({len(recs)} records)")
PY
  else
    echo "Missing proposal records: $PROPOSAL_RECORDS" >&2
    exit 1
  fi
fi

STARDIST_MODEL_DIR="$(dirname "$STARDIST_MODEL")"
STARDIST_MODEL_NAME="$(basename "$STARDIST_MODEL")"
if [[ ! -f "$STARDIST_MODEL_DIR/$STARDIST_MODEL_NAME/config.json" ]]; then
  echo "Missing StarDist config: $STARDIST_MODEL_DIR/$STARDIST_MODEL_NAME/config.json" >&2
  exit 1
fi

if [[ ! -f "$SCALE_STATS" ]]; then
  SCALE_STATS="$OUT_DIR/proposal_scale_stats.json"
  echo "==> Scale stats not found; building from proposal records: $SCALE_STATS"
  "$PYTHON_BIN" "$ROOT_DIR/examples/build_scale_stats_from_proposal_records.py" \
    --proposal-records "$PROPOSAL_RECORDS" \
    --out "$SCALE_STATS" \
    --nodata-values "-9999"
fi

echo "==> Run crown refinement"
STARDIST_CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/samstars/segmentation/refinement.py"
  --proposal-records "$PROPOSAL_RECORDS"
  --model-dir "$STARDIST_MODEL_DIR"
  --model-name "$STARDIST_MODEL_NAME"
  --out-dir "$CROWNS_DIR"
  --input-key "$STARDIST_INPUT_KEY"
  --use-aux
  --scale-stats "$SCALE_STATS"
  --preds-name "$CROWNS_NAME"
  --write-labels
)
"${STARDIST_CMD[@]}"

echo "Done."
echo "  Tile records:      $TILE_RECORDS"
echo "  Proposal records:  $PROPOSAL_RECORDS"
echo "  Crown polygons:    $CROWNS_DIR/$CROWNS_NAME"
echo "  Labels dir:        $CROWNS_DIR/labels"
