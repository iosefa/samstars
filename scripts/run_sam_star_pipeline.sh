#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_sam_star_pipeline.sh \
    --img /abs/path/image_8band.tif \
    --chm /abs/path/chm.tif \
    --den /abs/path/first_return_density.tif \
    --run-dir /abs/path/output_run_dir \
    [options]

Required:
  --img PATH                  8-band image GeoTIFF
  --chm PATH                  CHM GeoTIFF
  --den PATH                  First-return density GeoTIFF
  --run-dir PATH              Output directory for this run

Optional:
  --python BIN                Python executable (default: python3)
  --tile M                    Tile size in meters (default: 500)
  --overlap M                 Tile overlap in meters (default: 20)
  --sam-ckpt PATH             Base SAM checkpoint
                              (default: sam/sam_vit_l_0b3195.pth)
  --sam-dec PATH              Fine-tuned SAM decoder weights
                              (default: data/chips_128_100_full/0474/sam/sam_crowns_decoder.pth)
  --stardist-model-dir PATH   StarDist model parent dir (default: data/stardist_index/0474)
  --stardist-model-name NAME  StarDist model subdir (default: stardist)
  --skip-stardist             Skip StarDist step (SAM-only)
  --force-sam-infer           Recompute SAM probabilities even if they exist
  -h, --help                  Show this help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="python3"
IMG=""
CHM=""
DEN=""
RUN_DIR=""
TILE_SIZE="500"
OVERLAP_SIZE="20"
SAM_CKPT="$ROOT_DIR/sam/sam_vit_l_0b3195.pth"
SAM_DEC="$ROOT_DIR/data/chips_128_100_full/0474/sam/sam_crowns_decoder.pth"
STARDIST_MODEL_DIR="$ROOT_DIR/data/stardist_index/0474"
STARDIST_MODEL_NAME="stardist"
SKIP_STARDIST="false"
FORCE_SAM_INFER="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --img) IMG="$2"; shift 2 ;;
    --chm) CHM="$2"; shift 2 ;;
    --den) DEN="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --tile) TILE_SIZE="$2"; shift 2 ;;
    --overlap) OVERLAP_SIZE="$2"; shift 2 ;;
    --sam-ckpt) SAM_CKPT="$2"; shift 2 ;;
    --sam-dec) SAM_DEC="$2"; shift 2 ;;
    --stardist-model-dir) STARDIST_MODEL_DIR="$2"; shift 2 ;;
    --stardist-model-name) STARDIST_MODEL_NAME="$2"; shift 2 ;;
    --skip-stardist) SKIP_STARDIST="true"; shift ;;
    --force-sam-infer) FORCE_SAM_INFER="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$IMG" || -z "$CHM" || -z "$DEN" || -z "$RUN_DIR" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

for f in "$IMG" "$CHM" "$DEN"; do
  if [[ ! -f "$f" ]]; then
    echo "Required file not found: $f" >&2
    exit 1
  fi
done

if [[ ! -f "$SAM_CKPT" ]]; then
  echo "SAM checkpoint not found: $SAM_CKPT" >&2
  exit 1
fi
if [[ ! -f "$SAM_DEC" ]]; then
  echo "SAM decoder weights not found: $SAM_DEC" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
TILES_DIR="$RUN_DIR/tiles"

echo "==> Step 1/3: Tile data + build cost + generate seeds"
"$PYTHON_BIN" "$ROOT_DIR/helpers/make_tiles.py" \
  --img "$IMG" \
  --chm "$CHM" \
  --den "$DEN" \
  --out "$TILES_DIR" \
  --tile "$TILE_SIZE" \
  --overlap "$OVERLAP_SIZE"

echo "==> Step 2/3: Run SAM inference on all tiles"
SAM_ARGS=(
  "$PYTHON_BIN" "$ROOT_DIR/helpers/train_sam_pipeline.py"
  --tiles-dir "$TILES_DIR"
  --sam-ckpt "$SAM_CKPT"
  --sam-dec-out "$SAM_DEC"
)
if [[ "$FORCE_SAM_INFER" == "true" ]]; then
  SAM_ARGS+=(--force-sam-infer)
fi
"${SAM_ARGS[@]}"

if [[ "$SKIP_STARDIST" == "true" ]]; then
  echo "==> Done (SAM-only)."
  exit 0
fi

if [[ ! -f "$STARDIST_MODEL_DIR/$STARDIST_MODEL_NAME/config.json" ]]; then
  echo "StarDist config not found: $STARDIST_MODEL_DIR/$STARDIST_MODEL_NAME/config.json" >&2
  exit 1
fi

echo "==> Step 3/3: Run StarDist on each tile crown_probability.tif"
for t in "$TILES_DIR"/tile_*; do
  [[ -d "$t" ]] || continue
  IN_TIF="$t/crown_probability.tif"
  OUT_TIF="$t/crowns.tif"
  OUT_PNG="$t/preview.png"
  if [[ ! -f "$IN_TIF" ]]; then
    echo "Skipping $(basename "$t"): missing $IN_TIF"
    continue
  fi
  "$PYTHON_BIN" "$ROOT_DIR/helpers/predict_stars.py" \
    --model-dir "$STARDIST_MODEL_DIR" \
    --model-name "$STARDIST_MODEL_NAME" \
    --input "$IN_TIF" \
    --out-tif "$OUT_TIF" \
    --out-png "$OUT_PNG"
done

echo "==> Pipeline complete."
echo "Outputs:"
echo "  Tiles dir: $TILES_DIR"
echo "  SAM index: $TILES_DIR/tile_*/sam_index.tif"
echo "  SAM probs: $TILES_DIR/tile_*/crown_probability.tif"
echo "  SAM logit: $TILES_DIR/tile_*/logit.tif"
echo "  StarDist:  $TILES_DIR/tile_*/crowns.tif"
