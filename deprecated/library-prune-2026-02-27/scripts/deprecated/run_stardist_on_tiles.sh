#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_stardist_on_tiles.sh \
    --run-dir /abs/path/output_run_dir \
    [options]

Required:
  --run-dir PATH              Run directory containing tiles/ (from SAM stage)

Optional:
  --python BIN                Python executable (default: python3)
  --model-dir PATH            StarDist model parent dir
                              (default: data/stardist_index/0474)
  --model-name NAME           StarDist model subdir (default: stardist)
  --input-name NAME           Input raster per tile (default: crown_probability.tif)
  --aux-name NAME             Aux raster per tile (default: chm.tif)
  --no-aux                    Do not pass aux raster to StarDist
  --out-name NAME             Output label raster per tile (default: crowns.tif)
  --preview-name NAME         Output preview PNG per tile (default: preview.png)
  --write-polygons            Write per-tile polygons from label raster
  --polygon-name NAME         Polygon output name per tile (default: crowns.gpkg)
  --force                     Re-run even if out-name exists
  -h, --help                  Show this help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "WARNING: scripts/run_stardist_on_tiles.sh is legacy."
echo "         Prefer scripts/run_manifest_sam_stardist.sh --stardist-only."

PYTHON_BIN="python3"
RUN_DIR=""
MODEL_DIR="$ROOT_DIR/data/stardist_index/0474"
MODEL_NAME="stardist"
INPUT_NAME="crown_probability.tif"
AUX_NAME="chm.tif"
USE_AUX="true"
OUT_NAME="crowns.tif"
PREVIEW_NAME="preview.png"
WRITE_POLYGONS="false"
POLYGON_NAME="crowns.gpkg"
FORCE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --input-name) INPUT_NAME="$2"; shift 2 ;;
    --aux-name) AUX_NAME="$2"; shift 2 ;;
    --no-aux) USE_AUX="false"; shift ;;
    --out-name) OUT_NAME="$2"; shift 2 ;;
    --preview-name) PREVIEW_NAME="$2"; shift 2 ;;
    --write-polygons) WRITE_POLYGONS="true"; shift ;;
    --polygon-name) POLYGON_NAME="$2"; shift 2 ;;
    --force) FORCE="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  echo "Missing required argument: --run-dir" >&2
  usage
  exit 1
fi

TILES_DIR="$RUN_DIR/tiles"
if [[ ! -d "$TILES_DIR" ]]; then
  echo "Tiles directory not found: $TILES_DIR" >&2
  exit 1
fi

if [[ ! -f "$MODEL_DIR/$MODEL_NAME/config.json" ]]; then
  echo "Missing StarDist config: $MODEL_DIR/$MODEL_NAME/config.json" >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/$MODEL_NAME/weights_best.h5" && ! -f "$MODEL_DIR/$MODEL_NAME/weights_last.h5" ]]; then
  echo "Missing StarDist weights in: $MODEL_DIR/$MODEL_NAME" >&2
  exit 1
fi

count_total=0
count_done=0
count_skip=0

for t in "$TILES_DIR"/tile_*; do
  [[ -d "$t" ]] || continue
  count_total=$((count_total + 1))

  in_tif="$t/$INPUT_NAME"
  aux_tif="$t/$AUX_NAME"
  out_tif="$t/$OUT_NAME"
  out_png="$t/$PREVIEW_NAME"
  out_gpkg="$t/$POLYGON_NAME"

  if [[ ! -f "$in_tif" ]]; then
    echo "Skipping $(basename "$t"): missing $INPUT_NAME"
    count_skip=$((count_skip + 1))
    continue
  fi

  if [[ -f "$out_tif" && "$FORCE" != "true" ]]; then
    echo "Skipping $(basename "$t"): $OUT_NAME exists"
    count_skip=$((count_skip + 1))
    continue
  fi

  CMD=(
    "$PYTHON_BIN" "$ROOT_DIR/helpers/predict_stars.py"
    --model-dir "$MODEL_DIR"
    --model-name "$MODEL_NAME"
    --input "$in_tif"
    --out-tif "$out_tif"
    --out-png "$out_png"
  )
  if [[ "$USE_AUX" == "true" ]]; then
    if [[ ! -f "$aux_tif" ]]; then
      echo "Skipping $(basename "$t"): missing $AUX_NAME (required in aux mode)"
      count_skip=$((count_skip + 1))
      continue
    fi
    CMD+=(--aux "$aux_tif")
  fi

  "${CMD[@]}"

  if [[ "$WRITE_POLYGONS" == "true" ]]; then
    "$PYTHON_BIN" - "$out_tif" "$out_gpkg" <<'PY'
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

labels_tif = Path(sys.argv[1])
out_gpkg = Path(sys.argv[2])

with rasterio.open(labels_tif) as src:
    arr = src.read(1)
    crs = src.crs
    transform = src.transform

records = []
for geom, val in shapes(arr, mask=(arr > 0), transform=transform):
    v = int(val)
    if v <= 0:
        continue
    g = shape(geom)
    if g.is_empty:
        continue
    records.append({"label_id": v, "geometry": g})

if records:
    gdf = gpd.GeoDataFrame(pd.DataFrame(records), geometry="geometry", crs=crs)
    gdf["area_m2"] = gdf.geometry.area
else:
    gdf = gpd.GeoDataFrame(
        pd.DataFrame({"label_id": pd.Series(dtype="int64"), "area_m2": pd.Series(dtype="float64")}),
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )
out_gpkg.parent.mkdir(parents=True, exist_ok=True)
gdf.to_file(out_gpkg, driver="GPKG")
print(f"Wrote polygons: {out_gpkg} ({len(gdf)} features)")
PY
  fi

  count_done=$((count_done + 1))
done

echo "Done."
echo "  tiles seen:   $count_total"
echo "  tiles run:    $count_done"
echo "  tiles skipped:$count_skip"
