#!/usr/bin/env bash
# run_predict_tiles.sh
#
# 1. Runs predict.py (SAM → crown_probability.tif)
# 2. Immediately runs predict_stars.py on that crown_probability.tif
#
# You may optionally pass tile-directory names (relative to ./tiles/)
# as arguments.  With no arguments every tile directory is processed.
#
# Example:
#   ./run_predict_tiles.sh                       # all tiles
#   ./run_predict_tiles.sh tile_000_003 tile_000_007

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
TILES_DIR="${ROOT_DIR}/tiles"

[[ -d "${TILES_DIR}" ]] || { echo "Tiles dir not found: ${TILES_DIR}" >&2; exit 1; }

###############################################################################
# Build list of tile directories
###############################################################################
if [[ $# -eq 0 ]]; then
  mapfile -t TILE_DIRS < <(find "${TILES_DIR}" -mindepth 1 -maxdepth 1 -type d | sort)
else
  TILE_DIRS=()
  for name in "$@"; do
    dir="${TILES_DIR}/${name}"
    if [[ -d "${dir}" ]]; then
      TILE_DIRS+=("${dir}")
    else
      echo "⚠️  Skipping “${name}” – directory not found under ${TILES_DIR}" >&2
    fi
  done
fi

[[ ${#TILE_DIRS[@]} -gt 0 ]] || { echo "No valid tile directories to process." >&2; exit 1; }

###############################################################################
# Loop over tiles
###############################################################################
for tile_dir in "${TILE_DIRS[@]}"; do
  echo -e "\n=== Processing $(basename "${tile_dir}") ==="

  img="${tile_dir}/img.tif"
  seeds="${tile_dir}/seeds.gpkg"
  chm="${tile_dir}/chm.tif"
  cost="${tile_dir}/cost.tif"
  sam_dec="${ROOT_DIR}/sam_crowns_decoder.pth"
  sam_meta="${ROOT_DIR}/sam_vit_l_0b3195.pth"
  canopy_out="${tile_dir}/crown_probability.tif"
  crowns_out="${tile_dir}/crowns.tif"
  preview_png="${tile_dir}/preview.png"

  # ── Skip tile if final SAM output already exists ───────────────────────────
  if [[ -f "${canopy_out}" ]]; then
    echo "  ✔ ${canopy_out} already exists — skipping this tile."
    continue
  fi

  # Check required inputs for predict.py
  missing=false
  for f in "${img}" "${seeds}" "${chm}"; do
    [[ -f "${f}" ]] || { echo "  Missing ${f}" >&2; missing=true; }
  done
  $missing && { echo "  → Skipping $(basename "${tile_dir}")"; continue; }

  # ── Step 1: SAM → crown_probability.tif ────────────────────────────
  python "${ROOT_DIR}/predict_sam.py" \
    --img       "${img}"   \
    --seeds     "${seeds}" \
    --chm       "${chm}"   \
    --cost      "${cost}"   \
    --sam-dec   "${sam_dec}"   \
    --sam-meta  "${sam_meta}"  \
    --out       "${canopy_out}"

  # ── Step 2: StarDist crowns from crown_probability ──────────────────────────
  if [[ -f "${canopy_out}" ]]; then
    if ! python "${ROOT_DIR}/predict_stars.py" \
          --input    "${canopy_out}" \
          --out-tif  "${crowns_out}" \
          --out-png  "${preview_png}"; then
      echo "  ⚠️  predict_stars failed for $(basename "${tile_dir}") – skipping this tile."
      continue
    fi
  else
    echo "  crown_probability.tif missing – skipping StarDist step" >&2
  fi
done

echo -e "\n✓ All requested tiles processed."