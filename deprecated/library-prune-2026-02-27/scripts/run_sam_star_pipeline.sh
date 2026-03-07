#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT_DIR/scripts/deprecated/run_sam_star_pipeline.sh"

echo "WARNING: scripts/run_sam_star_pipeline.sh has moved to scripts/deprecated/."
echo "         Prefer scripts/run_manifest_sam_stardist.sh (manifest-first workflow)."

exec bash "$TARGET" "$@"
