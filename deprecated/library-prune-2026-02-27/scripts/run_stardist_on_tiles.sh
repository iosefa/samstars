#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT_DIR/scripts/deprecated/run_stardist_on_tiles.sh"

echo "WARNING: scripts/run_stardist_on_tiles.sh has moved to scripts/deprecated/."
echo "         Prefer scripts/run_manifest_sam_stardist.sh --stardist-only."

exec bash "$TARGET" "$@"
