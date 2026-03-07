#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$ROOT_DIR/deprecated/run_predict_tiles.sh"

echo "WARNING: helpers/run_predict_tiles.sh has moved to helpers/deprecated/."
echo "         Prefer manifest workflow: scripts/run_manifest_sam_stardist.sh."

exec bash "$TARGET" "$@"
