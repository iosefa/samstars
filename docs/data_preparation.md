# Data Preparation

This page covers data steps before segmentation/training.

## Core steps

1. Tile imagery + auxiliary rasters.
2. Build inference manifests.
3. Build pending manifests for resumable runs.
4. Build CHM scale stats for auxiliary normalization.

## Entry points

- `samstars/data/tiling.py`
- `examples/build_inference_manifests_from_tiles.py`
- `examples/build_pending_manifest_from_sam_outputs.py`
- `examples/build_scale_stats_from_probs_manifest.py`

## Why manifests exist

Manifests are explicit run records.
They make runs resumable, reproducible, and auditable.
