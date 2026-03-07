# Training

This page covers model training entrypoints.

## Available training modules

- `samstars/training/sam_training.py`
- `samstars/training/stardist_training.py`

## Typical workflow

1. Prepare training data/manifests.
2. Train SAM components.
3. Train StarDist refinement.
4. Run evaluation/inference on held-out areas.

## Notes

- Keep training runs in dedicated output directories.
- Pin environment versions for reproducibility.
- Use API docs for full parameter details.

See [API Levels](reference/api_levels.md) for which layer to call directly.
