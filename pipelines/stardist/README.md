# StarDist Pipelines

- `build_prob_manifests.py`: build train/val prob manifests.
- `train_stardist_on_probs.py`: train StarDist on SAM outputs.
- `eval_stardist_on_probs.py`: evaluate and write `val_preds.gpkg`.
- `predict_stardist_from_probs.py`: inference-only: read a SAM probs manifest and write predicted crown polygons (`preds.gpkg`).
- `train_stardist_panoptic.py` / `eval_stardist_panoptic.py`: panoptic helpers.
