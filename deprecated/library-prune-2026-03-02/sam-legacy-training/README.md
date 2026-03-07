# Deprecated SAM Legacy Training Scripts

These scripts predate the manifest-based SAMStars pipeline and are retained only for archive/reference.

Moved on 2026-03-02:
- `sam/01_prepare_chips.py`
- `sam/02_train_sam_decoder.py`
- `sam/03_train_stardist_predict.py`
- `sam/train_sam_end_to_end.py`

Use active scripts instead:
- `helpers/make_tiles.py`
- `scripts/build_inference_manifests_from_tiles.py`
- `pipelines/sam/predict_sam_from_manifests.py`
- `pipelines/stardist/predict_stardist_from_probs.py`
- `pipelines/sam/train_sam_decoder_from_chips.py`
- `pipelines/stardist/train_stardist_on_probs.py`
