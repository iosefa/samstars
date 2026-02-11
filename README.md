# samstar

Library + manuscript workflows for SAM -> StarDist crown delineation and
Detectron2 baselines.

## Repo layout

- `pipelines/`: training + inference scripts (SAM, StarDist, Detectron2, objects).
- `analysis/`: metric summaries (AP, PQ/RQ/SQ, class-aware stats, bin tests).
- `plots/`: plotting scripts for figures and comparisons.
- `data_prep/`: chip splits, chip building, and validation-set utilities.
- `docs/`: workflow notes and the reference data card.
- `data/`: generated artifacts (chips, splits, model outputs, figures).
- `manuscript/`: manuscript figures and supporting assets.
- `deprecated/`: archived scripts kept for reference.
- `sam/`, `helpers/`, `utils/`, `obia/`: core code used by the pipelines.

## Quickstart

- Start with `CODE_GUIDE.md` for the active scripts list.
- See `docs/detectron_instructions.md` for end-to-end pipeline commands.

## Data

All generated data live under `data/`. Older root-level artifacts were moved to
`data/legacy_root/` to keep the repo tidy.

## Notes

This repo is structured to support a future Python package release. Pipeline
scripts are kept in `pipelines/`, while reusable code stays in `sam/`,
`helpers/`, and `utils/`.
