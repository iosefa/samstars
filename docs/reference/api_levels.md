# API Levels

samstars is documented in two levels.

## High-level API (recommended)

Use these for most applications:

- `samstars.SamStars`
- `samstars.SamStarsResult`
- `samstars.train(...)`
- `samstars.segment(...)`
- `samstars.data`
- `samstars.segmentation`
- `samstars.training`

These provide end-to-end operations with sensible defaults.

In practice:

- `samstars.SamStars` is the runtime wrapper around a model bundle directory
- `samstars.train(...)` writes a single flat model bundle directory and returns `SamStars`
- `samstars.segment(...)` accepts a `SamStars` instance or a bundle path and returns `SamStarsResult`
- `samstars.data` exposes tiling, run-record, and scaling helpers
- `samstars.segmentation` exposes `run_segmentation(...)`
- `samstars.training` exposes training-chip preparation, StarDist record preparation, and the `sam_training` / `stardist_training` entrypoint modules

## Low-level API (advanced)

Use these when you need custom control:

- `samstars.features`
- `samstars.utils`

These expose building blocks (costs, seeds, transforms, utilities).

## How to choose

- Start with high-level APIs.
- Move to low-level APIs only when you need custom logic, parameter sweeps, or integration into a larger system.
