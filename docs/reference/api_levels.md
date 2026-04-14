# API Levels

samstars is documented in two levels.

## High-level API (recommended)

Use these for most applications:

- `samstars.data`
- `samstars.segmentation`
- `samstars.training`

These provide end-to-end operations with sensible defaults.

In practice:

- `samstars.data` exposes tiling, manifest, and scaling helpers
- `samstars.segmentation` exposes `run_segmentation(...)`
- `samstars.training` exposes the `sam_training` and `stardist_training` entrypoint modules

## Low-level API (advanced)

Use these when you need custom control:

- `samstars.features`
- `samstars.utils`

These expose building blocks (costs, seeds, transforms, utilities).

## How to choose

- Start with high-level APIs.
- Move to low-level APIs only when you need custom logic, parameter sweeps, or integration into a larger system.
