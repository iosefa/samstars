# Architecture Plan

This note is for design discussion. It is not a public user guide yet.

## Goals

The library should feel simple at the top level:

- one training entry point
- one inference entry point
- one model object or model path

But it also needs to handle two very different operating modes:

- small-area experimentation, where in-memory or temporary intermediates are fine
- large-raster production runs, where tiling, checkpointing, and resumability matter

The main mistake to avoid is forcing both cases into one storage model.

## Main recommendation

Use both:

1. a Python model class for the user-facing API
2. a bundle directory on disk for persistence

The Python object should be the runtime interface.
The directory on disk should be the durable artifact.

That gives a clean API without pretending PyTorch and Keras should become one physical weight file.

## Model design

### Recommendation

Introduce a lightweight `SamStars` class.

It should:

- point to a model bundle directory on disk
- lazily load the SAM and StarDist components only when needed
- expose metadata such as:
  - band selection
  - whether CHM aux is expected
  - recommended StarDist thresholds
  - recommended StarDist internal windowing

It should not eagerly load all weights on construction.

### Why

This solves the API problem cleanly:

- Python users can pass around a class instance
- scripts and CLI can still just point at a bundle directory
- the underlying artifacts remain framework-native

### Bundle contents

The bundle directory should stay simple:

```text
model/
  manifest.json
  decoder.pth
  config.json
  thresholds.json
  weights.h5
```

Everything should be at one directory depth.

Reasons:

- simpler to inspect manually
- easier to copy, archive, and reason about
- no value in splitting SAM and StarDist into nested folders if the bundle already represents one model package

The `SamStars` class is just the runtime wrapper around this directory.

## Public API shape

### Recommendation

Keep the public surface simple:

- `train(...) -> Model`
- `segment(...) -> SamStarsResult`

Where:

- `train(...)` writes a bundle directory and returns a `SamStars`
- `segment(...)` accepts either a `SamStars` or a bundle path

The current low-level functions should remain available, but they should not be the main story.

## Preparation design

The next important abstraction is preparation, but it does not need to be a separate public class.

### Recommendation

Preparation should belong inside segmentation.

That means the segmentation-side object should own:

- image preparation
- cost generation
- seed generation
- tile preparation when needed
- temporary vs persistent storage choices

This should not become a second public object that users have to reason about.

### Why seeds and costs do not belong in the bundle

Seeds and cost layers are:

- data-derived
- scene-specific
- potentially temporary

They are not learned parameters.
So they should live inside the segmentation workflow, not in the model bundle.

## Storage policy

This is the key architectural split.

### Small scenes

For small scenes, preparation should be allowed to be:

- in memory
- or in a temporary directory

This is appropriate for:

- notebooks
- local experimentation
- one-area training

The user should not be forced to keep all intermediate rasters on disk by default.

### Large rasters

For large rasters, preparation should be explicitly disk-backed.

This is required for:

- memory safety
- resumability
- parallelization
- long-running production workflows

For large rasters, tiles are not just an implementation detail.
They are the processing unit.

### Recommendation

Every segmentation workflow should accept or internally choose a storage policy, conceptually:

- `storage="memory"`
- `storage="temp"`
- `storage="persistent"`

Interpretation:

- `memory`: keep intermediates in RAM when feasible
- `temp`: write intermediates to a temporary working directory
- `persistent`: write intermediates to a user-specified durable directory

This lets one segmentation API support both notebooks and production runs.

## Exact storage layouts

The storage contract should be fixed and predictable.

The same subtree names should be used in both modes.
The only difference should be where those subtrees live.

### `segment(..., storage="persistent")`

The user-provided `out` directory is the full run root.

For a small-area run:

```text
out/
  crowns/
    crowns.gpkg
  labels/
    *.tif
  proposals/
    proposal_records.json
    rasters/
      *_sam_prob.tif
      *_sam_index.tif
      *_logit.tif
      *_cost.tif
      *_chm.tif
  prep/
    scene/
      img.tif
      chm.tif
      den.tif
      cost.tif
      seeds.gpkg
  run_records/
    tiles.json
    tiles.pending.json
  proposal_scale_stats.json
  manifest.json
```

For a large-raster tiled run:

```text
out/
  crowns/
    crowns.gpkg
  labels/
    tile_*/labels.tif
  proposals/
    proposal_records.json
    rasters/
      tile_*_sam_prob.tif
      tile_*_sam_index.tif
      tile_*_logit.tif
      tile_*_cost.tif
      tile_*_chm.tif
  prep/
    tiles/
      tile_000_000/
        img.tif
        chm.tif
        cost.tif
        seeds.gpkg
      tile_000_001/
        ...
  run_records/
    tiles.json
    tiles.pending.json
  proposal_scale_stats.json
  manifest.json
```

Meaning:

- everything needed for inspection and resume is durable
- `prep/` is durable
- `proposals/` is durable
- final outputs are under `crowns/`

### `segment(..., storage="temp")`

The user-provided `out` directory contains only user-facing outputs.

User-visible output:

```text
out/
  crowns/
    crowns.gpkg
  labels/
    *.tif
  manifest.json
```

Internal scratch layout:

```text
<temp_root>/
  prep/
    scene/ or tiles/
  proposals/
    proposal_records.json
    rasters/
  run_records/
  proposal_scale_stats.json
```

Meaning:

- the same intermediate structure still exists
- but it lives under a temporary root managed by the library
- it is not intended to be durable user-owned run state

### `train(..., storage="persistent")`

The user-provided `out` directory is the model bundle root and also keeps durable training intermediates.

```text
out/
  manifest.json
  decoder.pth
  config.json
  thresholds.json
  weights.h5
  _training/
    prep/
      scene/
        img.tif
        chm.tif
        den.tif
        cost.tif
        seeds.gpkg
    training_chips/
    training_run/
      proposals/
      run_records/
    stardist_training_records/
    training_scale_stats.json
```

Meaning:

- the top level is the durable model bundle
- `_training/` keeps the durable training run state

### `train(..., storage="temp")`

The user-provided `out` directory contains only the final trained bundle.

```text
out/
  manifest.json
  decoder.pth
  config.json
  thresholds.json
  weights.h5
```

Internal scratch layout:

```text
<temp_root>/
  prep/
  training_chips/
  training_run/
  stardist_training_records/
  training_scale_stats.json
```

Meaning:

- same intermediate structure as persistent mode
- but kept only in scratch storage

### Contract summary

The contract should be:

- `persistent`: `out/` is the full durable run root
- `temp`: `out/` only contains durable user-facing outputs
- intermediate subtree names stay consistent across both modes
- only the storage root changes

## Training design

### Core principle

Training data should come only from the labeled area.

That is true for both:

- SAM training
- StarDist training

The full inference raster should not automatically become the training raster.

### Recommendation

Treat training preparation as a separate, explicit stage.

Either:

1. the user clips the training data first
2. or the library takes an explicit training extent

The current clipped training package approach is fine.

The main point is:

- training inputs should be restricted to the trusted labeled area
- inference inputs can still be the full clipped or full tiled scene

## SAM design

### Recommendation

Keep SAM outer-tile-agnostic as much as possible.

SAM is relatively insensitive to outer tile size because the real work is already local:

- prompt-centered crops
- local mask decoding

So the key SAM contracts are:

- prompt generation
- crop logic
- band choice
- local crown appearance

Outer tile size still matters somewhat, but much less than it does for StarDist.

## StarDist design

### Recommendation

Do not run StarDist only on the full outer tile by default.

StarDist should operate on:

- smaller internal refinement windows
- with overlap
- followed by deduplication

This should be part of the normal design, not just a workaround.

### Why

Your experiments already showed:

- SAM can look fine on the large scene
- StarDist quality changes dramatically depending on local context size

So StarDist needs its own refinement-unit abstraction, separate from the outer prepared tile size.

### Current direction

The current internal-windowing change for StarDist is directionally correct:

- outer tile remains the processing container
- StarDist predicts on smaller overlapping windows inside it
- final polygons are deduplicated by IoU

This is the right pattern for production too.

## Recommended object model

If the library grows a fuller object-oriented layer, the clean shape is:

### `SamStars`

Represents the trained bundle.

Responsibilities:

- load manifest
- lazily load SAM / StarDist
- expose defaults
- save/load bundle

Preparation is not a separate public class here.
It is internal behavior owned by `SamStars.segment(...)`.

Internally, the implementation may still use prepared-scene or tile-store helpers.
But those should be internal details unless a strong advanced-use case demands exposing them later.

### `SamStarsResult`

Represents outputs.

Responsibilities:

- crown polygons path
- optional labels path
- optional proposal outputs
- metadata about what was written

## Recommended public workflow

### Small area

```python
model = train(
    img=training_img,
    chm=training_chm,
    den=training_den,
    crowns=training_crowns,
    out=model_dir,
)

result = segment(
    img=inference_img,
    chm=inference_chm,
    den=inference_den,
    model=model,
    out=run_dir,
)
```

Internally:

- segmentation preparation may be `memory` or `temp`
- no persistent tiles required by default

### Large raster

```python
result = segment(
    img=img,
    chm=chm,
    den=den,
    model=model,
    out=run_dir,
    storage="persistent",
)
```

Internally:

- segmentation preparation is disk-backed
- resumability is expected

## Practical near-term plan

This is the staged refactor I would recommend.

### Phase 1

Stabilize the top-level API shape:

- `SamStars` class
- `train(...)`
- `segment(...)`

But keep them mostly as wrappers around existing functions.

### Phase 2

Move preparation behind the segmentation abstraction:

- segmentation owns seed creation
- segmentation owns cost generation
- segmentation decides memory/temp/persistent handling

Make storage policy explicit, but keep the preparation internals private.

### Phase 3

Move seed/cost generation fully into the segmentation-side implementation.

At that point:

- model code only deals with learned components
- segmentation code owns derived scene artifacts

### Phase 4

Separate default persistence policies:

- small-area defaults to temporary intermediates
- large-raster defaults to persistent tiles and cached proposals

## What should not happen

These are the designs I would avoid.

### Avoid one physical model file

Do not try to force:

- PyTorch weights
- Keras weights

into one `.pth` or one `.h5`.

That is not a clean abstraction.

### Avoid putting seeds into the model bundle

Seeds are scene data, not learned model state.

### Avoid always writing every intermediate to disk

That is tolerable for large rasters.
It is not a good default for small scenes.

### Avoid making large-raster processing memory-only

That will fail operationally.
Large rasters need disk-backed tiling and resumability.

## Decision summary

The clean design is:

- a Python `SamStars` object backed by a flat bundle directory
- simple public functions `train(...)` and `segment(...)`
- segmentation that owns preparation for seeds, costs, and tiles
- storage policy chosen by scene scale
- StarDist operating on internal refinement windows, not just the full outer tile

If this design is accepted, the next code step should be:

1. replace the current raw bundle-path API with a `SamStars` class
2. flatten the bundle layout and filenames
3. push temporary-vs-persistent storage policy into the segmentation implementation
