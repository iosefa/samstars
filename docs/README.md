# Docs

- `index.md`: MkDocs landing page.
- `installation.md`: Installation and environment setup.
- `quickstart.md`: Minimal end-to-end run.
- `segmentation.md`: Task-focused segmentation docs.
- `training.md`: Training entrypoints and notes.
- `data_preparation.md`: Tiling/manifests/scale-stat prep.
- `examples.md`: Runnable example entrypoints.

## Install paths

- `pip install samstars`: full library install, including the SAM and StarDist runtime stack.
- `pip install -e .`: editable source install from a local clone.
- `pip install -e ".[docs]"`: editable install plus docs tooling.
- Conda environments: `environment.ubuntu.samstars.yml` and `environment.macos.samstars.yml`.

## Conda environments

- Ubuntu: `environment.ubuntu.samstars.yml`
- macOS: `environment.macos.samstars.yml`

## MkDocs

From repo root:

```bash
mamba env update -n samstars -f environment.ubuntu.samstars.yml
conda activate samstars
mkdocs serve
```

On macOS, use:

```bash
mamba env update -n samstars -f environment.macos.samstars.yml
conda activate samstars
mkdocs serve
```
