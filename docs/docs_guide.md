# Docs

- `index.md`: MkDocs landing page.
- `installation.md`: Installation and environment setup.
- `quickstart.md`: Minimal end-to-end run.
- `segmentation.md`: Task-focused segmentation docs.
- `training.md`: Training entrypoints and notes.
- `data_preparation.md`: Tiling/run-record/scale-stat prep.
- `examples.md`: Runnable example entrypoints.

## Install paths

- `pip install samstars`: full library install.
- `pip install -e .`: editable source install from a local clone.
- `pip install -e ".[docs]"`: editable install plus docs tooling.
- Conda environments: `environment.ubuntu.samstars.yml` and `environment.macos.samstars.yml`.

## Conda environments

- Ubuntu: `environment.ubuntu.samstars.yml`
- macOS: `environment.macos.samstars.yml`

## MkDocs

From repo root:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Notebook pages in the docs use `mkdocs-jupyter`.
If you add or update `.ipynb` files under `docs/`, make sure the docs extras are installed before building.

Or from the pinned project environment:

```bash
conda activate samstars
mkdocs serve
```

Open `http://127.0.0.1:8000`.
