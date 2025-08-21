# Manuscript Workspace

This folder contains all manuscript-related code, assets, and results separated from the core algorithm.

## Layout

- figures/
  - scripts/: Python scripts that generate figures and section plots.
  - output/: Generated images/artifacts (gitignored).
- experiments/
  - training/: Training/validation utilities used for experiments.
  - logs/: Long-running logs and training outputs (gitignored).
  - configs/: Experiment configurations.
- notebooks/: Jupyter notebooks used in the paper.
- assets/: Static assets used in the manuscript (images, styles, diagrams).
- results/: Derived results (metrics, SHAP outputs, reports).
- data/: Small sample data for figures (large data should be ignored or stored externally).

## Usage

- Create env (recommended):

```bash
conda env create -f environment.yml
conda activate sam-manuscript
```

- Generate figures: run `make figures` from this folder. The Makefile sets `PYTHONPATH` to the repository root so imports like `import sam`, `import obia`, and `import utils` work.
- Run a single figure: `cd figures/scripts && PYTHONPATH=..\.. python figure2.py`.
- Notebooks: open from `notebooks/` with your Python environment.

## Notes

- The core algorithm lives at the repository root (`sam/`, `obia/`, `utils/`, and pipeline scripts).
- CLI tools `create_objects.py` and `classify.py` remain at the root for reuse across projects.
- Consider adding Git LFS for large assets/results.
