#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "helpers" / "deprecated" / "predict_stars.py"

print(
    "WARNING: helpers/predict_stars.py has moved to helpers/deprecated/. "
    "Prefer pipelines/stardist/predict_stardist_from_probs.py.",
    file=sys.stderr,
)

runpy.run_path(str(TARGET), run_name="__main__")
