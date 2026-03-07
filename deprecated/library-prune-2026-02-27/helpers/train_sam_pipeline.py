#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "helpers" / "deprecated" / "train_sam_pipeline.py"

print(
    "WARNING: helpers/train_sam_pipeline.py has moved to helpers/deprecated/. "
    "Prefer pipelines/sam/predict_sam_from_manifests.py.",
    file=sys.stderr,
)

runpy.run_path(str(TARGET), run_name="__main__")
