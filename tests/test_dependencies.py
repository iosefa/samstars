"""Smoke tests for the machine-learning interfaces used at runtime."""

from __future__ import annotations

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    ("module_name", "symbol"),
    [
        ("csbdeep.utils", "normalize"),
        ("segment_anything", "SamPredictor"),
        ("stardist.models", "StarDist2D"),
        ("tensorflow", "__version__"),
        ("torch", "__version__"),
        ("torchvision", "__version__"),
    ],
)
def test_runtime_dependency_imports(module_name: str, symbol: str) -> None:
    module = import_module(module_name)

    assert getattr(module, symbol)


@pytest.mark.parametrize(
    ("module_name", "symbol"),
    [
        ("samstars.segmentation.proposals", "run_tile_proposals"),
        ("samstars.segmentation.refinement", "main"),
        ("samstars.training.sam_training", "train_decoder"),
        ("samstars.training.stardist_training", "train_stardist_model"),
    ],
)
def test_ml_runtime_modules_import(module_name: str, symbol: str) -> None:
    module = import_module(module_name)

    assert getattr(module, symbol)
