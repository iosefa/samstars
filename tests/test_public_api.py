"""Tests for the lightweight public API and model-bundle contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import samstars
from samstars import SamStars, SamStarsResult


def _write_bundle(root: Path, *, bundle_format: str = "samstars_model_bundle") -> Path:
    root.mkdir()
    manifest = {
        "bundle_format": bundle_format,
        "bundle_version": 1,
        "decoder": "decoder.pth",
        "config": "config.json",
        "weights": "weights.h5",
        "thresholds": "thresholds.json",
        "defaults": {"device": "cpu"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("decoder.pth", "config.json", "weights.h5", "thresholds.json"):
        (root / name).touch()
    return root


def test_top_level_public_surface() -> None:
    assert dir(samstars) == sorted(samstars.__all__)
    assert "SamStars" in samstars.__all__
    assert "utils" not in samstars.__all__
    for name in samstars.__all__:
        assert getattr(samstars, name) is not None
    with pytest.raises(AttributeError):
        getattr(samstars, "not_a_public_name")


def test_result_serializes_paths() -> None:
    result = SamStarsResult(
        out=Path("run"),
        crowns=Path("run/crowns.gpkg"),
        labels_dir=None,
        storage="persistent",
        work_dir=Path("run/work"),
    )

    assert result.to_dict() == {
        "out": "run",
        "crowns": "run/crowns.gpkg",
        "labels_dir": None,
        "storage": "persistent",
        "work_dir": "run/work",
    }


def test_model_bundle_loads_and_resolves_artifacts(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path / "model")

    model = SamStars.load(bundle_dir)

    assert model.root == bundle_dir
    assert model.decoder_path == bundle_dir / "decoder.pth"
    assert model.config_path == bundle_dir / "config.json"
    assert model.weights_path == bundle_dir / "weights.h5"
    assert model.thresholds_path == bundle_dir / "thresholds.json"
    assert model.defaults == {"device": "cpu"}
    assert SamStars.load(model) is model


def test_model_bundle_rejects_unknown_format(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path / "model", bundle_format="unknown")

    with pytest.raises(ValueError, match="Unsupported model bundle format"):
        SamStars(bundle_dir)


def test_model_bundle_requires_declared_artifacts(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path / "model")
    (bundle_dir / "decoder.pth").unlink()

    with pytest.raises(FileNotFoundError, match="Missing decoder checkpoint"):
        SamStars(bundle_dir)
