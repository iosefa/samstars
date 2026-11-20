"""The trained model owns refinement channels and height preprocessing."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from samstars import SamStars, segment, train
from samstars.segmentation import engine, refinement


def _model(tmp_path, channels=2):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"n_channel_in": channels}))
    (root / "weights.h5").touch()
    (root / "decoder.pth").touch()
    SamStars._write_bundle_manifest(out=root, defaults={})
    return SamStars(root)


@pytest.mark.parametrize("entry", ["bundle", "engine", "cli"])
@pytest.mark.parametrize("cached_prediction_scale", [False, True])
def test_missing_training_scale_fails_before_processing(tmp_path, monkeypatch, entry, cached_prediction_scale):
    model = _model(tmp_path)
    out = tmp_path / "prediction"
    if cached_prediction_scale:
        out.mkdir()
        (out / "proposal_scale_stats.json").write_text('{"chm_01": {"lo": 0, "hi": 99}}')
    before = {p.name: p.read_bytes() for p in out.glob('*')}

    if entry == "bundle":
        call = lambda: segment(model=model, out=out, storage="persistent")
    elif entry == "engine":
        call = lambda: engine.run_segmentation(
            tiles_dir=tmp_path / "missing_tiles", out_dir=out,
            sam_ckpt=model.decoder_path, stardist_model=model.root)
    else:
        records = tmp_path / "records.json"
        records.write_text("[]")
        monkeypatch.setattr(sys, "argv", [
            "refinement", "--proposal-records", str(records), "--model-dir", str(tmp_path),
            "--model-name", "model", "--out-dir", str(out),
        ])
        call = refinement.main
    with pytest.raises(ValueError, match="requires its training CHM scaling statistics"):
        call()
    assert out.exists() is cached_prediction_scale
    assert {p.name: p.read_bytes() for p in out.glob('*')} == before


def test_old_bundle_can_save_scaling_without_window_metadata(tmp_path):
    model = _model(tmp_path)
    stats = model.root / "scale_stats.json"
    stats.write_text('{"chm_01": {"lo": 0, "hi": 20}}')
    SamStars._write_bundle_manifest(out=model.root, defaults={})
    assert SamStars(model.root).scale_stats_path == stats


@pytest.mark.parametrize("entry", ["bundle", "engine", "cli"])
def test_unsupported_trained_channel_count_fails_before_processing(tmp_path, monkeypatch, entry):
    model = _model(tmp_path, channels=3)
    out = tmp_path / "prediction"
    if entry == "bundle":
        call = lambda: model.segment(out=out, storage="persistent")
    elif entry == "engine":
        call = lambda: engine.run_segmentation(out_dir=out, sam_ckpt=model.decoder_path, stardist_model=model.root)
    else:
        records = tmp_path / "records.json"
        records.write_text("[]")
        monkeypatch.setattr(sys, "argv", [
            "refinement", "--proposal-records", str(records), "--model-dir", str(tmp_path),
            "--model-name", "model", "--out-dir", str(out),
        ])
        call = refinement.main
    with pytest.raises(ValueError, match="one or two input channels"):
        call()
    assert not out.exists()


def test_old_cli_flag_cannot_change_a_trained_model(tmp_path, monkeypatch):
    _model(tmp_path, channels=1)
    records = tmp_path / "records.json"
    records.write_text("[]")
    out = tmp_path / "prediction"
    monkeypatch.setattr(sys, "argv", [
        "refinement", "--proposal-records", str(records), "--model-dir", str(tmp_path),
        "--model-name", "model", "--out-dir", str(out), "--use-aux",
    ])
    with pytest.raises(SystemExit, match="input channels cannot be changed"):
        refinement.main()
    assert not out.exists()


@pytest.mark.parametrize("flag,expected", [("--use-aux", True), ("--no-use-aux", False)])
@pytest.mark.parametrize("training", [False, True])
def test_bundle_cli_chm_choice_applies_only_during_training(monkeypatch, flag, expected, training):
    path = Path(__file__).resolve().parents[1] / "examples/run_prepared_tiles_inference.py"
    spec = importlib.util.spec_from_file_location("samstars_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []
    monkeypatch.setattr(cli, "train", lambda **kwargs: calls.append(kwargs) or SimpleNamespace(root=Path("model")))
    mode = ["--sam-base-ckpt", "base.pth"] if training else ["--model", "model"]
    monkeypatch.setattr(sys, "argv", ["bundles", "--tiles-dir", "tiles", "--out", "out", *mode, flag])
    if training:
        cli.main()
        assert calls[0]["stardist_use_aux"] is expected
    else:
        with pytest.raises(SystemExit, match="options apply only to training"):
            cli.main()
        assert not calls


@pytest.mark.parametrize("method", [train, SamStars.train])
@pytest.mark.parametrize("option", [{"chm_scale_m": 20}, {"chm_clip": True}])
def test_public_training_does_not_expose_manual_scaling(tmp_path, method, option):
    out = tmp_path / "model"
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        method(sam_base_ckpt=tmp_path / "base.pth", out=out, storage="persistent", **option)
    assert not out.exists()


@pytest.mark.parametrize("use_aux", [False, True])
def test_bundle_prediction_does_not_accept_channel_override(tmp_path, use_aux):
    model = _model(tmp_path)
    out = tmp_path / "prediction"
    with pytest.raises(TypeError, match="stardist_use_aux"):
        segment(model=model, out=out, storage="persistent", stardist_use_aux=use_aux)
    with pytest.raises(TypeError, match="stardist_use_aux"):
        model.segment(out=out, storage="persistent", stardist_use_aux=use_aux)
    assert not out.exists()
