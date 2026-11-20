"""Height detail, saved scaling, and train/predict preprocessing agreement."""

import json
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from samstars import train
from samstars.data.scaling import (
    build_scale_stats_from_proposal_records,
    prepare_stardist_input,
    scale_chm,
    scale_to_01,
)


def _settings(scale=10.0, clip=False):
    return {"chm_scaling": {"version": 1, "scale_m": scale, "clip": clip}}


def _raster(path, values, *, nodata=9999):
    values = np.asarray(values, dtype=np.float32)
    with rasterio.open(path, "w", driver="GTiff", count=1, dtype="float32",
                       width=values.shape[1], height=values.shape[0],
                       transform=from_origin(500000, 2200000, 1, 1),
                       crs="EPSG:32605", nodata=nodata) as dst:
        dst.write(values, 1)
    return path


def test_unclipped_heights_stay_distinct_and_clipping_is_optional():
    heights = np.array([-1, 0, 5, 10, 20, 35, np.nan, np.inf], np.float32)
    np.testing.assert_array_equal(
        scale_chm(heights, _settings()), [0, 0, .5, 1, 2, 3.5, 0, 0])
    np.testing.assert_array_equal(
        scale_chm(heights, _settings(clip=True)), [0, 0, .5, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(
        scale_chm(heights, _settings(scale=1)), [0, 0, 5, 10, 20, 35, 0, 0])


@pytest.mark.parametrize("height", [0.0, 10.0, 45.0])
def test_auto_scale_uses_each_training_forest_and_handles_flat_chm(tmp_path, height):
    chm = _raster(tmp_path / "chm.tif", [[height, height], [height, 9999]])
    records = tmp_path / "records.json"
    records.write_text(json.dumps([{"chm": chm.name}]))
    stats = build_scale_stats_from_proposal_records(
        proposal_records=records, out=tmp_path / "stats.json", chm_clip=False)
    assert stats["chm_scaling"]["scale_m"] == (height or 1.0)
    assert stats["chm_scaling"]["source"] == "training_percentile"
    saved = json.loads((tmp_path / "stats.json").read_text())
    # The saved scale applies to new heights without recomputing or capping it.
    np.testing.assert_array_equal(scale_chm(np.array([height, height * 2]), saved),
                                  [1, 2] if height else [0, 0])


def test_auto_percentile_and_low_level_legacy_behavior(tmp_path):
    chm = _raster(tmp_path / "chm.tif", np.arange(100).reshape(10, 10))
    records = tmp_path / "records.json"
    records.write_text(json.dumps([{"chm": str(chm)}]))
    stats = build_scale_stats_from_proposal_records(
        proposal_records=records, out=tmp_path / "new.json", chm_clip=False)
    assert stats["chm_scaling"]["scale_m"] == pytest.approx(97.02)
    assert scale_chm(np.array([99]), stats)[0] > 1
    legacy = build_scale_stats_from_proposal_records(
        proposal_records=records, out=tmp_path / "old.json")
    assert "chm_scaling" not in legacy
    assert legacy["chm_01"]["lo"] == pytest.approx(1.98)
    assert scale_chm(np.array([99]), legacy)[0] == pytest.approx(1)


@pytest.mark.parametrize("normalize", [False, True])
def test_legacy_bundle_keeps_exact_preprocessing(normalize):
    from csbdeep.utils import normalize as normalize_image

    primary = np.arange(16, dtype=np.float32).reshape(4, 4)
    heights = primary * 3
    stats = {"chm_01": {"lo": 2, "hi": 20}}
    expected = np.stack([primary, scale_to_01(heights, 2, 20)], axis=-1)
    if normalize:
        expected = normalize_image(expected, 1, 99.8, axis=(0, 1)).astype(np.float32)
    actual = prepare_stardist_input(primary, chm=heights, scale_stats=stats,
                                    normalize_inputs=normalize)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("mode", ["unclipped", "clipped", "legacy"])
def test_training_and_refinement_receive_identical_saved_height_values(tmp_path, monkeypatch, normalize, mode):
    from samstars.training.stardist_training import load_pairs
    from samstars.segmentation import refinement

    primary = _raster(tmp_path / "proposal.tif", np.arange(16).reshape(4, 4))
    heights = np.array([[0, 10, 20, 30], [40, 50, 9999, 60],
                        [70, 80, 90, 100], [np.nan, -1, np.inf, -9999]], np.float32)
    chm = _raster(tmp_path / "chm.tif", heights)
    mask = _raster(tmp_path / "mask.tif", np.ones((4, 4)))
    records = tmp_path / "records.json"
    records.write_text(json.dumps([{"index": str(primary), "chm": str(chm), "mask": str(mask)}]))
    stats_path = tmp_path / "stats.json"
    if mode == "legacy":
        stats = {"chm_01": {"lo": 2, "hi": 10}}
        stats_path.write_text(json.dumps(stats))
    else:
        stats = build_scale_stats_from_proposal_records(
            proposal_records=records, out=stats_path, chm_scale_m=10, chm_clip=mode == "clipped")
        assert stats["chm_scaling"]["source"] == "fixed"

    images, _ = load_pairs(records, use_aux=True, scale_stats=stats, do_normalize=normalize)
    captured = []

    class Model:
        def _guess_n_tiles(self, image):
            return (1, 1, 1)

        def predict_instances(self, image, **kwargs):
            captured.append(image.copy())
            return np.zeros(image.shape[:2], np.uint16), {}

    monkeypatch.setattr(refinement, "StarDist2D", lambda *args, **kwargs: Model())
    monkeypatch.setattr(sys, "argv", [
        "refinement", "--proposal-records", str(records), "--model-dir", str(tmp_path),
        "--out-dir", str(tmp_path / "prediction"), "--use-aux", "--scale-stats", str(stats_path),
        "--normalize" if normalize else "--no-normalize",
    ])
    refinement.main()
    np.testing.assert_array_equal(captured[0], images[0])
    if mode != "legacy":
        expected = np.array([[0, 1, 2, 3], [4, 5, 0, 6], [7, 8, 9, 10], [0, 0, 0, 0]], np.float32)
        if mode == "clipped":
            expected = np.minimum(expected, 1)
        np.testing.assert_array_equal(captured[0][..., 1], expected)


@pytest.mark.parametrize("kwargs", [
    {"chm_scale_m": 0}, {"chm_scale_m": -1}, {"chm_scale_m": np.nan},
    {"chm_scale_m": np.inf}, {"chm_scale_m": True}, {"chm_scale_m": "20"},
    {"chm_clip": "false"}, {"chm_clip": None},
])
def test_invalid_height_settings_leave_existing_bundle_untouched(tmp_path, kwargs):
    bundle = tmp_path / "model"
    bundle.mkdir()
    manifest = bundle / "manifest.json"
    manifest.write_text("existing model")
    with pytest.raises(ValueError, match="chm_scale_m|chm_clip"):
        train(img=tmp_path / "image.tif", sam_base_ckpt=tmp_path / "base.pth",
              out=bundle, storage="persistent", **kwargs)
    assert manifest.read_text() == "existing model"


def test_unknown_saved_height_transform_is_rejected():
    stats = _settings()
    stats["chm_scaling"]["version"] = 999
    with pytest.raises(ValueError, match="Unsupported CHM scaling"):
        scale_chm(np.array([10]), stats)
