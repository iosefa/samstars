"""Regression coverage for sensor-independent raster preparation."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from samstars import segment, train
from samstars.data import build_single_tile, build_tiles
from samstars.features import build_cost_surface
from samstars.features.cost_surface import _chm_gradient, _ndvi, _normalise, _texture_entropy
from samstars.segmentation import engine


def _write_raster(path, values, *, transform=None):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
    with rasterio.open(
        path, "w", driver="GTiff", count=len(values), height=values.shape[1],
        width=values.shape[2], dtype="float32", crs="EPSG:32605",
        transform=transform or from_origin(500000, 2200000, 1, 1), nodata=-9999,
    ) as dst:
        dst.write(values)
    return path


@pytest.fixture
def scene(tmp_path):
    rng = np.random.default_rng(7)
    image = rng.uniform(10, 1000, (8, 64, 64)).astype(np.float32)
    y, x = np.indices((64, 64))
    chm = np.ones((64, 64), dtype=np.float32)
    for cx, cy in ((16, 16), (16, 48), (48, 16), (48, 48)):
        chm += 18 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 45)
    return {
        "image": image, "heights": chm,
        "chm": _write_raster(tmp_path / "chm.tif", chm),
        "den": _write_raster(tmp_path / "den.tif", chm * 2),
        "root": tmp_path,
    }


@pytest.mark.parametrize("count", [3, 4, 8])
def test_rgb_cost_accepts_any_source_band_count(scene, count):
    image = _write_raster(scene["root"] / "image.tif", scene["image"][:count])
    cost, profile = build_cost_surface(img_path=image, chm_path=scene["chm"])
    rgb_only, _ = build_cost_surface(
        img_path=_write_raster(scene["root"] / "rgb.tif", scene["image"][:3]),
        chm_path=scene["chm"],
    )
    np.testing.assert_allclose(cost, rgb_only)
    assert cost.shape == (64, 64)
    assert np.isfinite(cost).all() and 0 <= cost.min() <= cost.max() <= 1
    assert profile["count"] == 1


def test_explicit_bands_reproduce_worldview_cost(scene):
    image = _write_raster(scene["root"] / "wv3.tif", scene["image"])
    cost, _ = build_cost_surface(
        img_path=image, chm_path=scene["chm"], rgb_bands=(5, 3, 2),
        nir_band=7, texture_band=1,
    )
    expected = (
        0.4 * _chm_gradient(scene["heights"])
        + 0.3 * _normalise(1 - _ndvi(scene["image"][4], scene["image"][6]))
        + 0.3 * _texture_entropy(scene["image"][0])
    )
    np.testing.assert_allclose(cost, expected, atol=1e-6)


def test_reordered_rgb_and_nir_produce_same_cost(scene):
    image = scene["image"][:4]
    original, _ = build_cost_surface(
        img_path=_write_raster(scene["root"] / "rgbn.tif", image),
        chm_path=scene["chm"], nir_band=4,
    )
    reordered, _ = build_cost_surface(
        img_path=_write_raster(scene["root"] / "nbgr.tif", image[[3, 2, 1, 0]]),
        chm_path=scene["chm"], rgb_bands=(4, 3, 2), nir_band=1,
    )
    rgb_only, _ = build_cost_surface(
        img_path=scene["root"] / "rgbn.tif", chm_path=scene["chm"],
    )
    np.testing.assert_allclose(original, reordered)
    assert not np.allclose(original, rgb_only)


@pytest.mark.parametrize("kwargs", [
    {"rgb_bands": (1, 2)}, {"rgb_bands": (0, 2, 3)},
    {"rgb_bands": (1, 1, 3)}, {"rgb_bands": (1.5, 2, 3)},
    {"nir_band": 4}, {"nir_band": 1}, {"texture_band": 4},
])
def test_invalid_bands_raise_clear_errors(scene, kwargs):
    image = _write_raster(scene["root"] / "rgb.tif", scene["image"][:3])
    with pytest.raises(ValueError, match="band|indices"):
        build_cost_surface(img_path=image, chm_path=scene["chm"], **kwargs)


def test_missing_cost_terms_cannot_have_all_the_weight(scene):
    image = _write_raster(scene["root"] / "rgb.tif", scene["image"][:3])
    with pytest.raises(ValueError, match="available cost term"):
        build_cost_surface(img_path=image, chm_path=scene["chm"], weights=(0, 1, 0, 0))


def test_cost_preserves_nodata_and_handles_constant_rgb(scene):
    values = np.full((3, 64, 64), 100, dtype=np.float32)
    values[:, 0, 0] = -9999
    image = _write_raster(scene["root"] / "constant.tif", values)
    cost, _ = build_cost_surface(img_path=image, chm_path=scene["chm"])
    assert cost[0, 0] == -9999
    assert np.isfinite(cost).all()
    assert cost[1:, 1:].max() > 0


def test_cost_requires_aligned_inputs_and_accepts_old_path_keyword(scene):
    image = _write_raster(scene["root"] / "rgb.tif", scene["image"][:3])
    cost, _ = build_cost_surface(wv3_path=image, chm_path=scene["chm"])
    assert cost.shape == (64, 64)
    with pytest.raises(ValueError, match="not both"):
        build_cost_surface(img_path=image, wv3_path=image, chm_path=scene["chm"])
    other = _write_raster(
        scene["root"] / "shifted.tif", scene["heights"],
        transform=from_origin(500001, 2200000, 1, 1),
    )
    with pytest.raises(ValueError, match="same CRS, transform, and shape"):
        build_cost_surface(img_path=image, chm_path=other)


def test_proposals_require_selected_prepared_channels(scene):
    from samstars.segmentation.proposals import process_chip

    image = _write_raster(scene["root"] / "multiband.tif", scene["image"])
    with pytest.raises(ValueError, match="three selected image channels"):
        process_chip(
            {"img_path": str(image), "tile": "tile_000_000", "chip_id": 0, "window": [0, 0, 32, 32]},
            predictor=None, device="cpu",
        )


@pytest.mark.parametrize("tiled", [False, True])
def test_preparation_writes_selected_rgb_cost_and_real_seeds(scene, tiled):
    img = _write_raster(scene["root"] / "bgr.tif", scene["image"][:3])
    out = scene["root"] / "prepared"
    kwargs = dict(img=img, chm=scene["chm"], den=scene["den"], rgb_bands=(3, 2, 1), out=out)
    if tiled:
        build_tiles(**kwargs, tile=40, overlap=8)
    else:
        build_single_tile(**kwargs)
    tiles = sorted(out.glob("tile_*"))
    assert len(tiles) == (4 if tiled else 1)
    for tile in tiles:
        with rasterio.open(tile / "img.tif") as selected:
            row, col = (int(v) for v in tile.name.split("_")[1:])
            row, col = row * 32, col * 32
            expected = scene["image"][[2, 1, 0], row:row+selected.height, col:col+selected.width]
            np.testing.assert_array_equal(selected.read(), expected)
        with rasterio.open(tile / "cost.tif") as cost:
            assert cost.count == 1 and np.isfinite(cost.read()).all()
        assert not gpd.read_file(tile / "seeds.gpkg").empty


def test_bundle_saves_and_reuses_preparation_bands(scene, monkeypatch):
    """Run real preparation and record building, replacing only model execution."""
    img = _write_raster(scene["root"] / "bgrn.tif", scene["image"][:4])
    crowns = scene["root"] / "crowns.gpkg"
    gpd.GeoDataFrame(geometry=[box(500010, 2199976, 500024, 2199990)], crs="EPSG:32605").to_file(crowns)

    def fake_decoder(**kwargs):
        kwargs["out_path"].write_bytes(b"test decoder")

    def fake_refinement(**kwargs):
        model = kwargs["out_dir"] / kwargs["model_name"]
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        (model / "weights.h5").write_bytes(b"test weights")

    def fake_proposals(**kwargs):
        tile = kwargs["tiles_dir"] / "tile_000_000"
        out = kwargs["out_dir"] / "proposals"
        out.mkdir(parents=True)
        index = out / "tile_000_000_0_sam_index.tif"
        shutil.copy2(tile / "cost.tif", index)
        with rasterio.open(index, "r+") as dst:
            dst.nodata = 0  # Proposal rasters use zero, unlike the cost raster.
        records = out / "proposal_records.json"
        records.write_text(json.dumps([{"index": str(index), "chm": str(tile / "chm.tif")}]))
        return {"proposal_records": records}

    monkeypatch.setitem(sys.modules, "samstars.training.sam_training", SimpleNamespace(train_decoder=fake_decoder))
    monkeypatch.setitem(sys.modules, "samstars.training.stardist_training", SimpleNamespace(train_stardist_model=fake_refinement))
    original_engine = engine.run_segmentation
    monkeypatch.setattr(engine, "run_segmentation", fake_proposals)
    model = train(
        img=img, chm=scene["chm"], den=scene["den"], crowns=crowns,
        sam_base_ckpt=scene["root"] / "base.pth", out=scene["root"] / "model",
        storage="persistent", rgb_bands=(3, 2, 1), nir_band=4, texture_band=2,
        chip_size=32, chip_stride=32,
    )
    assert model.defaults["rgb_bands"] == [3, 2, 1]
    assert model.defaults["nir_band"] == 4
    assert model.defaults["texture_band"] == 2

    monkeypatch.setattr(engine, "run_segmentation", original_engine)
    result = segment(
        model=model.root, img=img, chm=scene["chm"], den=scene["den"],
        out=scene["root"] / "run", storage="persistent", run_sam=False, run_stardist=False,
    )
    prepared = result.out / "prep" / "scene" / "tile_000_000"
    expected_cost, _ = build_cost_surface(
        img_path=img, chm_path=scene["chm"], rgb_bands=(3, 2, 1), nir_band=4, texture_band=2,
    )
    with rasterio.open(prepared / "cost.tif") as cost:
        np.testing.assert_allclose(cost.read(1), expected_cost, atol=1e-6)
    with rasterio.open(prepared / "img.tif") as rgb:
        np.testing.assert_array_equal(rgb.read(), scene["image"][[2, 1, 0]])
