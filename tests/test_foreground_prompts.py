"""Foreground-count regression tests with real prompts and mocked neural inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from shapely.geometry import Point

from samstars import SamStars, segment, train
from samstars.segmentation import engine, proposals
from samstars.training.sam_training import Chips, train_decoder


class CapturingPredictor:
    """Capture the actual SAM prompts without loading a model or checkpoint."""

    def __init__(self):
        self.calls = []
        self.transform = SimpleNamespace(
            apply_coords_torch=lambda points, shape: points,
            apply_boxes_torch=lambda boxes, shape: boxes,
        )

    def set_image(self, image):
        assert image.ndim == 3 and image.shape[2] == 3

    def predict_torch(self, **kwargs):
        self.calls.append(kwargs)
        count = kwargs['point_labels'].shape[0]
        return None, None, torch.zeros((count, 1, 8, 8))


@pytest.fixture
def prepared_tile(tmp_path):
    tile = tmp_path / 'tiles' / 'tile_000_000'
    tile.mkdir(parents=True)
    y, x = np.indices((32, 32))
    chm = (10 + y / 100 + x / 10000).astype('float32')
    chm[:3] = chm[28:] = 0
    chm[:, :3] = chm[:, 28:] = 0
    transform = from_origin(500000, 2200032, 1, 1)
    rasters = {'img': np.stack((x + 1, y + 2, x + y + 3)), 'chm': chm[None], 'cost': np.zeros((1, 32, 32))}
    for name, values in rasters.items():
        with rasterio.open(tile / f'{name}.tif', 'w', driver='GTiff',
                           width=32, height=32, count=len(values), dtype='float32',
                           crs='EPSG:32605', transform=transform, nodata=-9999) as dst:
            dst.write(values.astype('float32'))
    points = [Point(transform * (c + .5, r + .5)) for r, c in ((8, 8), (22, 22))]
    gpd.GeoDataFrame({'id': [0, 1]}, geometry=points, crs='EPSG:32605').to_file(tile / 'seeds.gpkg')
    return tile


def assert_prompts(predictor, expected_extra):
    seeds = 0
    for call in predictor.calls:
        for points, labels, box in zip(call['point_coords'], call['point_labels'], call['boxes']):
            seeds += 1
            assert int((labels == 1).sum()) == 1 + expected_extra
            assert labels[0] == 1
            torch.testing.assert_close(points[0], (box[:2] + box[2:]) / 2)
            # Existing box and ground-negative behavior is preserved.
            torch.testing.assert_close(box[2:] - box[:2], torch.tensor([8., 8.]))
            assert int((labels == 0).sum()) == 50
            foreground = points[labels == 1]
            assert torch.all(foreground >= box[:2])
            assert torch.all(foreground <= box[2:])
    assert seeds == 2


@pytest.mark.parametrize('fast', [False, True])
@pytest.mark.parametrize(('count', 'expected'), [(None, 80), (0, 0), (1, 1), (5, 5), (100, 81)])
def test_foreground_prompts_respect_count_in_both_modes(prepared_tile, fast, count, expected):
    predictor = CapturingPredictor()
    kwargs = {} if count is None else {'fg_extra': count}
    proposals.process_chip(
        {'img_path': str(prepared_tile / 'img.tif'), 'tile': prepared_tile.name,
         'chip_id': 0, 'window': [0, 0, 32, 32]}, predictor, 'cpu', fast=fast, **kwargs,
    )
    assert_prompts(predictor, expected)


@pytest.mark.parametrize('fast', [False, True])
def test_seed_remains_foreground_when_no_extra_pixels_qualify(prepared_tile, fast):
    with rasterio.open(prepared_tile / 'chm.tif', 'r+') as dst:
        dst.write(np.zeros((1, 32, 32), dtype='float32'))
    predictor = CapturingPredictor()
    proposals.process_chip(
        {'img_path': str(prepared_tile / 'img.tif'), 'tile': prepared_tile.name,
         'chip_id': 0, 'window': [0, 0, 32, 32]}, predictor, 'cpu', fast=fast, fg_extra=80,
    )
    assert_prompts(predictor, 0)


def write_bundle(path, defaults):
    path.mkdir()
    manifest = {'bundle_format': 'samstars_model_bundle', 'bundle_version': 1,
                'decoder': 'decoder.pth', 'config': 'config.json', 'weights': 'weights.h5',
                'defaults': defaults}
    for name in ('decoder.pth', 'config.json', 'weights.h5'):
        (path / name).touch()
    (path / 'manifest.json').write_text(json.dumps(manifest))
    return SamStars.load(path)


@pytest.mark.parametrize('fast', [False, True])
@pytest.mark.parametrize(('stored', 'override', 'expected'), [
    (None, None, 80), (0, None, 0), (5, None, 5), (5, 0, 0), (0, 3, 3),
])
def test_public_api_bundle_defaults_and_overrides_reach_sam(
    prepared_tile, tmp_path, monkeypatch, fast, stored, override, expected,
):
    defaults = {} if stored is None else {'fg_extra': stored}
    model = write_bundle(tmp_path / 'model', defaults)
    original_manifest = model.manifest_path.read_bytes()
    predictor = CapturingPredictor()
    fake_sam = SimpleNamespace()
    fake_sam.to = lambda device: fake_sam
    fake_sam.eval = lambda: fake_sam
    monkeypatch.setitem(proposals.sam_model_registry, 'vit_l', lambda checkpoint: fake_sam)
    monkeypatch.setattr(proposals, 'SamPredictor', lambda sam: predictor)
    kwargs = {} if override is None else {'fg_extra': override}
    result = segment(model=model.root, tiles_dir=prepared_tile.parent, out=tmp_path / 'run',
                     storage='persistent', run_stardist=False, fast=fast, **kwargs)
    assert_prompts(predictor, expected)
    records = json.loads((result.out / 'proposals/proposal_records.json').read_text())
    assert len(records) == 1
    with rasterio.open(records[0]['file_name']) as raster:
        assert raster.shape == (32, 32)
        assert np.isfinite(raster.read()).all()
    assert model.manifest_path.read_bytes() == original_manifest


@pytest.mark.parametrize('count', [0, 3, np.int64(5)])
def test_training_dataset_uses_requested_foreground_count(prepared_tile, tmp_path, count):
    chips = tmp_path / 'chips'
    for folder in ('images', 'masks', 'chm', 'prompts'):
        (chips / folder).mkdir(parents=True)
    for source, folder in (('img', 'images'), ('chm', 'chm')):
        with rasterio.open(prepared_tile / f'{source}.tif') as src:
            values = src.read() if source == 'img' else src.read(1)
        np.save(chips / folder / 'crown.npy', values)
    mask = np.zeros((32, 32), dtype='uint8')
    mask[4:13, 4:13] = 1
    np.save(chips / 'masks/crown.npy', mask)
    (chips / 'prompts/crown.json').write_text(json.dumps({'point': [8, 8], 'box': [4, 4, 12, 12]}))
    dataset = Chips(chips, img_size=32, enc_size=32, fg_extra=count, neg_max_gnd=5, neg_max_can=7)
    _, _, points, labels, box = dataset[0]
    assert int((labels == 1).sum()) == 1 + count
    assert int((labels == 0).sum()) == 12
    torch.testing.assert_close(points[0], torch.tensor([8., 8.]))
    torch.testing.assert_close(box, torch.tensor([4., 4., 12., 12.]))


@pytest.mark.parametrize('count', [-1, 1.5, 1.0, True, np.bool_(False), '0', float('nan'), float('inf')])
def test_invalid_counts_fail_before_processing_or_output_writes(tmp_path, count):
    model = write_bundle(tmp_path / 'model', {})
    out = tmp_path / 'output'
    missing = tmp_path / 'missing'
    calls = [
        lambda: train(sam_base_ckpt=missing, out=out, storage='persistent', fg_extra=count),
        lambda: model.segment(out=out, storage='persistent', fg_extra=count),
        lambda: engine.run_segmentation(tiles_dir=missing, out_dir=out, sam_ckpt=missing, fg_extra=count),
        lambda: proposals.run_tile_proposals(missing, out, missing, fg_extra=count),
        lambda: proposals.process_chip({}, None, 'cpu', fg_extra=count),
        lambda: train_decoder(chip_dir=missing, ckpt=missing, out_path=out / 'decoder.pth', fg_extra=count),
    ]
    for call in calls:
        with pytest.raises(ValueError, match='fg_extra must be a nonnegative integer'):
            call()
        assert not out.exists()


@pytest.mark.parametrize('count', [None, 0, 5])
def test_proposal_cli_passes_foreground_count(monkeypatch, count):
    calls = []
    monkeypatch.setattr(proposals, 'run_tile_proposals', lambda *args, **kwargs: calls.append(kwargs))
    argv = ['proposals', '--tile-records', 'tiles.json', '--out-root', 'output', '--ckpt-path', 'decoder.pth']
    if count is not None:
        argv += ['--fg-extra', str(count)]
    monkeypatch.setattr(sys, 'argv', argv)
    proposals.main()
    assert calls[0]['fg_extra'] == (80 if count is None else count)
