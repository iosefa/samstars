"""Ground-window regressions using real rasters and controlled neural outputs."""
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
from segment_anything.utils.transforms import ResizeLongestSide
from shapely.geometry import Point, box

from samstars import SamStars, train
from samstars._preprocessing import check_proposal_cache, training_contract, validate_contract
from samstars.segmentation import engine, proposals
from samstars.training.chips import build_training_chips
from samstars.training.sam_training import Chips, decoder_loss


def make_tile(tmp_path, shape=(40, 40), resolution=.5):
    tile = tmp_path / 'tiles' / 'tile_000_000'
    tile.mkdir(parents=True)
    y, x = np.indices(shape)
    transform = from_origin(500000, 2200000, resolution, resolution)
    values = {
        'img': np.stack((10 + x, 20 + y, x + y + 30)).astype('float32'),
        'chm': np.where((x % 12 < 3) | (y % 12 < 3), 1, 10 + x / 100).astype('float32')[None],
        'cost': (x / max(1, shape[1])).astype('float32')[None],
    }
    for name, data in values.items():
        with rasterio.open(tile / f'{name}.tif', 'w', driver='GTiff', count=len(data),
                           height=shape[0], width=shape[1], dtype='float32',
                           crs='EPSG:32605', transform=transform, nodata=-9999) as dst:
            dst.write(data)
    points = [Point(500000 + (c + .5) * resolution, 2200000 - (r + .5) * resolution)
              for r in range(6, min(shape[0], 80), 12) for c in range(6, min(shape[1], 80), 12)]
    gpd.GeoDataFrame(geometry=points, crs='EPSG:32605').to_file(tile / 'seeds.gpkg')
    crown = box(500000 + 4 * resolution, 2200000 - 12 * resolution,
                500000 + 12 * resolution, 2200000 - 4 * resolution)
    gpd.GeoDataFrame(geometry=[crown], crs='EPSG:32605').to_file(tile / 'crowns.gpkg')
    return tile


class Predictor:
    def __init__(self):
        self.images, self.calls = [], []
        self.transform = ResizeLongestSide(1024)

    def set_image(self, image):
        self.images.append(image.copy())

    def predict_torch(self, **kwargs):
        self.calls.append(kwargs)
        n = kwargs['point_labels'].shape[0]
        return None, None, torch.zeros((n, 1, 256, 256))


def bundle(tmp_path, settings):
    root = tmp_path / 'model'
    root.mkdir()
    for name in ('decoder.pth', 'config.json', 'weights.h5'):
        (root / name).touch()
    SamStars._write_bundle_manifest(out=root, defaults={}, preprocessing=settings)
    return SamStars(root)


@pytest.mark.parametrize(('metres', 'pixels'), [(20, 40), (50, 100), (100, 200), (200, 400), (512, 1024)])
def test_physical_window_resolves_at_native_resolution(tmp_path, metres, pixels):
    tile = make_tile(tmp_path, shape=(1024, 1024))
    settings = training_contract([tile / 'img.tif'], sam_window_size_m=metres)
    assert settings['sam_window_size_px'] == pixels
    assert settings['sam_stride_px'] == pixels * 3 // 4
    assert settings['sam_encoder_size'] == 1024
    assert validate_contract(settings) == settings


@pytest.mark.parametrize('kwargs', [
    {'sam_window_size_m': 0}, {'sam_window_size_m': float('nan')},
    {'sam_window_size_m': float('inf')}, {'sam_window_size_m': True},
    {'chip_size': 0}, {'chip_size': 3.5}, {'chip_stride': 0},
    {'chip_size': 32, 'chip_stride': 33},
    {'chip_size': 32, 'sam_window_size_m': 20},
    {'stardist_patch_size': 0}, {'stardist_patch_size': 40},
    {'stardist_patch_size': True},
])
def test_invalid_windows_fail_without_changing_existing_bundle(tmp_path, kwargs):
    tile = make_tile(tmp_path)
    out = tmp_path / 'existing_model'
    out.mkdir()
    (out / 'decoder.pth').write_bytes(b'preserve me')
    options = {'stardist_patch_size': 32, **kwargs}
    with pytest.raises(ValueError):
        train(tiles_dir=tile.parent, sam_base_ckpt=tmp_path / 'missing.pth',
              out=out, storage='persistent', **options)
    assert (out / 'decoder.pth').read_bytes() == b'preserve me'
    assert sorted(p.name for p in out.iterdir()) == ['decoder.pth']


def test_stardist_minimum_is_validated_before_sam_training(tmp_path):
    tile = make_tile(tmp_path)
    out = tmp_path / 'model'
    with pytest.raises(ValueError, match=r'40 x 40.*stardist_patch_size=256'):
        train(tiles_dir=tile.parent, sam_base_ckpt=tmp_path / 'missing.pth',
              out=out, storage='persistent', sam_window_size_m=20)
    assert not out.exists()


def test_padding_and_image_scaling_match_training_and_prediction(tmp_path):
    tile = make_tile(tmp_path, shape=(32, 32))
    # Missing data within the real footprint must also be excluded.
    with rasterio.open(tile / 'chm.tif', 'r+') as dst:
        data = dst.read(1)
        data[10, 10] = -9999
        dst.write(data, 1)
    with rasterio.open(tile / 'img.tif', 'r+') as dst:
        data = dst.read()
        data[:, 11, 11] = -9999
        dst.write(data)
    with pytest.warns(UserWarning, match='padded'):
        settings = training_contract([tile / 'img.tif'], sam_window_size_m=20, stardist_patch_size=32)
    chips = tmp_path / 'chips'
    build_training_chips(tile.parent, chips, chip_size=40, stride=30)
    dataset = Chips(chips, img_size=40, preprocessing=settings, fg_extra=0,
                    neg_max_gnd=1000, neg_max_can=1000)
    image, target, points, labels, _ = dataset[0]
    assert image.shape == (3, 1024, 1024)
    assert (target[:, 32:, :] == -1).all() and (target[:, :, 32:] == -1).all()
    assert target[0, 10, 10] == target[0, 11, 11] == -1
    real_points = points[labels >= 0] * (40 / 1024)
    assert (real_points >= 0).all() and (real_points < 32).all()
    prompt_path = next((chips / 'prompts').glob('*.json'))
    point = json.loads(prompt_path.read_text())['point']
    torch.testing.assert_close(points[0], torch.tensor(point) * (1024 / 40))

    predictor = Predictor()
    result = proposals.process_chip(
        dict(img_path=str(tile / 'img.tif'), tile=tile.name, chip_id=0, window=[0, 0, 40, 40]),
        predictor, 'cpu', fixed_window=True, fg_extra=0,
    )
    expected = predictor.transform.apply_image(predictor.images[0]).transpose(2, 0, 1)
    np.testing.assert_array_equal(image.numpy(), expected)
    assert not predictor.images[0][32:].any() and not predictor.images[0][:, 32:].any()
    assert all(np.isfinite(a).all() for a in result[:5])
    for call in predictor.calls:
        p = call['point_coords'][call['point_labels'] >= 0] * (40 / 1024)
        assert (p >= 0).all() and (p < 32).all()


def test_padding_does_not_contribute_to_decoder_loss_or_gradient():
    pred = torch.tensor([[[[.1, .2], [.3, .4]]]], requires_grad=True)
    truth = torch.tensor([[[[1., 0.], [0., 1.]]]])
    expected = decoder_loss(pred, truth)
    padded_pred = torch.nn.functional.pad(pred, (0, 2, 0, 2), value=20)
    padded_pred.retain_grad()
    padded_truth = torch.nn.functional.pad(truth, (0, 2, 0, 2), value=-1)
    actual = decoder_loss(padded_pred, padded_truth)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert (padded_pred.grad[:, :, 2:, :] == 0).all()
    assert (padded_pred.grad[:, :, :, 2:] == 0).all()
    assert (padded_pred.grad[:, :, :2, :2] != 0).all()


def test_dataset_rejects_wrong_chip_size_before_transforming_prompts(tmp_path):
    tile = make_tile(tmp_path)
    chips = tmp_path / 'chips'
    build_training_chips(tile.parent, chips, chip_size=40)
    with pytest.raises(ValueError, match='img_size requires 512'):
        Chips(chips)[0]


def test_new_bundle_tiles_larger_scenes_and_fast_only_changes_batching(tmp_path, monkeypatch):
    tile = make_tile(tmp_path, shape=(52, 70))
    settings = training_contract([tile / 'img.tif'], sam_window_size_m=20, stardist_patch_size=32)
    model = bundle(tmp_path, settings)
    fake_sam = SimpleNamespace()
    fake_sam.to = lambda device: fake_sam
    fake_sam.eval = lambda: fake_sam
    monkeypatch.setitem(proposals.sam_model_registry, 'vit_l', lambda checkpoint: fake_sam)
    outputs, predictors = [], []
    for fast in (False, True, None):
        predictor = Predictor()
        predictors.append(predictor)
        monkeypatch.setattr(proposals, 'SamPredictor', lambda sam: predictor)
        np.random.seed(17)
        result = model.segment(tiles_dir=tile.parent, out=tmp_path / str(fast), storage='persistent',
                               run_stardist=False, fg_extra=0,
                               **({} if fast is None else {'fast': fast}))
        records = json.loads((result.out / 'proposals/proposal_records.json').read_text())
        assert len(records) == 1
        with rasterio.open(records[0]['file_name']) as src, rasterio.open(tile / 'img.tif') as original:
            assert src.shape == (52, 70) and src.transform == original.transform and src.crs == original.crs
            values = src.read(1)
        assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
        outputs.append(values)
        assert len(predictor.images) == 4
        assert all(image.shape == (40, 40, 3) for image in predictor.images)
        assert json.loads((result.out / 'manifest.json').read_text())['preprocessing'] == settings
    for values in outputs[1:]:
        np.testing.assert_allclose(outputs[0], values, atol=1e-7)
    assert len(predictors[0].calls) > len(predictors[1].calls)
    assert len(predictors[1].calls) == len(predictors[2].calls)
    for slow, fast in zip(predictors[0].images, predictors[1].images):
        np.testing.assert_array_equal(slow, fast)


def test_bundle_checks_resolution_but_allows_smaller_prediction_extent(tmp_path):
    tile = make_tile(tmp_path)
    settings = training_contract([tile / 'img.tif'], sam_window_size_m=20, stardist_patch_size=32)
    model = bundle(tmp_path, settings)
    other = make_tile(tmp_path / 'other', resolution=1)
    with pytest.raises(ValueError, match='requires 0.5 m'):
        model.segment(tiles_dir=other.parent, out=tmp_path / 'run', storage='persistent')
    assert not (tmp_path / 'run').exists()
    small = make_tile(tmp_path / 'small', shape=(20, 20))
    model.segment(tiles_dir=small.parent, out=tmp_path / 'small_run', storage='persistent',
                  run_sam=False, run_stardist=False)


@pytest.mark.parametrize('channels', [1, 2])
def test_refinement_reuses_training_scaling_and_normalization(tmp_path, monkeypatch, channels):
    tile = make_tile(tmp_path)
    settings = training_contract([tile / 'img.tif'], sam_window_size_m=20, stardist_patch_size=32)
    model = bundle(tmp_path, settings)
    model.config_path.write_text(json.dumps({'n_channel_in': channels}))
    stats = model.root / 'scale_stats.json'
    stats.write_text(json.dumps({'chm_01': {'lo': 0, 'hi': 20}}))
    # The actual trained layout is authoritative, even if an older defaults entry disagrees.
    SamStars._write_bundle_manifest(out=model.root, defaults={'stardist_normalize_inputs': False,
                                                             'stardist_use_aux': channels == 1},
                                    preprocessing=settings)
    model = SamStars(model.root)
    fake_sam = SimpleNamespace()
    fake_sam.to = lambda device: fake_sam
    fake_sam.eval = lambda: fake_sam
    monkeypatch.setitem(proposals.sam_model_registry, 'vit_l', lambda checkpoint: fake_sam)
    monkeypatch.setattr(proposals, 'SamPredictor', lambda sam: Predictor())
    calls = []
    monkeypatch.setattr(engine.subprocess, 'run', lambda cmd, **kwargs: calls.append(cmd))

    def unexpected_recalculation(**kwargs):
        pytest.fail('Prediction must reuse the saved CHM limits')

    monkeypatch.setattr('samstars.data.build_scale_stats_from_proposal_records', unexpected_recalculation)
    model.segment(tiles_dir=tile.parent, out=tmp_path / 'run', storage='persistent')
    assert len(calls) == 1
    assert '--no-normalize' in calls[0]
    assert '--use-aux' not in calls[0]
    if channels == 2:
        assert calls[0][calls[0].index('--scale-stats') + 1] == str(stats)
    else:
        assert '--scale-stats' not in calls[0]
    assert not (tmp_path / 'run/proposal_scale_stats.json').exists()


def test_small_seedless_scene_produces_empty_native_grid(tmp_path):
    tile = make_tile(tmp_path, shape=(20, 24))
    settings = {'version': 1, 'resolution_m': .5, 'sam_window_size_px': 40,
                'sam_window_size_m': 20, 'sam_stride_px': 30, 'sam_encoder_size': 1024,
                'padding': 'constant_zero', 'rgb_scaling': 'valid_percentile_2_98',
                'stardist_patch_size_px': 16}
    gpd.GeoDataFrame(geometry=[], crs='EPSG:32605').to_file(tile / 'seeds.gpkg')
    predictor = Predictor()
    result = proposals.process_windowed_tile(
        dict(img_path=str(tile / 'img.tif'), tile=tile.name, chip_id=0, window=[0, 0, 24, 20]),
        predictor, 'cpu', settings,
    )
    assert not predictor.images
    for values in result[:3]:
        assert values.shape == (20, 24)
        assert not values.any() and np.isfinite(values).all()


def test_mixed_training_resolutions_rejected(tmp_path):
    a = make_tile(tmp_path / 'a')
    b = make_tile(tmp_path / 'b', resolution=1)
    with pytest.raises(ValueError, match='requires 0.5 m'):
        training_contract([a / 'img.tif', b / 'img.tif'], stardist_patch_size=32)


def test_resume_does_not_reuse_incompatible_window_outputs(tmp_path):
    root = tmp_path / 'proposals'
    (root / 'rasters').mkdir(parents=True)
    raster = root / 'rasters/old.tif'
    raster.touch()
    settings = {'test': 'changed'}
    with pytest.raises(ValueError, match='different preprocessing'):
        check_proposal_cache(root, settings)
    assert raster.exists()
    check_proposal_cache(root, settings, overwrite=True)
    assert not raster.exists()
    assert json.loads((root / 'preprocessing.json').read_text()) == settings


def test_stardist_patch_reaches_model_and_small_records_fail_early(tmp_path, monkeypatch):
    from samstars.training import stardist_training as training

    image, mask = np.zeros((40, 40), 'float32'), np.zeros((40, 40), 'uint16')
    mask[10:20, 10:20] = 1
    np.save(tmp_path / 'index.npy', image)
    np.save(tmp_path / 'mask.npy', mask)
    records = tmp_path / 'records.json'
    records.write_text(json.dumps([{'index': str(tmp_path / 'index.npy'), 'mask': str(tmp_path / 'mask.npy')}]))
    empty = tmp_path / 'empty.json'
    empty.write_text('[]')
    calls = []

    class Model:
        def __init__(self, config, **kwargs):
            assert config.train_patch_size == (32, 32)
            calls.append(config)

        def train(self, X, Y, **kwargs):
            assert X[0].shape == (40, 40, 1)
            assert kwargs['validation_data'][0] is X

    monkeypatch.setattr(training, 'StarDist2D', Model)
    kwargs = dict(train_records=records, val_records=empty, out_dir=tmp_path / 'out')
    training.train_stardist_model(**kwargs, patch_size=32)
    assert len(calls) == 1
    with pytest.raises(ValueError, match='must fit a 256 x 256 patch'):
        training.train_stardist_model(**kwargs)
    assert len(calls) == 1


def test_stardist_cli_forwards_patch_size(monkeypatch):
    from samstars.training import stardist_training as training

    calls = []
    monkeypatch.setattr(training, 'train_stardist_model', lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, 'argv', ['training', '--train-records', 'train.json', '--val-records',
                                    'val.json', '--out-dir', 'out', '--patch-size', '32'])
    training.main()
    assert calls[0]['patch_size'] == 32
