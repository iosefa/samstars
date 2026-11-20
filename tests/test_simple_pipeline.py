"""Scientific contracts for simplified proposals, caching, and training."""
import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import torch
from shapely.geometry import Point, box

from samstars._cache import check_cache, file_identity
from samstars._prompts import sample_seed_prompts
from samstars.segmentation import proposals, engine
from samstars.segmentation.simple import smooth_probabilities
from samstars.training.chips import build_training_chips
from samstars.training.sam_training import Chips, train_decoder
from samstars.training.splits import split_training_tiles
from test_window_preprocessing import make_tile, Predictor


def test_smoothing_keeps_valid_background_and_nodata_separate():
    values = np.array([[0, 0, 0, .8, .8, .8, 0, 0, 0]], dtype='float32')
    valid = np.ones(values.shape, bool)
    result = smooth_probabilities(values, 1, valid)
    assert 0 < result[0, 2] < .4
    assert .5 < result[0, 4] < .8
    valid[0, 0] = False
    assert smooth_probabilities(values, 1, valid)[0, 0] == 0
    np.testing.assert_array_equal(smooth_probabilities(values, 0, valid), values)


def test_short_tree_seed_cannot_also_be_a_negative_prompt():
    points, labels, _ = sample_seed_prompts(np.ones((5,5)), (2,2), .5, fg_extra=0)
    assert labels[0] == 1
    assert not np.any(np.all(points[labels == 0] == (2,2), axis=1))


@pytest.mark.parametrize('values', [[[1,0],[0,1]], [[1,1,1],[1,0,1],[1,1,0]]])
def test_diagonal_pixels_export_valid_polygons_without_changing_coverage(values):
    from samstars.segmentation.refinement import _iter_polys_from_labels
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    labels = np.asarray(values, np.int32)
    transform = from_origin(500000,2200000,.5,.5)
    polygons = list(_iter_polys_from_labels(labels, transform=transform))
    assert polygons and all(p.is_valid for _,p in polygons)
    assert sum(p.area for _,p in polygons) == np.count_nonzero(labels)*.25
    rebuilt = rasterize([(p,i) for i,p in polygons], out_shape=labels.shape, transform=transform)
    np.testing.assert_array_equal(rebuilt, labels)


def test_new_training_and_inference_use_identical_prompts(tmp_path):
    tile = make_tile(tmp_path, shape=(32, 32))
    chips = tmp_path/'chips'
    build_training_chips(tile.parent, chips, chip_size=32, shared_windows=True)
    prompt = json.loads(next((chips/'prompts').glob('*.json')).read_text())
    x, y = prompt['point']
    with rasterio.open(tile/'img.tif') as source:
        point = Point(source.transform*(x+.5, y+.5))
    gpd.GeoDataFrame(geometry=[point], crs='EPSG:32605').to_file(tile/'seeds.gpkg')
    data = Chips(chips, img_size=32, fg_extra=3, pipeline_mode='simple', random_seed=42)
    _, _, train_points, train_labels, train_box = data[0]
    predictor = Predictor()
    proposals.process_chip(dict(img_path=str(tile/'img.tif'), tile=tile.name, chip_id=0,
        window=[0,0,32,32]), predictor, 'cpu', pipeline_mode='simple', fg_extra=3,
        crop_smooth_sigma=0, smooth_sigma=0, random_seed=42)
    call = predictor.calls[0]
    keep = train_labels >= 0
    torch.testing.assert_close(train_points[keep], call['point_coords'][0])
    torch.testing.assert_close(train_labels[keep], call['point_labels'][0])
    torch.testing.assert_close(train_box, call['boxes'][0])


def test_simple_masks_are_not_multiplied_by_a_radial_prior(tmp_path):
    tile = make_tile(tmp_path, shape=(32,32))
    result = proposals.process_chip(dict(img_path=str(tile/'img.tif'), tile=tile.name,
        chip_id=0, window=[0,0,32,32]), Predictor(), 'cpu', pipeline_mode='simple',
        crop_smooth_sigma=0, smooth_sigma=0)
    np.testing.assert_allclose(result[1], .5)


def test_optional_smoothing_runs_once_on_the_completed_mosaic(tmp_path, monkeypatch):
    from samstars._preprocessing import training_contract
    tile = make_tile(tmp_path, shape=(32,48))
    settings = training_contract([tile/'img.tif'], chip_size=32, stardist_patch_size=16)
    calls = []
    def window(rec, *args, **kwargs):
        calls.append(kwargs['smooth_sigma'])
        probability = np.full((32,32), .2 if rec['window'][0] == 0 else .8, np.float32)
        return probability, probability, probability, None, None, None, None
    monkeypatch.setattr(proposals, 'process_chip', window)
    result = proposals.process_windowed_tile(
        dict(img_path=str(tile/'img.tif'), tile=tile.name, chip_id=0, window=[0,0,48,32]),
        None, 'cpu', settings, pipeline_mode='simple', smooth_sigma=2)
    assert calls == [0,0]
    np.testing.assert_allclose(result[1], smooth_probabilities(result[0], 2, np.ones((32,48), bool)))
    np.testing.assert_array_equal(result[1], result[2])


def test_changed_smoothing_cannot_reuse_proposals(tmp_path, monkeypatch):
    tile = make_tile(tmp_path, shape=(32,32))
    checkpoint = tmp_path/'decoder.pth'
    checkpoint.touch()
    sam = SimpleNamespace()
    sam.to = lambda device:sam
    sam.eval = lambda:sam
    predictor = Predictor()
    monkeypatch.setitem(proposals.sam_model_registry, 'vit_l', lambda checkpoint:sam)
    monkeypatch.setattr(proposals, 'SamPredictor', lambda sam:predictor)
    arguments = dict(tiles_dir=tile.parent, out_dir=tmp_path/'out', sam_ckpt=checkpoint,
                     run_stardist=False, pipeline_mode='simple', crop_smooth_sigma=0)
    engine.run_segmentation(**arguments, smooth_sigma=0)
    count = len(predictor.calls)
    with pytest.raises(ValueError, match='Cached outputs'):
        engine.run_segmentation(**arguments, smooth_sigma=1)
    assert len(predictor.calls) == count
    engine.run_segmentation(**arguments, smooth_sigma=1, skip_existing_sam=False)
    assert len(predictor.calls) > count


def test_checkpoint_content_is_part_of_cache_identity(tmp_path):
    weights = tmp_path/'weights'
    weights.write_bytes(b'first')
    root = tmp_path/'cache'
    check_cache(root, file_identity(weights))
    (root/'output').touch()
    weights.write_bytes(b'other')
    with pytest.raises(ValueError, match='Cached outputs'):
        check_cache(root, file_identity(weights))


def test_single_area_is_not_presented_as_held_out_validation(tmp_path):
    tile = make_tile(tmp_path)
    with pytest.warns(UserWarning, match='no held-out validation'):
        train, val, summary = split_training_tiles(tile.parent, tmp_path/'split')
    assert val is None and summary['validation'] == 'none'
    assert len(list(train.iterdir())) == 1
    with pytest.raises(ValueError, match='overlap'):
        split_training_tiles(tile.parent, tmp_path/'invalid', validation_tiles_dir=tile.parent)


def test_shared_windows_cache_frozen_embeddings(tmp_path, monkeypatch):
    tile = make_tile(tmp_path, shape=(32,32))
    crowns = gpd.read_file(tile/'crowns.gpkg')
    crowns.loc[1, 'geometry'] = box(500010, 2199986, 500014, 2199990)
    crowns.to_file(tile/'crowns.gpkg')
    chips = tmp_path/'chips'
    build_training_chips(tile.parent, chips, chip_size=32, shared_windows=True)
    assert len(list((chips/'images').glob('*.npy'))) == 1
    assert len(list((chips/'masks').glob('*.npy'))) == 2

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, image):
            self.calls += 1
            return image.mean((1,2,3), keepdim=True)
    class Prompts(torch.nn.Module):
        def get_dense_pe(self):
            return torch.zeros((1,1,1,1))
        def forward(self, **kwargs):
            return torch.zeros((1,1,1)), torch.zeros((1,1,1,1))
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))
        def forward(self, **kwargs):
            return self.bias.expand(1,1,4,4), None
    class Sam(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder, self.prompt_encoder, self.mask_decoder = Encoder(), Prompts(), Decoder()
        def preprocess(self, image):
            return image
    sam = Sam()
    from samstars.training import sam_training
    monkeypatch.setitem(sam_training.sam_model_registry, 'vit_l', lambda checkpoint:sam)
    train_decoder(chips, tmp_path/'base', tmp_path/'trained.pth', img_size=32,
                  epochs=2, pipeline_mode='simple', fg_extra=3)
    assert sam.image_encoder.calls == 1
    summary = json.loads((tmp_path/'trained.training.json').read_text())
    assert summary['epochs'] == 2 and summary['steps_per_epoch'] == 2


@pytest.mark.parametrize(('records', 'minimum', 'expected'), [(1,100,1), (200,1000,10)])
def test_stardist_budget_uses_optimizer_steps(tmp_path, monkeypatch, records, minimum, expected):
    from samstars.training import stardist_training as training
    image = np.zeros((16,16,1), np.float32)
    mask = np.ones((16,16), np.uint16)
    monkeypatch.setattr(training, 'load_pairs', lambda path, **kwargs:
        ([image]*records, [mask]*records) if path.name == 'train' else ([], []))
    calls = []
    monkeypatch.setattr(training, 'StarDist2D', lambda config, **kwargs:
        SimpleNamespace(train=lambda *args, **options:calls.append(options)))
    training.train_stardist_model(train_records=Path('train'), val_records=Path('val'),
        out_dir=tmp_path, patch_size=16, epochs=1, min_iters=minimum, steps_per_epoch=100)
    assert calls[0]['epochs'] == expected
    assert calls[0]['steps_per_epoch'] == 100


def test_disjoint_groups_are_split_before_decoder_training(tmp_path, monkeypatch):
    import shutil
    from samstars import train
    from samstars.training import sam_training, stardist_training
    from affine import Affine
    first = make_tile(tmp_path/'first', shape=(32,32))
    second = make_tile(tmp_path/'second', shape=(32,32))
    for file in second.glob('*.tif'):
        with rasterio.open(file, 'r+') as dst:
            dst.transform = Affine.translation(1000,0)*dst.transform
    for file in second.glob('*.gpkg'):
        data = gpd.read_file(file)
        data.geometry = data.translate(1000,0)
        data.to_file(file)
    inputs = tmp_path/'all_tiles'
    inputs.mkdir()
    (inputs/'tile_a').symlink_to(first, target_is_directory=True)
    (inputs/'tile_b').symlink_to(second, target_is_directory=True)
    checkpoint = tmp_path/'base.pth'
    checkpoint.touch()
    captured = {}
    def decoder(**kwargs):
        captured['images'] = sorted(p.name for p in (kwargs['chip_dir']/'images').glob('*.npy'))
        kwargs['out_path'].write_bytes(b'trained')
    def refinement(**kwargs):
        captured['train'] = json.loads(kwargs['train_records'].read_text())
        captured['val'] = json.loads(kwargs['val_records'].read_text())
        stats = json.loads(kwargs['scale_stats_path'].read_text())
        assert stats['records_total'] == len(captured['train']) == 1
        folder = kwargs['out_dir']/kwargs['model_name']
        folder.mkdir(parents=True)
        (folder/'config.json').write_text('{}')
        (folder/'weights_best.h5').write_bytes(b'weights')
    sam = SimpleNamespace()
    sam.to = lambda device:sam
    sam.eval = lambda:sam
    monkeypatch.setitem(proposals.sam_model_registry, 'vit_l', lambda checkpoint:sam)
    monkeypatch.setattr(proposals, 'SamPredictor', lambda sam:Predictor())
    monkeypatch.setattr(sam_training, 'train_decoder', decoder)
    monkeypatch.setattr(stardist_training, 'train_stardist_model', refinement)
    model = train(tiles_dir=inputs, sam_base_ckpt=checkpoint, out=tmp_path/'model',
        storage='persistent', chip_size=32, stardist_patch_size=16, val_fraction=.5,
        pipeline_mode='simple', crop_smooth_sigma=0, smooth_sigma=0)
    assert len(captured['val']) == 1
    train_name = Path(captured['train'][0]['mask']).name.split('_0_mask')[0]
    val_name = Path(captured['val'][0]['mask']).name.split('_0_mask')[0]
    assert train_name != val_name
    assert all(name.startswith(train_name) for name in captured['images'])
    assert json.loads((model.root/'training_split.json').read_text())['validation'] == 'held_out'
