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

from samstars._prompts import sample_seed_prompts
from samstars.segmentation import proposals, engine
from samstars.training.chips import build_training_chips
from samstars.training.sam_training import Chips, train_decoder
from test_window_preprocessing import make_tile, Predictor


def test_short_tree_seed_cannot_also_be_a_negative_prompt():
    points, labels, _ = sample_seed_prompts(np.ones((5,5)), (2,2), .5, fg_extra=0)
    assert labels[0] == 1
    assert not np.any(np.all(points[labels == 0] == (2,2), axis=1))



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
