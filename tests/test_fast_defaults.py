"""Fast defaults and explicit slow choices at actual API and CLI boundaries."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from samstars import segment
from samstars.segmentation import engine, proposals
from test_foreground_prompts import CapturingPredictor, prepared_tile, write_bundle


def capture_sam(monkeypatch):
    predictor = CapturingPredictor()
    sam = SimpleNamespace()
    sam.to = lambda device: sam
    sam.eval = lambda: sam
    monkeypatch.setitem(proposals.sam_model_registry, 'vit_l', lambda checkpoint: sam)
    monkeypatch.setattr(proposals, 'SamPredictor', lambda model: predictor)
    return predictor


@pytest.mark.parametrize(('saved', 'override', 'expected'), [
    (None, None, True), (False, None, False), (True, None, True),
    (None, False, False), (False, True, True), (True, False, False),
])
def test_segment_defaults_to_batches_and_respects_saved_or_explicit_choices(
    prepared_tile, tmp_path, monkeypatch, saved, override, expected,
):
    model = write_bundle(tmp_path / 'model', {} if saved is None else {'fast': saved})
    predictor = capture_sam(monkeypatch)
    options = {} if override is None else {'fast': override}
    segment(model=model, tiles_dir=prepared_tile.parent, out=tmp_path / 'run',
            storage='persistent', run_stardist=False, **options)
    # Two seeds are decoded together by default, individually with slow mode.
    assert [call['point_labels'].shape[0] for call in predictor.calls] == ([2] if expected else [1, 1])


@pytest.mark.parametrize('entry', ['chip', 'proposals', 'engine'])
def test_direct_proposal_interfaces_default_to_batches(prepared_tile, tmp_path, monkeypatch, entry):
    predictor = capture_sam(monkeypatch)
    record = dict(img_path=str(prepared_tile / 'img.tif'), tile=prepared_tile.name,
                  chip_id=0, window=[0, 0, 32, 32])
    checkpoint = tmp_path / 'decoder.pth'
    checkpoint.touch()
    if entry == 'chip':
        proposals.process_chip(record, predictor, 'cpu')
    elif entry == 'proposals':
        records = tmp_path / 'tiles.json'
        records.write_text(json.dumps([record]))
        proposals.run_tile_proposals(records, tmp_path / 'out', checkpoint)
    else:
        engine.run_segmentation(tiles_dir=prepared_tile.parent, out_dir=tmp_path / 'out',
                                sam_ckpt=checkpoint, run_stardist=False)
    assert [call['point_labels'].shape[0] for call in predictor.calls] == [2]


@pytest.mark.parametrize(('flag', 'expected'), [(None, True), ('--fast', True), ('--slow', False)])
def test_proposal_cli_fast_default_and_slow_switch(monkeypatch, flag, expected):
    calls = []
    monkeypatch.setattr(proposals, 'run_tile_proposals', lambda *args, **kwargs: calls.append(kwargs))
    argv = ['proposals', '--tile-records', 'tiles.json', '--out-root', 'out', '--ckpt-path', 'decoder.pth']
    monkeypatch.setattr(sys, 'argv', argv + ([] if flag is None else [flag]))
    proposals.main()
    assert calls[0]['fast'] is expected


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('flag', [None, '--fast', '--slow'])
def test_bundle_cli_fast_default_and_slow_switch(monkeypatch, training, flag):
    path = Path(__file__).resolve().parents[1] / 'examples/run_prepared_tiles_inference.py'
    spec = importlib.util.spec_from_file_location('samstars_example_cli', path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(root=Path('out'), crowns=Path('out/crowns.gpkg'))

    monkeypatch.setattr(cli, 'train' if training else 'segment', run)
    monkeypatch.setattr(cli, 'SamStars', SimpleNamespace(load=lambda path: path))
    mode = ['--sam-base-ckpt', 'base.pth'] if training else ['--model', 'model']
    argv = ['bundles', '--tiles-dir', 'tiles', '--out', 'out', *mode]
    monkeypatch.setattr(sys, 'argv', argv + ([] if flag is None else [flag]))
    cli.main()
    expected = (True if training else None) if flag is None else flag == '--fast'
    assert calls[0]['fast'] is expected
