"""Tests for tiled-run record bookkeeping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from samstars.data import build_pending_tile_records


def test_build_pending_tile_records_filters_only_complete_outputs(tmp_path: Path) -> None:
    records = [
        {"tile": "tile_001", "chip_id": 0, "value": "complete"},
        {"tile": "tile_002", "chip_id": 3, "value": "partial"},
        {"value": "missing-identifiers"},
    ]
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")

    proposal_dir = tmp_path / "proposals"
    proposal_dir.mkdir()
    for suffix in ("sam_index", "sam_prob", "logit"):
        (proposal_dir / f"tile_001_0_{suffix}.tif").touch()
    (proposal_dir / "tile_002_3_sam_index.tif").touch()

    out_path = tmp_path / "pending" / "records.json"
    counts = build_pending_tile_records(
        records_path=records_path,
        proposal_rasters_dir=proposal_dir,
        out_path=out_path,
    )

    assert counts == (3, 1, 2)
    assert json.loads(out_path.read_text(encoding="utf-8")) == records[1:]


def test_build_pending_tile_records_requires_inputs(tmp_path: Path) -> None:
    records_path = tmp_path / "records.json"
    proposal_dir = tmp_path / "proposals"

    with pytest.raises(FileNotFoundError, match="Missing records"):
        build_pending_tile_records(
            records_path=records_path,
            proposal_rasters_dir=proposal_dir,
            out_path=tmp_path / "pending.json",
        )

    records_path.write_text("[]", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Missing proposal raster dir"):
        build_pending_tile_records(
            records_path=records_path,
            proposal_rasters_dir=proposal_dir,
            out_path=tmp_path / "pending.json",
        )
