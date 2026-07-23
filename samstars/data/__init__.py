"""Data and metadata utilities for samstars."""

from .records import build_pending_tile_records, build_tile_records
from .scaling import (
    build_scale_stats_from_proposal_records,
    load_scale_stats,
    scale_to_01,
    scale_to_u8_per_channel,
)

__all__ = [
    "build_single_tile",
    "build_tiles",
    "build_tile_records",
    "build_pending_tile_records",
    "build_scale_stats_from_proposal_records",
    "load_scale_stats",
    "scale_to_01",
    "scale_to_u8_per_channel",
]


def __getattr__(name):
    """Lazy-load tiling helpers so ``python -m samstars.data.tiling`` is clean."""
    if name in {"build_single_tile", "build_tiles"}:
        from .tiling import build_single_tile, build_tiles

        return {"build_single_tile": build_single_tile, "build_tiles": build_tiles}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
