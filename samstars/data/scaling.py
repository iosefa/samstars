"""Scaling-stat helpers for samstars inputs and aux channels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio


def load_scale_stats(path: Path) -> dict:
    """Load scale stats."""
    return json.load(open(path))


def scale_to_u8_per_channel(img_hwc: np.ndarray, lo_hi: list[tuple[float, float]]) -> np.ndarray:
    """Scale to u8 per channel."""
    arr = np.asarray(img_hwc, np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 3:
        raise ValueError("scale_to_u8_per_channel expects HWC input")
    h, w, c = arr.shape
    if len(lo_hi) < c:
        raise ValueError("scale_to_u8_per_channel lo_hi length < channels")
    out = np.zeros((h, w, c), dtype=np.uint8)
    for i in range(c):
        lo, hi = lo_hi[i]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        ch = np.clip(arr[..., i], lo, hi)
        ch = (ch - lo) * 255.0 / (hi - lo)
        out[..., i] = np.clip(ch, 0, 255).astype(np.uint8)
    return out


def scale_to_01(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Scale to 01."""
    data = np.asarray(arr, np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    data = np.clip(data, lo, hi)
    data = (data - lo) / (hi - lo)
    return np.clip(data, 0.0, 1.0).astype(np.float32)


def build_scale_stats_from_probs_manifest(
    *,
    probs_json: Path,
    out: Path,
    percentiles: tuple[float, float] = (2.0, 98.0),
    chm_key: str = "chm",
    nodata_values: tuple[float, ...] = (-9999.0, -32768.0),
) -> dict:
    """Build StarDist CHM scale stats from a SAM probs manifest."""
    p_lo, p_hi = percentiles
    recs = json.load(open(probs_json))
    if not isinstance(recs, list):
        raise RuntimeError(f"Manifest must be a list: {probs_json}")

    parts: list[np.ndarray] = []
    missing = 0
    for rec in recs:
        chm_path = rec.get(chm_key, "")
        if not chm_path:
            missing += 1
            continue
        p = Path(chm_path)
        if not p.exists():
            missing += 1
            continue
        with rasterio.open(p) as src:
            marr = src.read(1, masked=True).astype(np.float32)
        arr = np.asarray(marr.compressed(), dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size and nodata_values:
            for nv in nodata_values:
                arr = arr[arr != np.float32(nv)]
        if arr.size:
            parts.append(arr)

    if not parts:
        raise RuntimeError(f"No finite CHM values found in {probs_json}")

    vals = np.concatenate(parts)
    lo, hi = np.nanpercentile(vals, [p_lo, p_hi])
    out_data = {
        "percentiles": [p_lo, p_hi],
        "chm_01": {"lo": float(lo), "hi": float(hi)},
        "source_probs_json": str(probs_json.resolve()),
        "records_total": int(len(recs)),
        "records_missing_or_unreadable_chm": int(missing),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_data, open(out, "w"), indent=2)
    return out_data
