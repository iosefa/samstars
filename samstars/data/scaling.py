"""Scaling-stat helpers for samstars inputs and aux channels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio


def _resolve_record_path(value: str, records_path: Path) -> Path:
    """Resolve a possibly relative record path against a records JSON file."""
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = (records_path.parent / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


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


def validate_chm_scaling(scale_m: float | None, clip: bool) -> tuple[float | None, bool]:
    """Validate user-supplied height scaling before training creates outputs."""
    from samstars._preprocessing import positive_float

    if scale_m is not None:
        scale_m = positive_float(scale_m, "chm_scale_m")
    if not isinstance(clip, bool):
        raise ValueError("chm_clip must be True or False.")
    return scale_m, clip


def scale_chm(arr: np.ndarray, stats: dict) -> np.ndarray:
    """Apply the saved height transform, including legacy clipped scaling."""
    if "chm_scaling" not in stats:
        bounds = stats.get("chm_01", {})
        return scale_to_01(arr, float(bounds.get("lo", 0.0)), float(bounds.get("hi", 1.0)))

    settings = stats["chm_scaling"]
    if settings.get("version") != 1:
        raise ValueError("Unsupported CHM scaling version; update samstars to load this model.")
    scale_m, clip = validate_chm_scaling(settings.get("scale_m"), settings.get("clip"))
    if scale_m is None:
        raise ValueError("Saved chm_scale_m must be a positive finite number.")
    data = np.array(arr, dtype=np.float32, copy=True)
    for nodata in settings.get("nodata_values", ()):
        data[data == nodata] = 0.0
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.maximum(data, 0.0) / scale_m
    if clip:
        data = np.minimum(data, 1.0)
    return data.astype(np.float32)


def prepare_stardist_input(
    primary: np.ndarray,
    *,
    chm: np.ndarray | None = None,
    scale_stats: dict | None = None,
    normalize_inputs: bool = True,
) -> np.ndarray:
    """Use identical channel scaling during refinement training and prediction.

    New CHM transforms retain their saved physical scale even when the primary
    channel is normalized. Older models retain their per-channel normalization.
    """
    primary = np.nan_to_num(np.asarray(primary, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if chm is not None:
        if scale_stats is not None:
            chm = scale_chm(chm, scale_stats)
        image = np.stack([primary, chm], axis=-1).astype(np.float32)
    else:
        image = primary[..., None]
    if normalize_inputs:
        from csbdeep.utils import normalize

        if chm is not None and scale_stats is not None and "chm_scaling" in scale_stats:
            image[..., 0] = normalize(image[..., 0], 1, 99.8, axis=(0, 1))
        else:
            image = normalize(image, 1, 99.8, axis=(0, 1)).astype(np.float32)
    return image


def build_scale_stats_from_proposal_records(
    *,
    proposal_records: Path,
    out: Path,
    percentiles: tuple[float, float] = (2.0, 98.0),
    chm_key: str = "chm",
    nodata_values: tuple[float, ...] = (-9999.0, -32768.0),
    chm_scale_m: float | None = None,
    chm_clip: bool | None = None,
) -> dict:
    """Build StarDist CHM scale stats from training proposal records.

    An explicit chm_clip selects zero-anchored height scaling. If chm_scale_m is
    None, the upper training percentile sets the scale, with 1 m as a fallback
    for nonpositive percentiles. Omitting both options preserves legacy scaling
    for low-level callers. The public train API always uses automatic scaling
    without upper clipping; manual scale/clipping choices belong to this helper.
    """
    use_height_scale = chm_clip is not None or chm_scale_m is not None
    if use_height_scale:
        chm_scale_m, chm_clip = validate_chm_scaling(
            chm_scale_m, False if chm_clip is None else chm_clip)
    p_lo, p_hi = percentiles
    recs = json.load(open(proposal_records))
    if not isinstance(recs, list):
        raise RuntimeError(f"Proposal records must be a list: {proposal_records}")

    parts: list[np.ndarray] = []
    missing = 0
    for rec in recs:
        chm_path = rec.get(chm_key, "")
        if not chm_path:
            missing += 1
            continue
        p = _resolve_record_path(chm_path, proposal_records)
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
        raise RuntimeError(f"No finite CHM values found in {proposal_records}")

    vals = np.concatenate(parts)
    lo, hi = np.nanpercentile(vals, [p_lo, p_hi])
    out_data = {
        "percentiles": [p_lo, p_hi],
        "chm_01": {"lo": float(lo), "hi": float(hi)},
        "source_proposal_records": str(proposal_records.resolve()),
        "records_total": int(len(recs)),
        "records_missing_or_unreadable_chm": int(missing),
    }
    if use_height_scale:
        resolved_scale = chm_scale_m if chm_scale_m is not None else (float(hi) if hi > 0 else 1.0)
        out_data["chm_01"] = {"lo": 0.0, "hi": resolved_scale}
        out_data["chm_scaling"] = {
            "version": 1,
            "scale_m": resolved_scale,
            "clip": chm_clip,
            "source": "fixed" if chm_scale_m is not None else "training_percentile",
            "nodata_values": list(nodata_values),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_data, open(out, "w"), indent=2)
    return out_data
