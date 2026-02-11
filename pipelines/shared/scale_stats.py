#!/usr/bin/env python3
"""
Shared scaling helpers for SAM/StarDist workflows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_scale_stats(path: Path) -> dict:
    return json.load(open(path))


def scale_to_u8_per_channel(img_hwc: np.ndarray, lo_hi: list[tuple[float, float]]) -> np.ndarray:
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
    data = np.asarray(arr, np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    data = np.clip(data, lo, hi)
    data = (data - lo) / (hi - lo)
    return np.clip(data, 0.0, 1.0).astype(np.float32)
