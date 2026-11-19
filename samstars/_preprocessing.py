"""Window and sampling contracts shared by training and prediction.

Heavy scientific imports stay inside functions so loading a bundle stays cheap.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from pathlib import Path
import warnings

SAM_ENCODER_SIZE = 1024


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def positive_float(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return float(value)


def patch_size(value):
    value = positive_int(value, "stardist_patch_size")
    if value % 16:
        raise ValueError("stardist_patch_size must be a multiple of 16 for the current StarDist architecture.")
    return value


def window_starts(length, size, stride):
    """Cover an axis without unnecessary sliver windows at the far edge."""
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def image_paths(tiles_dir=None, img=None):
    if tiles_dir is not None and img is not None:
        raise ValueError("Provide either tiles_dir or source imagery, not both.")
    if img is not None:
        return [Path(img)]
    if tiles_dir is None:
        raise ValueError("Provide tiles_dir or source imagery.")
    root = Path(tiles_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing tiles directory: {root}")
    paths = [p / "img.tif" for p in sorted(root.iterdir()) if p.is_dir()]
    if not paths:
        raise ValueError(f"No training or inference tiles in {root}.")
    return paths


def inspect_images(paths, *, expected_resolution=None, minimum_size=None):
    """Read only headers; reject incompatible sampling before model execution."""
    import rasterio

    resolution = expected_resolution
    dimensions = []
    for path in paths:
        with rasterio.open(path) as src:
            t = src.transform
            if (not src.crs or not src.crs.is_projected
                    or not math.isclose(src.crs.linear_units_factor[1], 1.0)
                    or t.a <= 0 or t.e >= 0 or t.b != 0 or t.d != 0
                    or not math.isclose(t.a, -t.e, rel_tol=1e-6)):
                raise ValueError(f"{path}: imagery must have a projected metre CRS and north-up square pixels.")
            if resolution is None:
                resolution = float(t.a)
            if not math.isclose(t.a, resolution, rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError(
                    f"{path}: pixel size is {t.a:g} m; this model requires {resolution:g} m. "
                    "Align/resample the inputs to the model resolution before processing."
                )
            if minimum_size and min(src.width, src.height) < minimum_size:
                raise ValueError(
                    f"{path}: training image is {src.width} x {src.height} pixels; "
                    f"stardist_patch_size={minimum_size} requires at least {minimum_size} pixels "
                    f"in each dimension ({minimum_size * resolution:g} m). Use a smaller valid "
                    "StarDist patch or larger annotated imagery."
                )
            dimensions.append((src.height, src.width))
    return resolution, dimensions


def training_contract(paths, *, chip_size=None, chip_stride=None,
                      sam_window_size_m=None, stardist_patch_size=256):
    patch = patch_size(stardist_patch_size)
    if chip_size is not None:
        chip_size = positive_int(chip_size, "chip_size")
    if chip_stride is not None:
        chip_stride = positive_int(chip_stride, "chip_stride")
    if sam_window_size_m is not None:
        sam_window_size_m = positive_float(sam_window_size_m, "sam_window_size_m")
        if chip_size is not None:
            raise ValueError("Choose sam_window_size_m or chip_size, not both.")
    resolution, shapes = inspect_images(paths, minimum_size=patch)
    size = (max(1, int(math.floor(sam_window_size_m / resolution + .5)))
            if sam_window_size_m is not None else (chip_size or 512))
    stride = chip_stride if chip_stride is not None else max(1, size * 3 // 4)
    if stride > size:
        raise ValueError("chip_stride must not exceed the SAM window size; that would leave gaps.")
    if any(min(shape) < size for shape in shapes):
        warnings.warn(
            f"Some training images are smaller than the {size}-pixel SAM window and will be padded. "
            "Padding is excluded from RGB scaling, prompt sampling, and decoder loss. "
            "Consider a smaller window to retain more real image context.", UserWarning, stacklevel=2,
        )
    return {
        "version": 1, "resolution_m": resolution,
        "sam_window_size_px": size, "sam_window_size_m": size * resolution,
        "sam_stride_px": stride, "sam_encoder_size": SAM_ENCODER_SIZE,
        "padding": "constant_zero", "rgb_scaling": "valid_percentile_2_98",
        "stardist_patch_size_px": patch,
    }


def validate_contract(contract):
    if not isinstance(contract, dict) or contract.get("version") != 1:
        raise ValueError("Unsupported model preprocessing contract.")
    try:
        res = positive_float(contract["resolution_m"], "resolution_m")
        size = positive_int(contract["sam_window_size_px"], "sam_window_size_px")
        stride = positive_int(contract["sam_stride_px"], "sam_stride_px")
        patch_size(contract["stardist_patch_size_px"])
        valid = (stride <= size and contract["sam_encoder_size"] == SAM_ENCODER_SIZE
                 and contract["padding"] == "constant_zero"
                 and contract["rgb_scaling"] == "valid_percentile_2_98"
                 and math.isclose(contract["sam_window_size_m"], size * res, rel_tol=1e-6))
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete model preprocessing contract.") from exc
    if not valid:
        raise ValueError("Inconsistent or unsupported model preprocessing settings.")
    return dict(contract)


def rgb_to_u8(image, valid):
    """Scale each HWC channel using real pixels only; keep padding black."""
    import numpy as np

    image = np.asarray(image, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(image).all(axis=-1)
    out = np.zeros(image.shape, dtype=np.uint8)
    if valid.any():
        for channel in range(image.shape[-1]):
            values = image[..., channel]
            lo, hi = np.percentile(values[valid], (2, 98))
            if hi > lo:
                out[..., channel][valid] = np.clip((values[valid] - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
    return out


def check_proposal_cache(root, preprocessing, *, overwrite=False):
    """Do not reuse proposals produced with a different ground-window contract."""
    import json
    import shutil

    root = Path(root)
    path = root / "preprocessing.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    has_rasters = any((root / "rasters").glob("*.tif"))
    if has_rasters and previous != preprocessing:
        if not overwrite:
            raise ValueError(
                "Existing proposals use different preprocessing settings. "
                "Choose a new output directory or set skip_existing_sam=False to regenerate them."
            )
        shutil.rmtree(root)
    if preprocessing is not None:
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(preprocessing, indent=2))
    elif path.exists():
        path.unlink()
