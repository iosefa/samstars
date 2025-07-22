#!/usr/bin/env python3
"""image_io_lazy.py – Lazy‑loading raster helpers (COG‑friendly)
──────────────────────────────────────────────────────────────────────────────
This is a **drop‑in** replacement for the original utilities, but:

* **No eager 40 GB read**: `open_geotiff()` now returns an `Image` whose
  data live on disk; arrays are fetched *per‑window* when you call
  `Image.read_window()`.
* Existing helpers (`crop_image_to_bbox`, `mask_image_with_polygon`,
  etc.) continue to work – they switch from `rasterio_obj.read()` to
  `image.read_window()`.
* `Image.to_image()` still works for quick visualisation; if the raster
  is huge, it will temporarily read the requested bands into RAM.

The public API (function names / params) is unchanged, so all calling
code can stay the same – it just uses far less memory and runs happily
in parallel.
"""

from __future__ import annotations

# stdlib
from typing import List, Tuple, Sequence

# 3rd‑party
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.features import geometry_mask
from PIL.Image import fromarray
from affine import Affine
import cv2
from scipy.ndimage import uniform_filter

###############################################################################
# ──────────────────────────────  NUMPY HELPERS  ─────────────────────────────
###############################################################################

def rescale_to_8bit(image: np.ndarray, *, min: int = 2, max: int = 98) -> np.ndarray:
    """Percentile stretch → uint8."""
    p_min, p_max = np.percentile(image, (min, max))
    if p_min == p_max:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = 255 * (image - p_min) / (p_max - p_min)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    eq = cv2.equalizeHist(gray)
    return np.stack((eq,) * 3, axis=-1)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if image.ndim == 3:
        chs = cv2.split(image)
        out = cv2.merge([clahe.apply(c) for c in chs])
    else:
        out = clahe.apply(image)
    return out


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    coeffs = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (rgb * coeffs).sum(axis=-1)


def variance_of_laplacian(gray: np.ndarray, win: int) -> np.ndarray:
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    mean  = uniform_filter(lap,       size=win)
    mean2 = uniform_filter(lap * lap, size=win)
    return mean2 - mean ** 2

###############################################################################
# ─────────────────────────────  SHARPNESS RASTER  ───────────────────────────
###############################################################################

def laplacian(in_path: str, out_path: str, win: int):
    with rasterio.open(in_path) as src:
        meta = src.meta.copy()
        vis_bands = [2, 3, 5]                       # B, G, R (WorldView)
        arr = src.read(vis_bands).astype(np.float32)
        band_min = arr.min(axis=(1, 2), keepdims=True)
        band_rng = np.ptp(arr, axis=(1, 2), keepdims=True) + 1e-8
        arr = (arr - band_min) / band_rng
        gray  = rgb_to_gray(arr.transpose(1, 2, 0))
        sharp = variance_of_laplacian(gray, win)
        lo, hi = np.percentile(sharp, [2, 98])
        sharp  = np.clip((sharp - lo) / (hi - lo), 0, 1)
        meta.update(dtype="float32", count=1, nodata=None)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(sharp.astype("float32"), 1)

###############################################################################
# ─────────────────────────────── LABEL UTILS ────────────────────────────────
###############################################################################

def label_segments(
    segments: gpd.GeoDataFrame,
    labelled_points: gpd.GeoDataFrame
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    mixed: List[str] = []
    labelled = segments.copy()
    intersections = gpd.sjoin(labelled, labelled_points, how="inner", predicate="intersects")
    for poly_id, grp in intersections.groupby(intersections.index):
        classes = grp["class"].unique()
        if len(classes) == 1:
            labelled.loc[poly_id, "feature_class"] = classes[0]
        else:
            mixed.append(grp["segment_id"].iloc[0])
    labelled = labelled[labelled["feature_class"].notna()]
    return labelled, mixed

###############################################################################
# ─────────────────────────────  RASTER CROP / MASK  ─────────────────────────
###############################################################################

def crop_image_to_bbox(image: "Image", geom):
    xmin, ymin, xmax, ymax = geom.bounds
    window = from_bounds(xmin, ymin, xmax, ymax, transform=image.transform)
    data   = image.read_window(window)             # (bands, rows, cols)
    trans  = image.rasterio_obj.window_transform(window)
    return data, trans


def mask_image_with_polygon(data: np.ndarray, poly, trans: Affine) -> np.ndarray:
    h, w = data.shape[1:]
    mask = geometry_mask([poly], transform=trans, invert=True, out_shape=(h, w))
    return np.where(mask[None, ...], data, np.nan)

###############################################################################
# ───────────────────────────────  IO HELPERS  ───────────────────────────────
###############################################################################

def _write_geotiff(pil_img, path: str, crs, transform: Affine):
    arr = np.array(pil_img).astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[None, ...]                       # add band dim
    bands, h, w = arr.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=bands,
        dtype=arr.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(arr)
    print(f"✓ GeoTIFF written → {path}")

###############################################################################
# ───────────────────────────────  IMAGE CLASS  ──────────────────────────────
###############################################################################
class Image:
    """Lazy raster wrapper. Keeps only a rasterio dataset handle in RAM."""

    def __init__(self, ds: rasterio.io.DatasetReader, bands: Sequence[int] | None = None, *, lazy: bool = True):
        self.rasterio_obj = ds                   # alias for older code
        self.ds           = ds
        self.transform    = ds.transform
        self.crs          = ds.crs
        self.band_indexes = list(bands) if bands else list(range(1, ds.count + 1))

        # Optional eager load (rarely used for huge rasters)
        self.img_data: np.ndarray | None
        if lazy:
            self.img_data = None
        else:
            self.img_data = ds.read(self.band_indexes).transpose(1, 2, 0).astype(np.float32)

    # ------------------------------------------------------------------
    def read_window(self, window: Window, *, indexes: Sequence[int] | None = None):
        """Read a window → (bands, rows, cols) float32."""
        return self.ds.read(window=window, indexes=indexes or self.band_indexes).astype(np.float32)

    # ------------------------------------------------------------------
    def to_image(self, bands: Sequence[int], *, p_min: int = 2, p_max: int = 98, stretch_type: str | None = None):
        if len(bands) != 3:
            raise ValueError("'bands' must have length 3 (RGB)")
        # read full raster if not in RAM
        if self.img_data is None:
            data = self.ds.read(indexes=[b for b in bands]).astype(np.float32).transpose(1, 2, 0)
        else:
            data = self.img_data[:, :, [b for b in bands]]
        rgb = rescale_to_8bit(data, min=p_min, max=p_max)
        if stretch_type == "histogram_equalization":
            rgb = apply_histogram_equalization(rgb)
        elif stretch_type == "clahe":
            rgb = apply_clahe(rgb)
        elif stretch_type not in (None, "linear"):
            raise ValueError(f"Unknown stretch_type: {stretch_type}")
        return fromarray(rgb.astype(np.uint8))

###############################################################################
# ─────────────────────────────  OPEN GEOTIFF  ───────────────────────────────
###############################################################################

def open_geotiff(path: str, bands: Sequence[int] | None = None, *, lazy: bool = True) -> Image:
    """Open a (CO)GeoTIFF lazily.

    Parameters
    ----------
    path   : str   –  file path
    bands  : list[int] | None – 1‑based band indexes to expose (default: all)
    lazy   : bool  – if False, eagerly load into RAM (not recommended for 40 GB)
    """
    ds = rasterio.open(path)
    return Image(ds, bands=bands, lazy=lazy)
