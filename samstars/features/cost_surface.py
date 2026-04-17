from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import warnings
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from scipy.ndimage import sobel
from skimage.filters.rank import entropy
from skimage.morphology import disk


def _read_band(path: str | Path, idx: int = 1):
    """Internal helper function."""
    with rasterio.open(path) as src:
        arr = src.read(idx, masked=True).astype(np.float32)
        profile = src.profile.copy()
    return arr.filled(np.nan), profile


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Internal helper function."""
    lo, hi = np.nanpercentile(arr, (2, 98))
    arr_clip = np.clip(arr, lo, hi)
    with np.errstate(invalid="ignore"):
        out = (arr_clip - lo) / (hi - lo)
    return np.nan_to_num(out)


def _chm_gradient(chm: np.ndarray) -> np.ndarray:
    """Internal helper function."""
    dx = sobel(chm, axis=1, mode="nearest")
    dy = sobel(chm, axis=0, mode="nearest")
    return _normalise(np.hypot(dx, dy))


def _ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Internal helper function."""
    return np.clip((nir - red) / (nir + red + 1e-9), -1, 1)


def _texture_entropy(pan: np.ndarray) -> np.ndarray:
    """Internal helper function."""
    pan_u8 = (_normalise(pan) * 255).astype(np.uint8)
    return _normalise(entropy(pan_u8, disk(3)))


def _slic_edge(label_img: np.ndarray) -> np.ndarray:
    """Internal helper function."""
    edge = np.zeros_like(label_img, dtype=np.uint8)
    edge[:-1, :] |= label_img[:-1, :] != label_img[1:, :]
    edge[:, :-1] |= label_img[:, :-1] != label_img[:, 1:]
    return _normalise(edge.astype(np.float32))


def _rasterise_slic_gpkg(gpkg_path: str | Path, tgt_profile: dict) -> np.ndarray:
    """Rasterise polygons (segment_id) onto target grid."""
    west, south, east, north = tgt_profile["bounds"]
    gdf = gpd.read_file(gpkg_path, bbox=(west, south, east, north))

    if gdf.empty:
        raise ValueError("SLIC GPKG has no polygons over this tile.")

    if gdf.crs != tgt_profile["crs"]:
        gdf = gdf.to_crs(tgt_profile["crs"])

    shapes = [
        (geom, int(seg))
        for geom, seg in zip(gdf.geometry, gdf["segment_id"])
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        raise ValueError("No valid SLIC polygons with 'segment_id' found.")

    H, W = tgt_profile["height"], tgt_profile["width"]
    return rasterize(
        shapes,
        out_shape=(H, W),
        transform=tgt_profile["transform"],
        fill=0,
        dtype=np.uint32,
        all_touched=False,
    )


@dataclass
class CostSurfaceWeights:
    grad: float = 0.4
    gap: float = 0.3
    tex: float = 0.3
    slic: float = 0

    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        s = self.grad + self.gap + self.tex + self.slic
        if abs(s - 1) > 1e-6:
            raise ValueError("Weights must sum to 1 (got {:.3f})".format(s))


def build_cost_surface(
    *,
    wv3_path: str | Path,
    chm_path: str | Path,
    slic_path: str | Path | None = None,
    weights: tuple[float, float, float, float] = (0.4, 0.3, 0.3, 0),
    nodata_value: float = -9999.0,
):
    """Build cost surface."""
    w = CostSurfaceWeights(*weights)

    with rasterio.open(wv3_path) as src:
        C, B, G, Y, R, RE, N1, N2 = src.read(masked=True).astype(np.float32)
        profile = src.profile.copy()
        profile["bounds"] = rasterio.transform.array_bounds(
            profile["height"], profile["width"], profile["transform"]
        )

    chm, _ = _read_band(chm_path)

    grad = _chm_gradient(chm)
    gap  = _normalise(1 - _ndvi(R, N1))
    tex  = _texture_entropy(C)

    if slic_path:
        if str(slic_path).lower().endswith(".gpkg"):
            slic_lab = _rasterise_slic_gpkg(slic_path, profile)
        else:                                  # assume raster label image
            slic_lab, _ = _read_band(slic_path)
        edge = _slic_edge(slic_lab)
        w_slic = w.slic
    else:
        edge = 0.0
        s = w.grad + w.gap + w.tex
        w.grad, w.gap, w.tex = (w.grad / s, w.gap / s, w.tex / s)
        w_slic = 0.0
        warnings.warn("No SLIC provided – cost built from 3 terms only.")

    cost = (
        w.grad * grad +
        w.gap  * gap  +
        w.tex  * tex  +
        w_slic * edge
    )
    cost = np.clip(cost, 0, 1).astype(np.float32)

    cost[np.isnan(cost)] = nodata_value
    profile.update(count=1, dtype="float32",
                   compress="deflate",
                   nodata=nodata_value)
    return cost, profile


def write_cost_surface(
    cost: np.ndarray,
    profile: dict,
    out_path: str | Path,
    *,
    exist_ok: bool = True,
):
    """Write cost surface."""
    out_path = Path(out_path)
    if not exist_ok and out_path.exists():
        raise FileExistsError(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(cost, 1)
    return out_path
