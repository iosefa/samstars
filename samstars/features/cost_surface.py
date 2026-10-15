from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
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
    """Scale finite values to 0–1, including constant or empty inputs."""
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    lo, hi = np.percentile(arr[finite], (2, 98))
    if hi <= lo:
        return np.zeros_like(arr)
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
    return _normalise(entropy(pan_u8, disk(3), mask=np.isfinite(pan).astype(np.uint8)))


def _resolve_image_bands(count, rgb_bands=None, nir_band=None, texture_band=None):
    """Validate one-based band selections without inferring a sensor layout."""
    rgb = (1, 2, 3) if rgb_bands is None else tuple(rgb_bands)
    if len(rgb) != 3:
        raise ValueError("rgb_bands must contain exactly three indices in red, green, blue order.")
    for name, values in (("rgb_bands", rgb), ("nir_band", (nir_band,)), ("texture_band", (texture_band,))):
        for value in values:
            if value is None and name != "rgb_bands":
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= count:
                raise ValueError(f"{name} must use integer band indices between 1 and {count}.")
    if len(set(rgb)) != 3:
        raise ValueError("rgb_bands must contain three distinct band indices.")
    if nir_band is not None and nir_band in rgb:
        raise ValueError("nir_band must be separate from the red, green, and blue bands.")
    return tuple(int(b) for b in rgb)


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
    """Weights for height edges, NDVI gaps, image texture, and experimental SLIC edges.

    Weights must be nonnegative and sum to one. Terms without a supplied
    NIR band or SLIC layer are omitted and the remaining weights are rescaled.
    """
    grad: float = 0.4
    gap: float = 0.3
    tex: float = 0.3
    slic: float = 0

    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        values = (self.grad, self.gap, self.tex, self.slic)
        if any(not np.isfinite(v) or v < 0 for v in values):
            raise ValueError("Weights must be finite and nonnegative.")
        s = sum(values)
        if abs(s - 1) > 1e-6:
            raise ValueError("Weights must sum to 1 (got {:.3f})".format(s))


def build_cost_surface(
    *,
    img_path: str | Path | None = None,
    chm_path: str | Path,
    rgb_bands: tuple[int, int, int] | None = None,
    nir_band: int | None = None,
    texture_band: int | None = None,
    wv3_path: str | Path | None = None,
    slic_path: str | Path | None = None,
    weights: tuple[float, float, float, float] = (0.4, 0.3, 0.3, 0),
    nodata_value: float = -9999.0,
) -> tuple[np.ndarray, dict]:
    """Build a 0–1 crown-boundary cost raster from aligned imagery and canopy height.

    Args:
        img_path: Three-band RGB or multiband imagery raster.
        chm_path: Single-band canopy heights in meters, on the imagery grid.
        rgb_bands: One-based red, green, blue indices. Defaults to `(1, 2, 3)`.
        nir_band: Optional near-infrared band for the NDVI gap term. With no NIR
            band, the cost uses height edges and image texture only.
        texture_band: Optional image band for texture. Defaults to the mean of
            the selected RGB channels.
        wv3_path: Legacy alias for `img_path`; band choices are still explicit.
        slic_path: Experimental SLIC labels on the imagery grid.
        weights: Height-edge, NDVI-gap, texture, and SLIC weights. Unavailable
            terms are omitted and the remaining weights are rescaled to sum to one.
        nodata_value: Output value at pixels with missing input data.

    Returns:
        cost: Float32 array of boundary costs. Higher values indicate stronger cues.
        profile: Rasterio profile for the single-band output.
    """
    if (img_path is None) == (wv3_path is None):
        raise ValueError("Provide img_path (or its legacy alias wv3_path), but not both.")
    w = CostSurfaceWeights(*weights)

    with rasterio.open(img_path if img_path is not None else wv3_path) as src:
        rgb_bands = _resolve_image_bands(src.count, rgb_bands, nir_band, texture_band)
        rgb = src.read(rgb_bands, masked=True).astype(np.float32).filled(np.nan)
        texture = rgb.mean(axis=0) if texture_band is None else src.read(
            texture_band, masked=True
        ).astype(np.float32).filled(np.nan)
        nir = None if nir_band is None else src.read(
            nir_band, masked=True
        ).astype(np.float32).filled(np.nan)
        profile = src.profile.copy()
        profile["bounds"] = rasterio.transform.array_bounds(
            profile["height"], profile["width"], profile["transform"]
        )

    chm, chm_profile = _read_band(chm_path)
    if (chm.shape != rgb.shape[1:] or chm_profile["crs"] != profile["crs"]
            or chm_profile["transform"] != profile["transform"]):
        raise ValueError("Imagery and CHM must have the same CRS, transform, and shape.")

    grad = _chm_gradient(chm)
    gap = 0.0 if nir is None else _normalise(1 - _ndvi(rgb[0], nir))
    tex = _texture_entropy(texture)
    valid = np.isfinite(chm) & np.isfinite(rgb).all(axis=0) & np.isfinite(texture)
    if nir is not None:
        valid &= np.isfinite(nir)

    if slic_path:
        if str(slic_path).lower().endswith(".gpkg"):
            slic_lab = _rasterise_slic_gpkg(slic_path, profile)
        else:                                  # assume raster label image
            slic_lab, _ = _read_band(slic_path)
        edge = _slic_edge(slic_lab)
    else:
        edge = 0.0

    w_gap = w.gap if nir is not None else 0.0
    w_slic = w.slic if slic_path is not None else 0.0
    total_weight = w.grad + w_gap + w.tex + w_slic
    if total_weight <= 0:
        raise ValueError("At least one available cost term must have a positive weight.")

    cost = (
        w.grad * grad +
        w_gap  * gap  +
        w.tex  * tex  +
        w_slic * edge
    ) / total_weight
    cost = np.clip(cost, 0, 1).astype(np.float32)

    cost[~valid | ~np.isfinite(cost)] = nodata_value
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
) -> Path:
    """Write a cost array and Rasterio profile to a single-band GeoTIFF.

    The parent directory is created when needed. Set `exist_ok=False` to
    reject an existing output file. Returns the output path.
    """
    out_path = Path(out_path)
    if not exist_ok and out_path.exists():
        raise FileExistsError(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(cost, 1)
    return out_path
