from __future__ import annotations
import math, sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol, xy
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from sklearn.cluster import DBSCAN
from samstars._prompts import validate_canopy_height


def detect_peaks(raster_path: Path | str,
                 origin: str,
                 d_min: float,
                 min_dist_px: int,
                 sigma: int = 0, *, threshold_on_raw: bool = False) -> gpd.GeoDataFrame:
    """Detect peaks."""
    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(raster_path)

    with rasterio.open(raster_path) as src:
        arr = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        tfm, crs = src.transform, src.crs

    raw = arr.copy()
    valid = np.isfinite(raw)
    if sigma > 0:
        if threshold_on_raw:
            weights = gaussian_filter(valid.astype(np.float32), sigma=sigma)
            values = gaussian_filter(np.where(valid, raw, 0), sigma=sigma)
            arr = np.divide(values, weights, out=np.zeros_like(values), where=weights > 0)
        else:
            arr = gaussian_filter(arr, sigma=sigma)
    if threshold_on_raw:
        arr[~valid] = -np.inf

    mask_local_max = arr == maximum_filter(arr, size=2 * min_dist_px + 1)
    peaks = mask_local_max & valid & ((raw if threshold_on_raw else arr) >= d_min)
    rows, cols = np.where(peaks)
    xs, ys = xy(tfm, rows, cols, offset="center")
    heights = (raw if threshold_on_raw else arr)[rows, cols]

    return gpd.GeoDataFrame(
        dict(id=np.arange(rows.size),
             height=heights,
             origin=origin),
        geometry=gpd.points_from_xy(xs, ys),
        crs=crs
    )


def _sample_raster_values(
    raster_path: Path | str,
    xs: np.ndarray,
    ys: np.ndarray,
    *, legacy_round: bool = False,
) -> np.ndarray:
    """Sample raster values at world-coordinate point locations."""
    with rasterio.open(raster_path) as src:
        arr = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        rr, cc = rowcol(src.transform, xs, ys, op=float)

    # Pixel centres have fractional coordinates n + 0.5: rounding can select
    # the neighbouring cell (or leave the raster at its last row/column).
    sample_index = np.rint if legacy_round else np.floor
    rr = sample_index(rr).astype(int)
    cc = sample_index(cc).astype(int)
    valid = (
        (rr >= 0) & (rr < arr.shape[0]) &
        (cc >= 0) & (cc < arr.shape[1])
    )
    out = np.full(xs.shape, np.nan, dtype=np.float32)
    out[valid] = arr[rr[valid], cc[valid]]
    return out


def build_sparse_cost_graph(xs: np.ndarray,
                            ys: np.ndarray,
                            cost: np.ndarray,
                            tfm,
                            weight: float,
                            xy_thresh: float,
                            merge_radius: float,
                            samples: int = 8) -> "coo_matrix":
    """Build sparse cost graph."""
    n = len(xs)
    kd = cKDTree(np.c_[xs, ys])
    rows, cols, data = [], [], []
    ts = np.linspace(0., 1., samples + 2, dtype=np.float32)[1:-1]

    search_r = merge_radius + xy_thresh + 1e-6
    for i in range(n):
        xi, yi = xs[i], ys[i]
        neigh = kd.query_ball_point([xi, yi], search_r)
        neigh = [j for j in neigh if j > i]

        for j in neigh:
            dx, dy = xs[j] - xi, ys[j] - yi
            xy_dist = math.hypot(dx, dy)
            if xy_dist == 0.:
                continue

            if xy_dist <= xy_thresh or weight == 0.:
                d_eff = xy_dist
            else:
                xs_line = xi + ts * dx
                ys_line = yi + ts * dy
                rr, cc = rowcol(tfm, xs_line, ys_line, op=float)
                rr = np.clip(rr.round().astype(int), 0, cost.shape[0] - 1)
                cc = np.clip(cc.round().astype(int), 0, cost.shape[1] - 1)
                mean_cost = cost[rr, cc].mean()
                d_eff = xy_dist * (1. + weight * mean_cost)

            if d_eff <= merge_radius:
                rows.extend((i, j)); cols.extend((j, i))
                data.extend((d_eff, d_eff))

    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


@dataclass
class SeedParams:
    """Seed detection settings for aligned imagery-grid rasters.

    Attributes:
        chm_raster: Canopy height GeoTIFF in meters.
        den_raster: First-return counts or density GeoTIFF.
        cost_raster: Boundary cost GeoTIFF on the same grid.
        d_min: Minimum peak value in both CHM and density; interpret against each
            raster's units.
        min_dist_px: Peak-search radius in pixels.
        gauss_sigma: Gaussian smoothing scale in pixels before peak detection.
        merge_radius: Candidate grouping distance in projected CRS units.
        cost_weight: Strength of the cost penalty between candidate seeds.
        min_canopy_height: Minimum canopy height in meters at a candidate seed.
    """
    chm_raster: Path
    den_raster: Path
    cost_raster: Path
    d_min: float = 4.5
    min_dist_px: int = 4
    gauss_sigma: int = 2
    eps_scale: float = 0.15
    min_eps: float = 1.0
    max_eps: float = 6.0
    z_thresh: float = 1.0
    min_samples: int = 1
    merge_radius: float = 3.0
    cost_weight: float = 3.0
    xy_thresh: float = 0.3
    samples: int = 8
    dz_merge: float = 1.0
    min_canopy_height: float = 2.0
    debug_dist: bool = False
    rule_version: int = 1

    def __post_init__(self):
        self.min_canopy_height = validate_canopy_height(self.min_canopy_height)
        if self.rule_version not in (1, 2):
            raise ValueError('Unknown seed rule version.')
        if isinstance(self.min_samples, bool) or not isinstance(self.min_samples, int) or self.min_samples < 1:
            raise ValueError('min_samples must be a positive integer.')


def canonical_seeds(params: SeedParams) -> gpd.GeoDataFrame:
    """Detect CHM and density peaks and return one point per candidate cluster.

    Args:
        params: Raster paths and detection controls. Rasters must share a CRS,
            pixel grid, and extent; use a projected CRS in meters.

    Returns:
        A GeoDataFrame with id and geometry columns in the input CRS. Save it
        with to_file to create seeds.gpkg. Raises SystemExit if no candidates
        survive detection and canopy-height filtering.
    """
    consistent = params.rule_version == 2
    chm_peaks = detect_peaks(params.chm_raster, "chm",
                             params.min_canopy_height if consistent else params.d_min,
                             params.min_dist_px, params.gauss_sigma, threshold_on_raw=consistent)
    den_peaks = detect_peaks(params.den_raster, "den",
                             params.d_min, params.min_dist_px, params.gauss_sigma)

    seeds = gpd.GeoDataFrame(pd.concat([chm_peaks, den_peaks], ignore_index=True),
                             geometry="geometry", crs=chm_peaks.crs)
    if seeds.empty:
        sys.exit("No seeds after CHM & density sampling.")

    seeds["x"], seeds["y"] = seeds.geometry.x, seeds.geometry.y
    seeds["canopy_height"] = _sample_raster_values(
        params.chm_raster,
        seeds["x"].to_numpy(),
        seeds["y"].to_numpy(),
        legacy_round=not consistent,
    )
    seeds = seeds[np.isfinite(seeds["canopy_height"])]
    seeds = seeds[seeds["canopy_height"] >= params.min_canopy_height].reset_index(drop=True)
    if seeds.empty:
        sys.exit(
            "No seeds after applying the CHM canopy-height filter "
            f"(min_canopy_height={params.min_canopy_height})."
        )

    if consistent:
        # Density values are counts, never tree heights. All height-based
        # comparisons use the CHM sampled at the same physical candidate.
        seeds['height'] = seeds['canopy_height']
        seeds = seeds.drop_duplicates(['x', 'y']).reset_index(drop=True)

    with rasterio.open(params.cost_raster) as src:
        cost = src.read(1).astype(np.float32)
        nodata = src.nodatavals[0] if src.nodatavals[0] is not None else np.nan
        cost[cost == nodata] = 1.0
        tfm = src.transform

    xs, ys = seeds["x"].values, seeds["y"].values
    G = build_sparse_cost_graph(xs, ys, cost, tfm,
                                params.cost_weight,
                                params.xy_thresh,
                                params.merge_radius,
                                samples=params.samples)

    if params.debug_dist and G.nnz:
        print(f"d_eff  min/median/max = "
              f"{G.data.min():.2f} / {np.median(G.data):.2f} / {G.data.max():.2f}")

    db2 = DBSCAN(eps=params.merge_radius, min_samples=params.min_samples if consistent else 1,
                 metric="precomputed", n_jobs=-1).fit(G)
    seeds["cluster"] = db2.labels_
    if consistent:
        seeds = seeds[seeds['cluster'] >= 0].reset_index(drop=True)
        if seeds.empty:
            sys.exit('No seeds after candidate grouping.')

    # optional height‑based split inside a cluster
    if params.dz_merge > 0:
        new_cl = np.empty_like(seeds["cluster"])
        new_id = 0
        for old_id, sub in seeds.groupby("cluster"):
            if np.ptp(sub.height) <= params.dz_merge:
                new_cl[sub.index] = new_id
                new_id += 1
            else:
                mid = sub.height.median()
                lo = sub[sub.height <= mid].index
                hi = sub[sub.height >  mid].index
                for idx in (lo, hi):
                    if idx.size:
                        new_cl[idx] = new_id
                        new_id += 1
        seeds["cluster"] = new_cl


    if consistent:
        # Keep an observed canopy peak; averaging positions can put a seed
        # between trees or below the very height cutoff used to accept it.
        selected = seeds.groupby('cluster', sort=True)['canopy_height'].idxmax()
        centres = seeds.loc[selected, ['x', 'y', 'height']].reset_index(drop=True)
    else:
        centres = (seeds.groupby("cluster", as_index=False)
                   .agg(x=("x", "mean"), y=("y", "mean"), height=("height", "max")))
    centres["id"] = np.arange(len(centres))

    gdf_centres = gpd.GeoDataFrame(
        centres,
        geometry=gpd.points_from_xy(centres.x, centres.y),
        crs=seeds.crs
    )[["id", "geometry"]]
    return gdf_centres
