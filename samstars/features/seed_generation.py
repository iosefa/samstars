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


def detect_peaks(raster_path: Path | str,
                 origin: str,
                 d_min: float,
                 min_dist_px: int,
                 sigma: int = 0) -> gpd.GeoDataFrame:
    """Detect peaks."""
    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(raster_path)

    with rasterio.open(raster_path) as src:
        arr = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        tfm, crs = src.transform, src.crs

    if sigma > 0:
        arr = gaussian_filter(arr, sigma=sigma)

    mask_local_max = arr == maximum_filter(arr, size=2 * min_dist_px + 1)
    peaks = np.logical_and(mask_local_max, arr >= d_min)
    rows, cols = np.where(peaks)
    xs, ys = xy(tfm, rows, cols, offset="center")
    heights = arr[rows, cols]

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
) -> np.ndarray:
    """Sample raster values at world-coordinate point locations."""
    with rasterio.open(raster_path) as src:
        arr = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        rr, cc = rowcol(src.transform, xs, ys, op=float)

    rr = np.rint(rr).astype(int)
    cc = np.rint(cc).astype(int)
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


def canonical_seeds(params: SeedParams) -> gpd.GeoDataFrame:
    """Return **one** seed (centre point) per cluster."""
    chm_peaks = detect_peaks(params.chm_raster, "chm",
                             params.d_min, params.min_dist_px, params.gauss_sigma)
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
    )
    seeds = seeds[np.isfinite(seeds["canopy_height"])]
    seeds = seeds[seeds["canopy_height"] >= params.min_canopy_height].reset_index(drop=True)
    if seeds.empty:
        sys.exit(
            "No seeds after applying the CHM canopy-height filter "
            f"(min_canopy_height={params.min_canopy_height})."
        )

    pts_xy = seeds[["x", "y"]].values

    kd = cKDTree(pts_xy)
    cluster = -np.ones(len(seeds), dtype=int)
    cid = 0
    for i in range(len(seeds)):
        if cluster[i] != -1:
            continue
        eps = float(np.clip(params.eps_scale * seeds.height.iloc[i],
                            params.min_eps, params.max_eps))
        idx = kd.query_ball_point(pts_xy[i], eps)
        if params.z_thresh >= 0 and np.ptp(seeds.height.iloc[idx]) > params.z_thresh:
            continue
        if len(idx) >= params.min_samples:
            cluster[idx] = cid
            cid += 1
    seeds["cluster1"] = cluster

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

    db2 = DBSCAN(eps=params.merge_radius, min_samples=1,
                 metric="precomputed", n_jobs=-1).fit(G)
    seeds["cluster"] = db2.labels_

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


    centres = (seeds
               .groupby("cluster", as_index=False)
               .agg(x=("x", "mean"),
                    y=("y", "mean"),
                    height=("height", "max")))
    centres["id"] = np.arange(len(centres))

    gdf_centres = gpd.GeoDataFrame(
        centres,
        geometry=gpd.points_from_xy(centres.x, centres.y),
        crs=seeds.crs
    )[["id", "geometry"]]
    return gdf_centres
