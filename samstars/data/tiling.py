#!/usr/bin/env python3
"""
make_tiles.py – split IMG / CHM / COST / SEEDS into 500 m tiles (+20 m overlap)
               *with optional spatial thinning of seeds (0.7 m)*
"""

from __future__ import annotations
import sys, argparse, tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from shapely.geometry import box as Box
import geopandas as gpd
from scipy.spatial import cKDTree                     #  ← NEW
from tqdm.auto import tqdm

# local libraries
from samstars.features import build_cost_surface, canonical_seeds, SeedParams

# ───────────────────────── helpers ─────────────────────────────
def _align_to_img(src: Path | str, img_ds: rasterio.DatasetReader):
    """Internal helper function."""
    raw = rasterio.open(src)
    if (
        raw.crs != img_ds.crs
        or raw.transform != img_ds.transform
        or any(abs(a - b) > 1e-9 for a, b in zip(raw.res, img_ds.res))
    ):
        return WarpedVRT(raw, crs=img_ds.crs,
                         transform=img_ds.transform,
                         height=img_ds.height,
                         width=img_ds.width,
                         resampling=Resampling.bilinear), raw
    return raw, raw


def _write_temp_raster(arr: np.ndarray, transform, crs, path: Path):
    """Internal helper function."""
    count = arr.shape[0] if arr.ndim == 3 else 1
    meta = dict(driver="GTiff", width=arr.shape[-1], height=arr.shape[-2],
                count=count, dtype=arr.dtype, transform=transform, crs=crs)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr if count > 1 else arr[None, ...])


def _required_tile_outputs(tile_dir: Path, *, write_den: bool = False) -> list[Path]:
    """Internal helper function."""
    required = [
        tile_dir / "img.tif",
        tile_dir / "chm.tif",
        tile_dir / "cost.tif",
        tile_dir / "seeds.gpkg",
    ]
    if write_den:
        required.append(tile_dir / "den.tif")
    return required


def _is_empty_window(full: np.ndarray, nodata) -> bool:
    """Internal helper function."""
    if nodata is None:
        return False
    if np.isnan(nodata):
        return np.isnan(full).all()
    return (full == nodata).all()


def _spatial_thin(gdf: gpd.GeoDataFrame, dist: float) -> gpd.GeoDataFrame:
    """
    Greedy thinning: keep one point per <dist> metre neighbourhood.
    Points are processed in descending `height` order if that column exists;
    otherwise in input order.  Complexity O(N log N).
    """
    if gdf.empty or len(gdf) == 1:
        return gdf
    if "height" in gdf.columns:
        gdf = gdf.sort_values("height", ascending=False, ignore_index=True)
    coords = np.c_[gdf.geometry.x.values, gdf.geometry.y.values]
    tree   = cKDTree(coords)
    keep   = np.ones(len(gdf), bool)
    for i in range(len(gdf)):
        if not keep[i]:
            continue
        # mask neighbours j>i within radius
        idx = tree.query_ball_point(coords[i], dist)
        for j in idx:
            if j > i:
                keep[j] = False
    return gdf[keep].reset_index(drop=True)


def _prepare_tile(
    *,
    tile_dir: Path,
    img_ds: rasterio.DatasetReader,
    chm_ds,
    den_ds,
    img_tfm,
    img_crs,
    rgb_bands: tuple[int, int, int],
    base_meta: dict,
    win: Window,
    h: int,
    w: int,
    res: float,
    slic: Path | None,
    cost_weights,
    seed_params: SeedParams | None,
    seed_d_min: float,
    seed_min_dist_px: int,
    seed_gauss_sigma: int,
    seed_min_samples: int,
    seed_merge_radius: float,
    seed_min_canopy_height: float,
    thin_seeds: bool,
    thin_dist: float,
    write_den: bool,
    skip_existing: bool,
) -> bool:
    """Internal helper function."""
    required = _required_tile_outputs(tile_dir, write_den=write_den)
    if skip_existing and all(p.exists() for p in required):
        return True

    tfm = rasterio.windows.transform(win, img_tfm)
    full = img_ds.read(window=win)
    if _is_empty_window(full, img_ds.nodata):
        return False

    rgb = full[[b - 1 for b in rgb_bands]]
    chm_tile = chm_ds.read(1, window=win).astype(np.float32)
    den_tile = den_ds.read(1, window=win).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpd:
        tmpd = Path(tmpd)
        _write_temp_raster(full, tfm, img_crs, tmpd / "wv3.tif")
        _write_temp_raster(chm_tile, tfm, img_crs, tmpd / "chm.tif")
        _write_temp_raster(den_tile, tfm, img_crs, tmpd / "den.tif")

        cost_arr, _ = build_cost_surface(
            wv3_path=tmpd / "wv3.tif",
            chm_path=tmpd / "chm.tif",
            slic_path=slic,
            weights=cost_weights,
        )
        _write_temp_raster(cost_arr, tfm, img_crs, tmpd / "cost.tif")

        sp = seed_params or SeedParams(
            chm_raster=tmpd / "chm.tif",
            den_raster=tmpd / "den.tif",
            cost_raster=tmpd / "cost.tif",
            d_min=seed_d_min,
            min_dist_px=seed_min_dist_px,
            gauss_sigma=seed_gauss_sigma,
            min_samples=seed_min_samples,
            merge_radius=seed_merge_radius,
            min_canopy_height=seed_min_canopy_height,
        )
        try:
            seeds = canonical_seeds(sp)
        except SystemExit:
            return False

    west, north = tfm.c, tfm.f
    bbox = Box(west, north - h * res, west + w * res, north)
    seeds_tile = seeds[seeds.intersects(bbox)]
    if thin_seeds and not seeds_tile.empty:
        seeds_tile = _spatial_thin(seeds_tile, thin_dist)
    if seeds_tile.empty:
        return False

    tile_dir.mkdir(parents=True, exist_ok=True)

    meta_img = base_meta | {"height": h, "width": w, "transform": tfm}
    with rasterio.open(tile_dir / "img.tif", "w", **meta_img) as dst:
        dst.write(rgb)

    def _w1(arr, name):
        """Internal helper function."""
        meta = meta_img | {"count": 1, "dtype": "float32"}
        with rasterio.open(tile_dir / name, "w", **meta) as dst:
            dst.write(arr.astype("float32"), 1)

    _w1(chm_tile, "chm.tif")
    _w1(cost_arr, "cost.tif")
    if write_den:
        _w1(den_tile, "den.tif")
    seeds_tile.to_file(tile_dir / "seeds.gpkg", driver="GPKG")
    return True

# ───────────────────────── main builder ────────────────────────
def build_tiles(*,
                img: Path,
                chm: Path,
                den: Path,
                slic: Path | None = None,
                out: Path = Path("tiles"),
                tile: float = 500.0,
                overlap: float = 20.0,
                cost_weights=(0.4, 0.3, 0.3, 0.0),
                seed_params: SeedParams | None = None,
                thin_seeds: bool = True,                    # ← NEW
                thin_dist: float = 0.8,
                seed_d_min: float = 4.5,
                seed_min_dist_px: int = 4,
                seed_gauss_sigma: int = 2,
                seed_min_samples: int = 1,
                seed_merge_radius: float = 3.0,
                seed_min_canopy_height: float = 2.0,
                write_den: bool = False,                    # ← NEW
                skip_existing: bool = True) -> None:

    """Build tiles."""
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(img) as img_ds:
        res = abs(img_ds.transform.a)
        assert abs(res - abs(img_ds.transform.e)) < 1e-6, "non‑square pixels"

        img_h, img_w = img_ds.height, img_ds.width
        img_tfm, img_crs = img_ds.transform, img_ds.crs
        rgb_bands = (7, 5, 3) if img_ds.count >= 7 else (1, 2, 3)

        base_meta = img_ds.profile | {"driver": "GTiff", "count": 3,
                                      "compress": "deflate", "predictor": 2,
                                      "zlevel": 6}

        chm_ds, chm_raw = _align_to_img(chm, img_ds)
        den_ds, den_raw = _align_to_img(den, img_ds)

        tile_px   = int(round(tile / res))
        stride_px = int(round((tile - overlap) / res))
        tiles = [(ty, tx, r0, c0,
                  min(tile_px, img_h - r0),
                  min(tile_px, img_w - c0))
                 for ty, r0 in enumerate(range(0, img_h, stride_px))
                 for tx, c0 in enumerate(range(0, img_w, stride_px))
                 if r0 < img_h and c0 < img_w]

        pbar = tqdm(total=len(tiles), desc="Tiling", unit="tile")
        for ty, tx, r0, c0, h, w in tiles:
            tile_dir = out / f"tile_{ty:03d}_{tx:03d}"
            win = Window(c0, r0, w, h)
            _prepare_tile(
                tile_dir=tile_dir,
                img_ds=img_ds,
                chm_ds=chm_ds,
                den_ds=den_ds,
                img_tfm=img_tfm,
                img_crs=img_crs,
                rgb_bands=rgb_bands,
                base_meta=base_meta,
                win=win,
                h=h,
                w=w,
                res=res,
                slic=slic,
                cost_weights=cost_weights,
                seed_params=seed_params,
                seed_d_min=seed_d_min,
                seed_min_dist_px=seed_min_dist_px,
                seed_gauss_sigma=seed_gauss_sigma,
                seed_min_samples=seed_min_samples,
                seed_merge_radius=seed_merge_radius,
                seed_min_canopy_height=seed_min_canopy_height,
                thin_seeds=thin_seeds,
                thin_dist=thin_dist,
                write_den=write_den,
                skip_existing=skip_existing,
            )
            pbar.update(1)

        pbar.close()
        chm_ds.close(); den_ds.close()
        if chm_raw is not chm_ds: chm_raw.close()
        if den_raw is not den_ds: den_raw.close()
    print("✓ Tiles written to", out.resolve())


def build_single_tile(*,
                      img: Path,
                      chm: Path,
                      den: Path,
                      slic: Path | None = None,
                      out: Path = Path("tiles"),
                      tile_name: str = "tile_000_000",
                      cost_weights=(0.4, 0.3, 0.3, 0.0),
                      seed_params: SeedParams | None = None,
                      thin_seeds: bool = True,
                      thin_dist: float = 0.8,
                      seed_d_min: float = 4.5,
                      seed_min_dist_px: int = 4,
                      seed_gauss_sigma: int = 2,
                      seed_min_samples: int = 1,
                      seed_merge_radius: float = 3.0,
                      seed_min_canopy_height: float = 2.0,
                      write_den: bool = False,
                      skip_existing: bool = True) -> Path:
    """Prepare a single tile-style input directory from one scene."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    with rasterio.open(img) as img_ds:
        res = abs(img_ds.transform.a)
        assert abs(res - abs(img_ds.transform.e)) < 1e-6, "non-square pixels"

        rgb_bands = (7, 5, 3) if img_ds.count >= 7 else (1, 2, 3)
        base_meta = img_ds.profile | {
            "driver": "GTiff",
            "count": 3,
            "compress": "deflate",
            "predictor": 2,
            "zlevel": 6,
        }
        img_tfm, img_crs = img_ds.transform, img_ds.crs
        chm_ds, chm_raw = _align_to_img(chm, img_ds)
        den_ds, den_raw = _align_to_img(den, img_ds)
        try:
            tile_dir = out / tile_name
            win = Window(0, 0, img_ds.width, img_ds.height)
            written = _prepare_tile(
                tile_dir=tile_dir,
                img_ds=img_ds,
                chm_ds=chm_ds,
                den_ds=den_ds,
                img_tfm=img_tfm,
                img_crs=img_crs,
                rgb_bands=rgb_bands,
                base_meta=base_meta,
                win=win,
                h=img_ds.height,
                w=img_ds.width,
                res=res,
                slic=slic,
                cost_weights=cost_weights,
                seed_params=seed_params,
                seed_d_min=seed_d_min,
                seed_min_dist_px=seed_min_dist_px,
                seed_gauss_sigma=seed_gauss_sigma,
                seed_min_samples=seed_min_samples,
                seed_merge_radius=seed_merge_radius,
                seed_min_canopy_height=seed_min_canopy_height,
                thin_seeds=thin_seeds,
                thin_dist=thin_dist,
                write_den=write_den,
                skip_existing=skip_existing,
            )
        finally:
            chm_ds.close()
            den_ds.close()
            if chm_raw is not chm_ds:
                chm_raw.close()
            if den_raw is not den_ds:
                den_raw.close()

    if not written:
        raise RuntimeError("No valid tile outputs generated from the provided inputs.")

    print("✓ Single-tile inputs written to", out.resolve())
    return out

# ───────────────────── CLI wrapper ─────────────────────────────
def _parse_cli():
    """Parse command-line arguments."""
    pa = argparse.ArgumentParser("make_tiles.py – per‑tile cost + seeds")
    pa.add_argument("--img", required=True, type=Path)
    pa.add_argument("--chm", required=True, type=Path)
    pa.add_argument("--den", required=True, type=Path)
    pa.add_argument("--slic", type=Path, default=None)
    pa.add_argument("--out",  type=Path, default=Path("tiles"))
    pa.add_argument("--tile", type=float, default=500.0)
    pa.add_argument("--overlap", type=float, default=20.0)
    pa.add_argument("--weights", nargs=4, type=float,
                    default=(0.4, 0.3, 0.3, 0.0),
                    metavar=("GRAD", "GAP", "TEX", "SLIC"))
    # new flag  → thinning ON by default; pass --no-thin to disable
    thin_grp = pa.add_mutually_exclusive_group()
    thin_grp.add_argument("--thin",     dest="thin", action="store_true",  default=True,
                          help="Spatially thin seeds at 0.7 m (default)")
    thin_grp.add_argument("--no-thin",  dest="thin", action="store_false",
                          help="Keep all seeds (no thinning)")
    pa.add_argument("--write-den", action="store_true",
                    help="Write per-tile den.tif (density raster)")
    pa.add_argument("--seed-d-min", type=float, default=4.5,
                    help="Minimum peak value for CHM/density seed candidates (default: 4.5)")
    pa.add_argument("--seed-min-dist-px", type=int, default=4,
                    help="Peak suppression radius in pixels for seed candidates (default: 4)")
    pa.add_argument("--seed-gauss-sigma", type=int, default=2,
                    help="Gaussian smoothing sigma in pixels before peak finding (default: 2)")
    pa.add_argument("--seed-min-samples", type=int, default=1,
                    help="Minimum nearby candidates needed to keep a seed cluster (default: 1)")
    pa.add_argument("--seed-merge-radius", type=float, default=3.0,
                    help="Maximum effective distance in metres when merging nearby seed candidates (default: 3.0)")
    pa.add_argument("--seed-min-canopy-height", type=float, default=2.0,
                    help="Drop seeds where local CHM is below this height in metres (default: 2.0)")
    skip_grp = pa.add_mutually_exclusive_group()
    skip_grp.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True,
                          help="Skip tiles that already have complete outputs (default)")
    skip_grp.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                          help="Recompute and overwrite all tiles")
    return pa.parse_args()

def _main():
    """Internal helper function."""
    ns = _parse_cli()
    try:
        build_tiles(img=ns.img, chm=ns.chm, den=ns.den,
                    slic=ns.slic, out=ns.out,
                    tile=ns.tile, overlap=ns.overlap,
                    cost_weights=tuple(ns.weights),
                    thin_seeds=ns.thin,
                    seed_d_min=ns.seed_d_min,
                    seed_min_dist_px=ns.seed_min_dist_px,
                    seed_gauss_sigma=ns.seed_gauss_sigma,
                    seed_min_samples=ns.seed_min_samples,
                    seed_merge_radius=ns.seed_merge_radius,
                    seed_min_canopy_height=ns.seed_min_canopy_height,
                    write_den=ns.write_den,
                    skip_existing=ns.skip_existing)
    except Exception as e:
        print("Error:", e, file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    _main()
