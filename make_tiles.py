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
from sam.cost import build_cost_surface
from sam.seed import canonical_seeds, SeedParams

# ───────────────────────── helpers ─────────────────────────────
def _align_to_img(src: Path | str, img_ds: rasterio.DatasetReader):
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
    count = arr.shape[0] if arr.ndim == 3 else 1
    meta = dict(driver="GTiff", width=arr.shape[-1], height=arr.shape[-2],
                count=count, dtype=arr.dtype, transform=transform, crs=crs)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr if count > 1 else arr[None, ...])


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
                thin_dist: float = 0.8) -> None:            # ← NEW

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
            win  = Window(c0, r0, w, h)
            tfm  = rasterio.windows.transform(win, img_tfm)
            full = img_ds.read(window=win)
            if np.all(full == img_ds.nodata):
                pbar.update(1); continue
            rgb  = full[[b-1 for b in rgb_bands]]

            chm_tile = chm_ds.read(1, window=win).astype(np.float32)
            den_tile = den_ds.read(1, window=win).astype(np.float32)

            with tempfile.TemporaryDirectory() as tmpd:
                tmpd = Path(tmpd)
                _write_temp_raster(full,     tfm, img_crs, tmpd/"wv3.tif")
                _write_temp_raster(chm_tile, tfm, img_crs, tmpd/"chm.tif")
                _write_temp_raster(den_tile, tfm, img_crs, tmpd/"den.tif")

                cost_arr, _ = build_cost_surface(
                    wv3_path=tmpd/"wv3.tif", chm_path=tmpd/"chm.tif",
                    slic_path=slic, weights=cost_weights)
                _write_temp_raster(cost_arr, tfm, img_crs, tmpd/"cost.tif")

                sp = seed_params or SeedParams(
                    chm_raster=tmpd/"chm.tif",
                    den_raster=tmpd/"den.tif",
                    cost_raster=tmpd/"cost.tif")
                seeds = canonical_seeds(sp)

            # crop seeds to tile & optionally thin ---------------------------
            west, north = tfm.c, tfm.f
            bbox = Box(west, north - h*res, west + w*res, north)
            seeds_tile = seeds[seeds.intersects(bbox)]
            if thin_seeds and not seeds_tile.empty:
                seeds_tile = _spatial_thin(seeds_tile, thin_dist)

            # write outputs ---------------------------------------------------
            tile_dir = out / f"tile_{ty:03d}_{tx:03d}"
            tile_dir.mkdir(parents=True, exist_ok=True)

            meta_img = base_meta | {"height": h, "width": w, "transform": tfm}
            with rasterio.open(tile_dir/"img.tif", "w", **meta_img) as dst:
                dst.write(rgb)

            def _w1(arr, name):
                m = meta_img | {"count": 1, "dtype": "float32"}
                with rasterio.open(tile_dir/name, "w", **m) as d:
                    d.write(arr.astype("float32"), 1)

            _w1(chm_tile,  "chm.tif")
            _w1(cost_arr,  "cost.tif")
            if not seeds_tile.empty:
                seeds_tile.to_file(tile_dir/"seeds.gpkg", driver="GPKG")

            pbar.update(1)

        pbar.close()
        chm_ds.close(); den_ds.close()
        if chm_raw is not chm_ds: chm_raw.close()
        if den_raw is not den_ds: den_raw.close()
    print("✓ Tiles written to", out.resolve())

# ───────────────────── CLI wrapper ─────────────────────────────
def _parse_cli():
    pa = argparse.ArgumentParser("make_tiles.py – per‑tile cost + seeds")
    pa.add_argument("--img", required=True, type=Path)
    pa.add_argument("--chm", required=True, type=Path)
    pa.add_argument("--den", required=True, type=Path)
    pa.add_argument("--slic", type=Path, default=None)
    pa.add_argument("--out",  type=Path, default=Path("tiles_new"))
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
    return pa.parse_args()

def _main():
    ns = _parse_cli()
    try:
        build_tiles(img=ns.img, chm=ns.chm, den=ns.den,
                    slic=ns.slic, out=ns.out,
                    tile=ns.tile, overlap=ns.overlap,
                    cost_weights=tuple(ns.weights),
                    thin_seeds=ns.thin)
    except Exception as e:
        print("Error:", e, file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    _main()

# #!/usr/bin/env python3
# """
# make_tiles.py – split IMG / CHM / COST / SEEDS into 500 m tiles (+20 m overlap)
#
# tiles/
#  ├─ tile_000_000/
#  │   ├─ img.tif    (RGB 7-5-3 or 1-2-3, striped GeoTIFF)
#  │   ├─ chm.tif    (float32)
#  │   ├─ cost.tif   (float32)
#  │   └─ seeds.gpkg (per-tile seed points)
# """
#
# from __future__ import annotations
# import sys, argparse, tempfile
# from pathlib import Path
#
# import numpy as np
# import rasterio
# from rasterio.enums import Resampling
# from rasterio.vrt import WarpedVRT
# from rasterio.windows import Window
# from shapely.geometry import box as Box
# import geopandas as gpd
# from tqdm.auto import tqdm
#
# # local library calls (change the import paths if yours differ)
# from sam.cost import build_cost_surface
# from sam.seed import canonical_seeds, SeedParams
#
#
# def _align_to_img(src: Path | str, img_ds: rasterio.DatasetReader):
#     """Return (dataset aligned to IMG grid, original dataset handle)."""
#     raw = rasterio.open(src)
#     if (
#         raw.crs != img_ds.crs
#         or raw.transform != img_ds.transform
#         or any(abs(a - b) > 1e-9 for a, b in zip(raw.res, img_ds.res))
#     ):
#         vrt = WarpedVRT(
#             raw,
#             crs=img_ds.crs,
#             transform=img_ds.transform,
#             height=img_ds.height,
#             width=img_ds.width,
#             resampling=Resampling.bilinear,
#         )
#         return vrt, raw
#     return raw, raw
#
#
# def _write_temp_raster(arr: np.ndarray, transform, crs, path: Path):
#     """Write *arr* (H×W or C×H×W) to GeoTIFF *path* with given georef."""
#     count = arr.shape[0] if arr.ndim == 3 else 1
#     dtype = arr.dtype
#     meta = dict(
#         driver="GTiff",
#         width=arr.shape[-1],
#         height=arr.shape[-2],
#         count=count,
#         dtype=dtype,
#         transform=transform,
#         crs=crs,
#     )
#     with rasterio.open(path, "w", **meta) as dst:
#         if count == 1:
#             dst.write(arr, 1)
#         else:
#             dst.write(arr)
#
#
# # ──────────────────────────── main tile builder ────────────────────────────
# def build_tiles(*,
#                 img: Path,
#                 chm: Path,
#                 den: Path,
#                 slic: Path | None = None,
#                 out: Path = Path("tiles"),
#                 tile: float = 500.0,
#                 overlap: float = 20.0,
#                 cost_weights: tuple[float, float, float, float] = (0.4, 0.3, 0.3, 0.0),
#                 seed_params: SeedParams | None = None) -> None:
#
#     out.mkdir(parents=True, exist_ok=True)
#
#     # open reference image (assumed full WV-3 stack)
#     with rasterio.open(img) as img_ds:
#         res = abs(img_ds.transform.a)                         # metres / pixel
#         assert abs(res - abs(img_ds.transform.e)) < 1e-6, "pixels not square"
#
#         img_h, img_w = img_ds.height, img_ds.width
#         img_tfm, img_crs = img_ds.transform, img_ds.crs
#
#         rgb_bands = (7, 5, 3) if img_ds.count >= 7 else (1, 2, 3)
#         base_meta = img_ds.profile.copy()
#         base_meta.update(driver="GTiff", count=3, compress="deflate",
#                          predictor=2, zlevel=6)
#
#         # prepare CHM & density readers (may be reprojected on-the-fly)
#         chm_ds, chm_raw = _align_to_img(chm, img_ds)
#         den_ds, den_raw = _align_to_img(den, img_ds)
#
#         # tiling grid
#         tile_px   = int(round(tile / res))
#         stride_px = int(round((tile - overlap) / res))
#         tiles = [
#             (ty, tx, r0, c0,
#              min(tile_px, img_h - r0),
#              min(tile_px, img_w - c0))
#             for ty, r0 in enumerate(range(0, img_h, stride_px))
#             for tx, c0 in enumerate(range(0, img_w, stride_px))
#             if r0 < img_h and c0 < img_w
#         ]
#
#         pbar = tqdm(total=len(tiles), desc="Tiling", unit="tile")
#         for ty, tx, r0, c0, h, w in tiles:
#             win = Window(c0, r0, w, h)
#             tfm = rasterio.windows.transform(win, img_tfm)
#
#             # read full-band WV-3 window (for cost) and RGB subset (for output)
#             img_tile_full = img_ds.read(window=win)           # all bands
#             if np.all(img_tile_full == img_ds.nodata):
#                 pbar.update(1)
#                 continue
#             img_tile_rgb = img_tile_full[[b - 1 for b in rgb_bands], :, :]
#
#             chm_tile = chm_ds.read(1, window=win).astype(np.float32)
#             den_tile = den_ds.read(1, window=win).astype(np.float32)
#
#             with tempfile.TemporaryDirectory() as tmp:
#                 tmp = Path(tmp)
#
#                 # write temp rasters
#                 tmp_wv3 = tmp / "wv3.tif"
#                 tmp_chm = tmp / "chm.tif"
#                 tmp_den = tmp / "den.tif"
#
#                 _write_temp_raster(img_tile_full, tfm, img_crs, tmp_wv3)
#                 _write_temp_raster(chm_tile,      tfm, img_crs, tmp_chm)
#                 _write_temp_raster(den_tile,      tfm, img_crs, tmp_den)
#
#                 # --- COST (per-tile) ---
#                 cost_arr, _ = build_cost_surface(
#                     wv3_path=tmp_wv3,
#                     chm_path=tmp_chm,
#                     slic_path=slic,          # None or global SLIC; small window ok
#                     weights=cost_weights,
#                 )
#                 tmp_cost = tmp / "cost.tif"
#                 _write_temp_raster(cost_arr, tfm, img_crs, tmp_cost)
#
#                 # --- SEEDS (per-tile) ---
#                 sp = seed_params or SeedParams(
#                     chm_raster=tmp_chm,
#                     den_raster=tmp_den,
#                     cost_raster=tmp_cost,
#                 )
#                 seeds_gdf = canonical_seeds(sp)
#
#             # crop seeds to exact bbox (extra safety)
#             west, north = tfm.c, tfm.f
#             east, south = west + w * res, north - h * res
#             bbox = Box(west, south, east, north)
#             seeds_tile = seeds_gdf[seeds_gdf.intersects(bbox)]
#
#             # write outputs ---------------------------------------------------
#             out_dir = out / f"tile_{ty:03d}_{tx:03d}"
#             out_dir.mkdir(parents=True, exist_ok=True)
#
#             meta_img = base_meta | {"height": h, "width": w, "transform": tfm}
#             with rasterio.open(out_dir / "img.tif", "w", **meta_img) as dst:
#                 dst.write(img_tile_rgb)
#
#             def _w_single(arr: np.ndarray, name: str):
#                 m = meta_img | {"count": 1, "dtype": "float32"}
#                 with rasterio.open(out_dir / name, "w", **m) as dst:
#                     dst.write(arr.astype("float32"), 1)
#
#             _w_single(chm_tile,  "chm.tif")
#             _w_single(cost_arr, "cost.tif")
#
#             if not seeds_tile.empty:
#                 seeds_tile.to_file(out_dir / "seeds.gpkg", driver="GPKG")
#
#             pbar.update(1)
#
#         pbar.close()
#         chm_ds.close(); den_ds.close()
#         if chm_raw is not chm_ds: chm_raw.close()
#         if den_raw is not den_ds: den_raw.close()
#
#     print(f"✓ Tiles written to {out.resolve()}")
#
#
# # ────────────────────────── CLI wrapper ──────────────────────────
# def _parse_cli():
#     p = argparse.ArgumentParser("make_tiles.py – per-tile cost + seeds")
#     p.add_argument("--img", required=True, type=Path)
#     p.add_argument("--chm", required=True, type=Path)
#     p.add_argument("--den", required=True, type=Path)
#     p.add_argument("--slic", type=Path, default=None)
#     p.add_argument("--out",  type=Path, default=Path("tiles"))
#     p.add_argument("--tile", type=float, default=500.0)
#     p.add_argument("--overlap", type=float, default=20.0)
#     p.add_argument("--weights", nargs=4, type=float,
#                    default=(0.4, 0.3, 0.3, 0.0),
#                    metavar=("GRAD", "GAP", "TEX", "SLIC"))
#     return p.parse_args()
#
#
# def _main():
#     ns = _parse_cli()
#     try:
#         build_tiles(img=ns.img, chm=ns.chm, den=ns.den,
#                     slic=ns.slic, out=ns.out,
#                     tile=ns.tile, overlap=ns.overlap,
#                     cost_weights=tuple(ns.weights))
#     except Exception as e:
#         print(f"Error: {e}", file=sys.stderr)
#         sys.exit(1)
#
#
# if __name__ == "__main__":
#     _main()
#
#
# # #!/usr/bin/env python3
# # """
# # make_tiles.py – split IMG / CHM / COST / SEEDS into 1 000 m tiles (+25 m overlap)
# #
# # tiles/
# #  ├─ tile_000_000/
# #  │   ├─ img.tif    (3 bands: 7-5-3 or 1-2-3, striped GeoTIFF)
# #  │   ├─ chm.tif    (1 band, float32, striped)
# #  │   ├─ cost.tif   (1 band, float32, striped)
# #  │   └─ seeds.gpkg (only if seeds intersect tile)
# #  …
# #
# # Run with defaults
# #     python make_tiles.py
# # Customise
# #     python make_tiles.py --tile 500 --overlap 10 --out my_tiles
# # """
# #
# # from __future__ import annotations
# # from pathlib import Path
# # import argparse
# #
# # import numpy as np
# # import rasterio
# # from rasterio.windows import Window
# # from rasterio.vrt import WarpedVRT
# # from rasterio.enums import Resampling
# # from shapely.geometry import box as Box
# # import geopandas as gpd
# # from tqdm.auto import tqdm
# #
# # # ----------------------------------------------------------------------
# # IMG_PATH   = Path("image.tif")   # multispectral cube
# # CHM_PATH   = Path("chm.tif")     # canopy-height model
# # COST_PATH  = Path("cost.tif")    # cost / boundary probability
# # SEEDS_PATH = Path("seeds.gpkg")  # seed points/clusters
# # # ----------------------------------------------------------------------
# #
# #
# # def align_to_img(src_path: Path, img_ds: rasterio.DatasetReader):
# #     """
# #     Return (dataset aligned to IMG grid, original dataset).
# #
# #     • If the source already matches IMG’s CRS, transform & res –> returned twice.
# #     • Otherwise wrap it in a WarpedVRT so window reads are in IMG pixels.
# #     """
# #     raw = rasterio.open(src_path)
# #     if (
# #         raw.crs != img_ds.crs
# #         or raw.transform != img_ds.transform
# #         or any(abs(a - b) > 1e-9 for a, b in zip(raw.res, img_ds.res))
# #     ):
# #         vrt = WarpedVRT(
# #             raw,
# #             crs=img_ds.crs,
# #             transform=img_ds.transform,
# #             height=img_ds.height,
# #             width=img_ds.width,
# #             resampling=Resampling.bilinear,
# #         )
# #         return vrt, raw
# #     return raw, raw
# #
# #
# # def main() -> None:
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--tile",    type=float, default=500, help="tile size [m]")
# #     ap.add_argument("--overlap", type=float, default=10,    help="overlap [m]")
# #     ap.add_argument("--out", default="tiles",               help="output dir")
# #     args = ap.parse_args()
# #
# #     out_root = Path(args.out).resolve()
# #     out_root.mkdir(parents=True, exist_ok=True)
# #
# #     # ── open master IMG once ─────────────────────────────────────
# #     with rasterio.open(IMG_PATH) as img_ds:
# #         res       = abs(img_ds.transform.a)      # m → px
# #         img_h, img_w = img_ds.height, img_ds.width
# #         img_tfm   = img_ds.transform
# #         img_crs   = img_ds.crs
# #
# #         # choose 3 bands (prefer 7-5-3)
# #         band_ids = (7, 5, 3) if img_ds.count >= 7 else (1, 2, 3)
# #         base_meta = img_ds.profile.copy()
# #         base_meta.update(
# #             driver="GTiff",
# #             count=3,
# #             dtype=img_ds.dtypes[0],
# #             compress="deflate",
# #             predictor=2,
# #             zlevel=6,      # **no “tiled” key** → striped TIFF avoids block-size issues
# #         )
# #
# #         # CHM and COST aligned to IMG grid
# #         chm_ds,  chm_raw  = align_to_img(CHM_PATH,  img_ds)
# #         cost_ds, cost_raw = align_to_img(COST_PATH, img_ds)
# #
# #         # seeds in IMG CRS
# #         seeds = gpd.read_file(SEEDS_PATH).to_crs(img_crs)
# #
# #         # ── tiling grid ──────────────────────────────────────────
# #         tile_px   = int(round(args.tile / res))
# #         stride_px = int(round((args.tile - args.overlap) / res))
# #
# #         tiles = [
# #             (ty, tx, row0, col0,
# #              min(tile_px, img_h - row0),
# #              min(tile_px, img_w - col0))
# #             for ty, row0 in enumerate(range(0, img_h, stride_px))
# #             for tx, col0 in enumerate(range(0, img_w, stride_px))
# #             if row0 < img_h and col0 < img_w
# #         ]
# #
# #         pbar = tqdm(total=len(tiles), desc="Tiling", unit="tile")
# #
# #         for ty, tx, row0, col0, h, w in tiles:
# #             win = Window(col0, row0, w, h)
# #             tfm = rasterio.windows.transform(win, img_tfm)
# #
# #             img_tile = img_ds.read(band_ids, window=win)
# #             if np.all(img_tile == img_ds.nodata):
# #                 pbar.update(1)
# #                 continue  # skip completely empty tiles
# #
# #             chm_tile  = chm_ds.read(1, window=win)
# #             cost_tile = cost_ds.read(1, window=win)
# #
# #             # bounding box for seeds selection
# #             west, north = tfm.c, tfm.f
# #             east  = west + w * res
# #             south = north - h * res
# #             bbox  = Box(west, south, east, north)
# #             seeds_tile = seeds[seeds.intersects(bbox)]
# #
# #             # output folder
# #             out_dir = out_root / f"tile_{ty:03d}_{tx:03d}"
# #             out_dir.mkdir(parents=True, exist_ok=True)
# #
# #             # img.tif (3-band)
# #             meta_img = base_meta.copy()
# #             meta_img.update(height=h, width=w, transform=tfm)
# #             with rasterio.open(out_dir / "img.tif", "w", **meta_img) as dst:
# #                 dst.write(img_tile)
# #
# #             # helper to write single band (CHM & COST)
# #             def write_single(arr: np.ndarray, fname: str):
# #                 meta = meta_img.copy()
# #                 meta.update(count=1, dtype="float32")
# #                 with rasterio.open(out_dir / fname, "w", **meta) as dst:
# #                     dst.write(arr.astype("float32"), 1)
# #
# #             write_single(chm_tile,  "chm.tif")
# #             write_single(cost_tile, "cost.tif")
# #
# #             # seeds.gpkg
# #             if not seeds_tile.empty:
# #                 seeds_tile.to_file(out_dir / "seeds.gpkg", driver="GPKG")
# #
# #             pbar.update(1)
# #
# #         pbar.close()
# #
# #         # close datasets
# #         chm_ds.close(); cost_ds.close()
# #         if chm_raw is not chm_ds:   chm_raw.close()
# #         if cost_raw is not cost_ds: cost_raw.close()
# #
# #     print(f"✓ Tiles saved under {out_root}")
# #
# #
# # if __name__ == "__main__":
# #     main()