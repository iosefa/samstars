#!/usr/bin/env python3
"""
Create chip npys (and optional GeoTIFF/GPKG pairs) from precomputed manifests.
Uses recorded windows/metadata; no overlap/stride logic here.
For each size dir:
  - reads train_manifest.json (and val_manifest.json if present)
  - extracts image/chm/cost patches, pads to size if needed
  - writes npys under chips_{train,val}/{img,msk,chm,cost,json}
  - writes detectron-ready chips under chips_{train,val}/{tif,gpkg}
  - writes per-chip CHM GeoTIFFs under chips_{train,val}/tif as *_chm.tif
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box


def clip_crowns(crowns: gpd.GeoDataFrame, win_geom):
    sub = crowns[crowns.intersects(win_geom)].copy()
    if sub.empty:
        return sub
    sub["geometry"] = sub.geometry.buffer(0)
    sub["geometry"] = sub.geometry.intersection(win_geom)
    sub = sub[~sub.geometry.is_empty]
    return sub


def build_chips(
    manifest: Path,
    out_dir: Path,
    size: int = 512,
    bands=(7, 5, 3),
    write_tif: bool = True,
    write_gpkg: bool = True,
):
    recs = json.load(open(manifest))
    for sub in ("img", "msk", "chm", "cost", "json"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    if write_tif:
        (out_dir / "tif").mkdir(parents=True, exist_ok=True)
    if write_gpkg:
        (out_dir / "gpkg").mkdir(parents=True, exist_ok=True)

    crowns_cache = {}
    detectron_pairs = []
    for rec in recs:
        img_path = Path(rec["img_path"])
        crowns_path = Path(rec["crowns_path"])
        chm_path = img_path.parent / "chm.tif"
        cost_path = img_path.parent / "cost.tif"
        x0, y0, x1, y1 = rec["window"]
        win = Window(x0, y0, x1 - x0, y1 - y0)
        with rasterio.open(img_path) as src:
            available = src.count
            sel_bands = [b for b in bands if b <= available] or [1]
            img_win = src.read(sel_bands, window=win, boundless=True, fill_value=0)
            tfm = rasterio.windows.transform(win, src.transform)
            crs = src.crs
            if write_tif:
                tif_win = src.read(
                    list(range(1, src.count + 1)),
                    window=win,
                    boundless=True,
                    fill_value=0,
                )
                tif_profile = src.profile.copy()
        with rasterio.open(chm_path) as csrc:
            chm_win = csrc.read(1, window=win, boundless=True, fill_value=0)
        with rasterio.open(cost_path) as cosrc:
            cost_win = cosrc.read(1, window=win, boundless=True, fill_value=0)

        h, w = img_win.shape[1], img_win.shape[2]
        pad_h, pad_w = size - h, size - w
        if pad_h < 0 or pad_w < 0:
            img_win = img_win[:, : size, : size]
            chm_win = chm_win[: size, : size]
            cost_win = cost_win[: size, : size]
            pad_h = pad_w = 0
        if pad_h > 0 or pad_w > 0:
            img_win = np.pad(img_win, ((0, 0), (0, pad_h), (0, pad_w)))
            chm_win = np.pad(chm_win, ((0, pad_h), (0, pad_w)))
            cost_win = np.pad(cost_win, ((0, pad_h), (0, pad_w)))
            if write_tif:
                tif_win = np.pad(tif_win, ((0, 0), (0, pad_h), (0, pad_w)))

        if crowns_path not in crowns_cache:
            crowns = gpd.read_file(crowns_path)
            if crowns.crs is None:
                crowns = crowns.set_crs(crs)
            elif crs and crowns.crs != crs:
                crowns = crowns.to_crs(crs)
            crowns_cache[crowns_path] = crowns
        crowns = crowns_cache[crowns_path]
        win_geom = box(*rasterio.transform.array_bounds(h, w, tfm))
        clipped = clip_crowns(crowns, win_geom)
        # Instance-labeled mask: each crown gets a unique ID (1..N)
        shapes = [(geom, i + 1) for i, geom in enumerate(clipped.geometry)]
        mask = rasterize(
            shapes,
            out_shape=(h, w),
            transform=tfm,
            fill=0,
            all_touched=True,
            dtype="uint16",
        )
        if pad_h > 0 or pad_w > 0:
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)))

        name = f"{rec['tile']}_{rec['chip_id']}"
        np.save(out_dir / "img" / f"{name}.npy", img_win.astype(np.float32))
        np.save(out_dir / "msk" / f"{name}.npy", mask.astype(np.uint16))
        np.save(out_dir / "chm" / f"{name}.npy", chm_win.astype(np.float32))
        np.save(out_dir / "cost" / f"{name}.npy", cost_win.astype(np.float32))
        # simple point prompt: centroid of mask if non-empty
        ys, xs = np.where(mask)
        pt = [int(xs.mean()), int(ys.mean())] if xs.size else [size // 2, size // 2]
        json.dump({"point": pt}, open(out_dir / "json" / f"{name}.json", "w"))

        tif_out = None
        if write_tif:
            tif_profile.update(
                height=size,
                width=size,
                transform=tfm,
                count=tif_win.shape[0],
            )
            tif_out = out_dir / "tif" / f"{name}.tif"
            with rasterio.open(tif_out, "w", **tif_profile) as dst:
                dst.write(tif_win)
            chm_profile = tif_profile.copy()
            chm_profile.update(count=1, dtype="float32")
            chm_out = out_dir / "tif" / f"{name}_chm.tif"
            with rasterio.open(chm_out, "w", **chm_profile) as dst:
                dst.write(chm_win.astype(np.float32), 1)

        gpkg_out = None
        if write_gpkg:
            if clipped.empty:
                clipped = crowns.iloc[:0].copy()
                clipped = clipped.set_geometry("geometry")
                clipped = clipped.set_crs(crs)
            gpkg_out = out_dir / "gpkg" / f"{name}.gpkg"
            clipped.to_file(gpkg_out, driver="GPKG", layer="crowns")

        if tif_out and gpkg_out:
            detectron_pairs.append({
                "name": name,
                "tif_path": str(tif_out),
                "gpkg_path": str(gpkg_out),
                "img_path": str(img_path),
                "crowns_path": str(crowns_path),
                "window": [int(x0), int(y0), int(x1), int(y1)],
                "transform": list(tfm)[:6],
                "bounds": [float(b) for b in win_geom.bounds],
                "crs": crs.to_wkt() if crs else None,
            })

    if detectron_pairs:
        json.dump(detectron_pairs, open(out_dir / "detectron_manifest.json", "w"))


def main():
    pa = argparse.ArgumentParser(description="Build chip npys from train_manifest.json.")
    pa.add_argument("--manifest", type=Path, help="train_manifest.json (single size)")
    pa.add_argument("--splits-root", type=Path, help="Root containing per-size folders with train_manifest.json")
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes to process (with --splits-root)")
    pa.add_argument("--out-dir", type=Path, required=True, help="Output root; per-size chips under out-dir/<size>/chips_train")
    pa.add_argument("--size", type=int, default=512)
    pa.add_argument("--no-tif", action="store_true", help="Skip writing GeoTIFF chips")
    pa.add_argument("--no-gpkg", action="store_true", help="Skip writing GeoPackage crowns per chip")
    pa.add_argument("--fixed-val-manifest", type=Path, default=None, help="Single fixed val manifest")
    pa.add_argument("--fixed-val-out", type=Path, default=None, help="Output dir for fixed val chips")
    args = pa.parse_args()
    if args.fixed_val_manifest:
        out_dir = args.fixed_val_out or (args.out_dir / "fixed_val" / "chips_val")
        build_chips(
            args.fixed_val_manifest,
            out_dir,
            args.size,
            write_tif=not args.no_tif,
            write_gpkg=not args.no_gpkg,
        )
        print(f"✓ built fixed val chips to {out_dir}")
    if args.splits_root:
        if args.sizes:
            size_set = {s.strip() for s in args.sizes.split(",") if s.strip()}
            size_dirs = [args.splits_root / s for s in sorted(size_set)]
        else:
            size_dirs = sorted(p for p in args.splits_root.iterdir() if p.is_dir())
        for size_dir in size_dirs:
            train_manifest = size_dir / "train_manifest.json"
            val_manifest = size_dir / "val_manifest.json"
            if train_manifest.exists():
                out_dir = args.out_dir / size_dir.name / "chips_train"
                build_chips(
                    train_manifest,
                    out_dir,
                    args.size,
                    write_tif=not args.no_tif,
                    write_gpkg=not args.no_gpkg,
                )
                print(f"✓ built train chips to {out_dir}")
            if val_manifest.exists():
                out_dir = args.out_dir / size_dir.name / "chips_val"
                build_chips(
                    val_manifest,
                    out_dir,
                    args.size,
                    write_tif=not args.no_tif,
                    write_gpkg=not args.no_gpkg,
                )
                print(f"✓ built val chips to {out_dir}")
    elif args.manifest:
        build_chips(
            args.manifest,
            args.out_dir,
            args.size,
            write_tif=not args.no_tif,
            write_gpkg=not args.no_gpkg,
        )
        print(f"✓ built chips to {args.out_dir}")
    else:
        raise SystemExit("Provide either --manifest or --splits-root")


if __name__ == "__main__":
    main()
