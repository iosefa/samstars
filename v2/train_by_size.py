#!/usr/bin/env python3
"""
Train Detectron and fused SAM-conditioned StarDist on cumulative chip subsets.
Deterministic chip-level holdout via split-seed; per-size outputs under an output root (default v2/models/sizes/<size>/).
"""
from __future__ import annotations
import argparse, json, random, subprocess, sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent
CHIPS_DIR = ROOT / "chips"
RECORDS_PATH = CHIPS_DIR / "detectron_records.json"
DEFAULT_MODELS_ROOT = ROOT / "models" / "sizes"

SAM_BASE_CKPT = ROOT.parent / "sam" / "sam_vit_l_0b3195.pth"
SAM_TRAIN_SCRIPT = ROOT.parent / "sam" / "train_sam_end_to_end.py"
DETECTRON_TRAIN = ROOT / "train_detectron_on_chips.py"
FUSED_TRAIN = ROOT / "train_logit_aware_stardist.py"


def run(cmd):
    print(" ", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def load_records():
    recs = json.load(open(RECORDS_PATH))
    if not recs:
        raise RuntimeError(f"No records in {RECORDS_PATH}")
    return recs


def ensure_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def prepare_subset_dirs(subset, size_dir: Path, holdout: float, split_seed: int):
    chip_dir_train = size_dir / "chips_train"
    chip_dir_val = size_dir / "chips_val"
    for base in (chip_dir_train, chip_dir_val):
        for sub in ("img", "msk", "json", "chm", "cost"):
            (base / sub).mkdir(parents=True, exist_ok=True)

    recs = subset[:]
    random.seed(split_seed)
    random.shuffle(recs)
    k = int(round(len(recs) * (1.0 - holdout)))
    train_recs = recs[:k]
    val_recs = recs[k:] if holdout > 0 else []

    def exists_for(name: str, sub: str):
        ext = "json" if sub == "json" else "npy"
        return (CHIPS_DIR / sub / f"{name}.{ext}").exists()

    def link_recs(recs_list, base_dir):
        filtered = []
        for rec in recs_list:
            name = Path(rec["file_name"]).name.replace(".npy", "")
            if not exists_for(name, "img") or not exists_for(name, "msk"):
                continue
            filtered.append(rec)
            for sub in ("img", "msk", "json", "chm", "cost"):
                src = CHIPS_DIR / sub / f"{name}.{ 'json' if sub=='json' else 'npy'}"
                if src.exists():
                    ensure_symlink(src, base_dir / sub / src.name)
        return filtered

    train_recs = link_recs(train_recs, chip_dir_train)
    val_recs = link_recs(val_recs, chip_dir_val)

    def adjust_paths(recs_list, base_dir):
        out = []
        for rec in recs_list:
            fname = Path(rec["file_name"]).name
            r = rec.copy()
            r["file_name"] = str(base_dir / "img" / fname)
            out.append(r)
        return out

    train_recs_adj = adjust_paths(train_recs, chip_dir_train)
    val_recs_adj = adjust_paths(val_recs, chip_dir_val)

    train_json = size_dir / "detectron_records_train.json"
    val_json = size_dir / "detectron_records_val.json"
    json.dump(train_recs_adj, open(train_json, "w"))
    json.dump(val_recs_adj, open(val_json, "w"))

    return chip_dir_train, chip_dir_val, train_json, val_json


def write_val_extent(val_json: Path, out_dir: Path):
    if not val_json.exists():
        return
    recs = json.load(open(val_json))
    polys = []
    crs = None
    for rec in recs:
        meta = rec.get("meta") or {}
        bounds = meta.get("bounds")
        if not bounds:
            continue
        polys.append(box(*bounds))
        if crs is None:
            crs = meta.get("crs")
    if not polys:
        return
    gdf = gpd.GeoDataFrame({"id": range(1, len(polys) + 1)}, geometry=polys, crs=crs)
    gdf.to_file(out_dir / "val_extent.gpkg", layer="val_extent", driver="GPKG")


def train_sam(size_dir: Path, chip_dir: Path):
    sam_dir = size_dir / "sam"
    sam_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = sam_dir / "sam_crowns_decoder.pth"
    cmd = [
        sys.executable, str(SAM_TRAIN_SCRIPT),
        "--chip-dir", str(chip_dir),
        "--ckpt", str(SAM_BASE_CKPT),
        "--out", str(ckpt_out),
        "--skip-chips",
        "--device", "cpu",
        "--epochs", "3",
    ]
    run(cmd)
    return ckpt_out


def train_detectron(size_dir: Path, train_json: Path, val_json: Path, chip_dir_train: Path, chip_dir_val: Path, split_seed: int):
    det_dir = size_dir / "detectron"
    det_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(DETECTRON_TRAIN),
        "--records", str(train_json),
        "--val-records", str(val_json),
        "--img-dir", str(chip_dir_train / "img"),
        "--val-img-dir", str(chip_dir_val / "img"),
        "--output-dir", str(det_dir / "train_logs"),
        "--holdout", str(0.0),
        "--metrics-out", str(det_dir / "train_logs" / "metrics.json"),
        "--split-seed", str(split_seed),
    ]
    run(cmd)


def train_fused(size_dir: Path, chip_dir_train: Path, train_json: Path, val_json: Path, sam_ckpt: Path, split_seed: int):
    sd_dir = size_dir / "stardist"
    sd_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(FUSED_TRAIN),
        "--chip-dir", str(chip_dir_train),
        "--records-json", str(train_json),
        "--val-records-json", str(val_json),
        "--holdout", str(0.0),
        "--split-seed", str(split_seed),
        "--sam-ckpt", str(sam_ckpt),
        "--out", str(sd_dir / "logit_stardist_head.pth"),
        "--metrics-out", str(sd_dir / "metrics.json"),
    ]
    run(cmd)


def main():
    pa = argparse.ArgumentParser(description="Train models on cumulative chip subsets (fused SAM-conditioned StarDist).")
    pa.add_argument("--step", type=int, default=10)
    pa.add_argument("--start-chips", type=int, default=10, help="Start size (chips) for first subset")
    pa.add_argument("--max-chips", type=int, default=None)
    pa.add_argument("--seed", type=int, default=42, help="Global shuffle seed")
    pa.add_argument("--split-seed", type=int, default=42, help="Seed for deterministic train/val split")
    pa.add_argument("--holdout", type=float, default=0.3, help="Holdout fraction for validation")
    pa.add_argument("--run-detectron", action="store_true", help="Train Detectron on subsets")
    pa.add_argument("--run-fused", action="store_true", help="Train SAM + fused StarDist on subsets")
    pa.add_argument("--output-root", type=Path, default=DEFAULT_MODELS_ROOT, help="Base directory for per-size outputs")
    args = pa.parse_args()

    if not (args.run_detectron or args.run_fused):
        args.run_detectron = args.run_fused = True

    records = load_records()
    random.seed(args.seed)
    random.shuffle(records)
    total = len(records)
    max_chips = args.max_chips or total
    sizes = [s for s in range(args.start_chips, max_chips + 1, args.step) if s <= max_chips]

    for n in sizes:
        subset = records[:n]
        size_dir = args.output_root / f"{n:04d}"
        size_dir.mkdir(parents=True, exist_ok=True)
        chip_dir_train, chip_dir_val, train_json, val_json = prepare_subset_dirs(subset, size_dir, args.holdout, args.split_seed)
        write_val_extent(val_json, size_dir)

        if args.run_detectron:
            train_detectron(size_dir, train_json, val_json, chip_dir_train, chip_dir_val, args.split_seed)
        if args.run_fused:
            sam_ckpt = train_sam(size_dir, chip_dir_train)
            train_fused(size_dir, chip_dir_train, train_json, val_json, sam_ckpt, args.split_seed)
        print(f"✓ completed size {n}")


if __name__ == "__main__":
    main()
