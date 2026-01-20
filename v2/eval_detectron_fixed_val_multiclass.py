#!/usr/bin/env python3
"""
Evaluate multiclass Detectron2 weights across multiple sizes against a fixed validation set.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_sizes_arg(sizes_arg: str | None, weights_root: Path) -> list[str]:
    if sizes_arg:
        return [s.strip() for s in sizes_arg.split(",") if s.strip()]
    sizes = []
    for p in sorted(weights_root.iterdir()):
        if not p.is_dir():
            continue
        model = p / "train_logs" / "maskrcnn_crowns_model_final.pth"
        if model.exists():
            sizes.append(p.name)
    return sizes


def main() -> None:
    pa = argparse.ArgumentParser(
        description="Run eval_detectron_on_chip_tifs_multiclass.py for multiple sizes using a fixed val manifest."
    )
    pa.add_argument("--val-manifest", type=Path, required=True)
    pa.add_argument("--weights-root", type=Path, default=Path("v2/detectron"))
    pa.add_argument("--out-root", type=Path, default=Path("v2/detectron_fixed_val"))
    pa.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes; default auto-detect from weights-root")
    pa.add_argument("--device", type=str, default="cpu")
    pa.add_argument("--score-thresh", type=float, default=0.5)
    pa.add_argument("--min-mask-px", type=int, default=50)
    pa.add_argument("--max-dets", type=int, default=100, help="Max detections per image at test time")
    pa.add_argument("--rpn-post-nms-topk", type=int, default=1000, help="RPN post-NMS topk at test time")
    pa.add_argument("--class-field", type=str, default="class")
    pa.add_argument("--class-mapping", type=Path, default=None, help="Optional class_mapping.json override")
    pa.add_argument("--chm-suffix", type=str, default="_chm.tif", help="Suffix for CHM chip tif files")
    pa.add_argument("--write-gpkg", action="store_true", help="Write GPKG predictions for each size")
    pa.add_argument("--write-per-chip", action="store_true", help="Write per-chip prediction GPKGs")
    pa.add_argument("--per-chip-dir", type=Path, default=None, help="Optional override for per-chip output dir")
    args = pa.parse_args()

    sizes = _parse_sizes_arg(args.sizes, args.weights_root)
    if not sizes:
        raise SystemExit("No sizes found. Provide --sizes or ensure weights exist under --weights-root.")

    args.out_root.mkdir(parents=True, exist_ok=True)

    for sz in sizes:
        weights = args.weights_root / sz / "train_logs" / "maskrcnn_crowns_model_final.pth"
        if not weights.exists():
            print(f"[skip] missing weights: {weights}")
            continue

        out_dir = args.out_root / sz
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_out = out_dir / "val_metrics.json"
        cmd = [
            sys.executable,
            "v2/eval_detectron_on_chip_tifs_multiclass.py",
            "--val-manifest",
            str(args.val_manifest),
            "--weights",
            str(weights),
            "--device",
            args.device,
            "--score-thresh",
            str(args.score_thresh),
            "--min-mask-px",
            str(args.min_mask_px),
            "--max-dets",
            str(args.max_dets),
            "--rpn-post-nms-topk",
            str(args.rpn_post_nms_topk),
            "--class-field",
            args.class_field,
            "--chm-suffix",
            args.chm_suffix,
            "--metrics-out",
            str(metrics_out),
        ]
        if args.class_mapping:
            cmd += ["--class-mapping", str(args.class_mapping)]
        if args.write_gpkg:
            cmd += ["--output-gpkg", str(out_dir / "val_preds.gpkg")]
        if args.write_per_chip:
            per_chip_dir = args.per_chip_dir if args.per_chip_dir else (out_dir / "per_chip_preds")
            cmd += ["--per-chip-dir", str(per_chip_dir)]

        print(f"[eval] size={sz} -> {metrics_out}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
