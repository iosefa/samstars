#!/usr/bin/env python3
"""Run bundled samstars inference on an existing prepared tiles directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samstars import SamStars, segment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a flat samstars model bundle and run segmentation on an existing "
            "prepared tiles directory."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Flat samstars model bundle directory containing manifest.json.",
    )
    parser.add_argument(
        "--tiles-dir",
        required=True,
        type=Path,
        help=(
            "Prepared tiles directory. Each tile_* directory should already contain "
            "the expected inputs such as img.tif, chm.tif, cost.tif, and seeds.gpkg."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Run output directory.",
    )
    parser.add_argument(
        "--storage",
        default="persistent",
        choices=("persistent", "temp"),
        help="Intermediate storage policy.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="SAM runtime device, for example cpu, mps, or cuda.",
    )
    parser.add_argument(
        "--stardist-cpu",
        action="store_true",
        help="Force StarDist refinement onto CPU.",
    )
    parser.add_argument(
        "--stardist-input-key",
        choices=("index", "file_name", "logit"),
        default=None,
        help="Override the bundle default StarDist input key.",
    )
    parser.add_argument(
        "--stardist-prob-thr",
        type=float,
        default=None,
        help="Override StarDist probability threshold.",
    )
    parser.add_argument(
        "--stardist-nms-thr",
        type=float,
        default=None,
        help="Override StarDist NMS threshold.",
    )
    parser.add_argument(
        "--stardist-window-size-m",
        type=float,
        default=None,
        help="Internal StarDist refinement window size in map units.",
    )
    parser.add_argument(
        "--stardist-window-overlap-m",
        type=float,
        default=None,
        help="Internal StarDist refinement window overlap in map units.",
    )
    parser.add_argument(
        "--stardist-dedup-iou-thr",
        type=float,
        default=None,
        help="IoU threshold for final polygon deduplication.",
    )
    parser.add_argument(
        "--crop-smooth-sigma",
        type=float,
        default=None,
        help="Per-crop SAM logit blur sigma.",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=None,
        help="Global SAM probability mosaic blur sigma.",
    )
    parser.add_argument(
        "--post-smooth-sigma",
        type=float,
        default=None,
        help="Optional post-fusion index blur sigma.",
    )
    parser.add_argument(
        "--fast",
        dest="fast",
        action="store_true",
        default=None,
        help="Force the fast SAM proposal path.",
    )
    parser.add_argument(
        "--slow",
        dest="fast",
        action="store_false",
        help="Force the slow SAM proposal path.",
    )
    parser.add_argument(
        "--use-aux",
        dest="stardist_use_aux",
        action="store_true",
        default=None,
        help="Force CHM auxiliary input on for StarDist refinement.",
    )
    parser.add_argument(
        "--no-use-aux",
        dest="stardist_use_aux",
        action="store_false",
        help="Force CHM auxiliary input off for StarDist refinement.",
    )
    parser.add_argument(
        "--sam-only",
        action="store_true",
        help="Run SAM proposals only.",
    )
    parser.add_argument(
        "--stardist-only",
        action="store_true",
        help="Run StarDist refinement only. Proposal records must already exist under --out.",
    )
    parser.add_argument(
        "--no-skip-existing-sam",
        dest="skip_existing_sam",
        action="store_false",
        default=True,
        help="Recompute SAM outputs even if proposal rasters already exist.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.sam_only and args.stardist_only:
        raise SystemExit("Use only one of --sam-only or --stardist-only.")

    model = SamStars.load(args.model)
    result = segment(
        model=model,
        out=args.out,
        storage=args.storage,
        tiles_dir=args.tiles_dir,
        device=args.device,
        stardist_input_key=args.stardist_input_key,
        stardist_use_aux=args.stardist_use_aux,
        stardist_prob_thr=args.stardist_prob_thr,
        stardist_nms_thr=args.stardist_nms_thr,
        stardist_window_size_m=args.stardist_window_size_m,
        stardist_window_overlap_m=args.stardist_window_overlap_m,
        stardist_dedup_iou_thr=args.stardist_dedup_iou_thr,
        stardist_cpu=args.stardist_cpu,
        run_sam=not args.stardist_only,
        run_stardist=not args.sam_only,
        skip_existing_sam=args.skip_existing_sam,
        fast=args.fast,
        crop_smooth_sigma=args.crop_smooth_sigma,
        smooth_sigma=args.smooth_sigma,
        post_smooth_sigma=args.post_smooth_sigma,
    )
    print(result.crowns)


if __name__ == "__main__":
    main()
