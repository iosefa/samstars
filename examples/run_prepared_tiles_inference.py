#!/usr/bin/env python3
"""Train or run bundled samstars workflows from prepared tiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samstars import SamStars, segment, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a flat samstars model bundle from prepared labeled tiles or run "
            "segmentation on prepared tiles with an existing bundle."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--model",
        type=Path,
        help="Existing flat samstars model bundle directory containing manifest.json.",
    )
    mode.add_argument(
        "--sam-base-ckpt",
        type=Path,
        help="Base SAM checkpoint. When provided, the script trains a new model bundle.",
    )

    parser.add_argument(
        "--tiles-dir",
        required=True,
        type=Path,
        help=(
            "Tiles directory. In training mode, each tile may be either raw labeled "
            "(img.tif, chm.tif, den.tif, crowns.gpkg) or fully prepared labeled "
            "(img.tif, chm.tif, cost.tif, seeds.gpkg, crowns.gpkg). In segmentation "
            "mode, tiles must already contain the prepared inference inputs such as "
            "img.tif, chm.tif, cost.tif, and seeds.gpkg."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Model bundle directory for training mode, or run output directory for segmentation mode.",
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
        "--chip-size",
        type=int,
        default=512,
        help="SAM training chip size in pixels for training mode.",
    )
    parser.add_argument(
        "--chip-stride",
        type=int,
        default=384,
        help="SAM training chip stride in pixels for training mode.",
    )
    parser.add_argument(
        "--sam-epochs",
        type=int,
        default=15,
        help="SAM training epochs in training mode.",
    )
    parser.add_argument(
        "--stardist-epochs",
        type=int,
        default=50,
        help="StarDist training epochs in training mode.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction for StarDist training mode.",
    )

    parser.add_argument(
        "--stardist-cpu",
        action="store_true",
        help="Force StarDist refinement onto CPU in segmentation mode.",
    )
    parser.add_argument(
        "--stardist-input-key",
        choices=("index", "file_name", "logit"),
        default=None,
        help="StarDist primary input key. In segmentation mode, omit this to use the bundle default.",
    )
    parser.add_argument(
        "--stardist-prob-thr",
        type=float,
        default=None,
        help="Override StarDist probability threshold in segmentation mode.",
    )
    parser.add_argument(
        "--stardist-nms-thr",
        type=float,
        default=None,
        help="Override StarDist NMS threshold in segmentation mode.",
    )
    parser.add_argument(
        "--stardist-window-size-m",
        type=float,
        default=None,
        help="Internal StarDist window size in map units.",
    )
    parser.add_argument(
        "--stardist-window-overlap-m",
        type=float,
        default=None,
        help="Internal StarDist window overlap in map units.",
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
        help="Force CHM auxiliary input on.",
    )
    parser.add_argument(
        "--no-use-aux",
        dest="stardist_use_aux",
        action="store_false",
        help="Force CHM auxiliary input off.",
    )
    parser.add_argument(
        "--sam-only",
        action="store_true",
        help="Segmentation mode only: run SAM proposals only.",
    )
    parser.add_argument(
        "--stardist-only",
        action="store_true",
        help="Segmentation mode only: run StarDist refinement only. Proposal records must already exist under --out.",
    )
    parser.add_argument(
        "--no-skip-existing-sam",
        dest="skip_existing_sam",
        action="store_false",
        default=True,
        help="Segmentation mode only: recompute SAM outputs even if proposal rasters already exist.",
    )
    return parser


def _default_if_none(value, fallback):
    return fallback if value is None else value


def main() -> None:
    args = build_parser().parse_args()

    if args.model is not None:
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
        return

    if args.sam_only or args.stardist_only or not args.skip_existing_sam:
        raise SystemExit(
            "--sam-only, --stardist-only, and --no-skip-existing-sam apply only in segmentation mode."
        )
    if args.stardist_cpu:
        raise SystemExit("--stardist-cpu applies only in segmentation mode.")
    if args.stardist_prob_thr is not None or args.stardist_nms_thr is not None:
        raise SystemExit("--stardist-prob-thr and --stardist-nms-thr apply only in segmentation mode.")

    model = train(
        tiles_dir=args.tiles_dir,
        sam_base_ckpt=args.sam_base_ckpt,
        out=args.out,
        storage=args.storage,
        device=args.device,
        chip_size=args.chip_size,
        chip_stride=args.chip_stride,
        sam_epochs=args.sam_epochs,
        stardist_epochs=args.stardist_epochs,
        val_fraction=args.val_fraction,
        crop_smooth_sigma=_default_if_none(args.crop_smooth_sigma, 2.0),
        smooth_sigma=_default_if_none(args.smooth_sigma, 4.0),
        post_smooth_sigma=_default_if_none(args.post_smooth_sigma, 0.0),
        stardist_input_key=_default_if_none(args.stardist_input_key, "index"),
        stardist_use_aux=_default_if_none(args.stardist_use_aux, True),
        stardist_window_size_m=args.stardist_window_size_m,
        stardist_window_overlap_m=_default_if_none(args.stardist_window_overlap_m, 0.0),
        stardist_dedup_iou_thr=_default_if_none(args.stardist_dedup_iou_thr, 0.0),
        fast=_default_if_none(args.fast, False),
    )
    print(model.root)


if __name__ == "__main__":
    main()
