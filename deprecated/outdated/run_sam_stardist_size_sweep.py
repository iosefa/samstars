#!/usr/bin/env python3
"""
End-to-end SAM->StarDist pipeline per size with a fixed validation set.
For each size:
  1) build fixed-val manifests (once)
  2) train SAM decoder
  3) predict SAM logits/probs for train + fixed val
  4) build prob manifests
  5) train StarDist
  6) eval StarDist (fixed val)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd):
    """Internal helper function."""
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(description="Run SAM+StarDist per size with fixed val set.")
    pa.add_argument("--sizes", type=str, required=True, help="Comma-separated sizes, e.g. 0001,0003,0005")
    pa.add_argument("--splits-root", type=Path, required=True)
    pa.add_argument("--manifests-root", type=Path, required=True)
    pa.add_argument("--fixed-val-size", type=str, required=True)
    pa.add_argument("--chips-root", type=Path, required=True)
    pa.add_argument("--sam-ckpt", type=Path, default=None)
    pa.add_argument("--sam-probs-root", type=Path, required=True)
    pa.add_argument("--stardist-root", type=Path, default=None)
    pa.add_argument("--stardist-eval-root", type=Path, default=None)
    pa.add_argument("--sam-epochs", type=int, default=15)
    pa.add_argument("--sam-min-iters", type=int, default=0, help="Minimum SAM decoder iterations")
    pa.add_argument("--sam-scale-stats", type=Path, default=None, help="Global scaling stats JSON")
    pa.add_argument("--stardist-epochs", type=int, default=50)
    pa.add_argument("--stardist-min-iters", type=int, default=0, help="Minimum StarDist iterations")
    pa.add_argument("--stardist-use-aux", action="store_true", help="Use CHM + COST aux channels for StarDist")
    pa.add_argument("--stardist-scale-stats", type=Path, default=None, help="Global scaling stats JSON for StarDist")
    pa.add_argument("--device", type=str, default="cpu")
    pa.add_argument("--skip-eval", action="store_true", help="Skip StarDist eval step")
    pa.add_argument(
        "--stage",
        type=str,
        default="both",
        choices=["sam", "stardist", "both"],
        help="Run SAM only, StarDist only, or both",
    )
    pa.add_argument(
        "--sam-splits",
        type=str,
        default="train,val",
        help="Comma-separated SAM splits to run (train,val). Use train to skip val.",
    )
    pa.add_argument("--prob-thr", type=float, default=None)
    pa.add_argument("--nms-thr", type=float, default=None)
    args = pa.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    sizes_csv = ",".join(sizes)
    stardist_scale_stats = args.stardist_scale_stats or args.sam_scale_stats

    # 1) Build fixed-val manifests once (needed for SAM)
    if args.stage in ("sam", "both"):
        if args.sam_ckpt is None:
            raise SystemExit("--sam-ckpt is required for stage=sam")
        if args.stardist_root is None or args.stardist_eval_root is None:
            # Not used in SAM stage, but keep paths consistent for later stages.
            pass
        _run([
            sys.executable, "v2/build_fixed_val_manifests.py",
            "--splits-root", str(args.splits_root),
            "--out-root", str(args.manifests_root),
            "--fixed-val-size", args.fixed_val_size,
            "--sizes", sizes_csv,
        ])

    # 2-6) Per-size pipeline
    for sz in sizes:
        if args.stage in ("sam", "both"):
            cmd = [
                sys.executable, "v2/train_sam_decoder_from_chips.py",
                "--chips-root", str(args.chips_root),
                "--ckpt", str(args.sam_ckpt),
                "--epochs", str(args.sam_epochs),
                "--min-iters", str(args.sam_min_iters),
                "--device", args.device,
                "--sizes", sz,
            ]
            if args.sam_scale_stats:
                cmd += ["--scale-stats", str(args.sam_scale_stats)]
            _run(cmd)

            cmd = [
                sys.executable, "v2/predict_sam_from_manifests.py",
                "--manifests-root", str(args.manifests_root),
                "--ckpt-root", str(args.chips_root),
                "--out-root", str(args.sam_probs_root),
                "--device", args.device,
                "--sizes", sz,
                "--splits", args.sam_splits,
            ]
            if args.sam_scale_stats:
                cmd += ["--scale-stats", str(args.sam_scale_stats)]
            _run(cmd)

        if args.stage in ("stardist", "both"):
            if args.stardist_root is None or args.stardist_eval_root is None:
                raise SystemExit("--stardist-root and --stardist-eval-root are required for stage=stardist")
            _run([
                sys.executable, "v2/build_prob_manifests.py",
                "--probs-root", str(args.sam_probs_root),
                "--chips-root", str(args.chips_root),
                "--fixed-val-size", args.fixed_val_size,
                "--sizes", sz,
            ])

            stardist_cmd = [
                sys.executable, "v2/train_stardist_on_probs.py",
                "--train-probs", str(args.sam_probs_root / sz / "train_probs.json"),
                "--val-probs", str(args.sam_probs_root / sz / "val_probs.json"),
                "--out-dir", str(args.stardist_root / sz),
                "--epochs", str(args.stardist_epochs),
                "--min-iters", str(args.stardist_min_iters),
            ]
            if args.stardist_use_aux:
                stardist_cmd += ["--use-aux"]
            if stardist_scale_stats:
                stardist_cmd += ["--scale-stats", str(stardist_scale_stats)]
            _run(stardist_cmd)

            if not args.skip_eval:
                eval_cmd = [
                    sys.executable, "v2/eval_stardist_on_probs.py",
                    "--val-probs", str(args.sam_probs_root / sz / "val_probs.json"),
                    "--model-dir", str(args.stardist_root / sz),
                    "--out-dir", str(args.stardist_eval_root / sz),
                ]
                if args.stardist_use_aux:
                    eval_cmd += ["--use-aux"]
                if stardist_scale_stats:
                    eval_cmd += ["--scale-stats", str(stardist_scale_stats)]
                if (args.prob_thr is None) ^ (args.nms_thr is None):
                    raise SystemExit("Provide both --prob-thr and --nms-thr, or neither.")
                if args.prob_thr is not None and args.nms_thr is not None:
                    eval_cmd += ["--prob-thr", str(args.prob_thr), "--nms-thr", str(args.nms_thr)]
                _run(eval_cmd)


if __name__ == "__main__":
    main()
