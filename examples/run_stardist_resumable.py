#!/usr/bin/env python3
"""Run StarDist inference one record at a time so native crashes do not kill the whole run."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def guess_source_name(input_path: str | Path) -> str:
    stem = Path(str(input_path)).stem
    for suf in ("_sam_index", "_sam_prob", "_sam_logit", "_index", "_prob"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robust per-record StarDist runner")
    p.add_argument("--python", default="python", help="Python executable")
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--probs-json", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--model-name", default="stardist")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--input-key", default="index")
    p.add_argument("--scale-stats", type=Path, required=True)
    p.add_argument("--chm-key", default="chm")
    p.add_argument("--stardist-cpu", action="store_true")
    p.add_argument("--use-aux", action="store_true")
    p.add_argument("--skip-existing-labels", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    recs = json.load(open(args.probs_json))
    if not isinstance(recs, list):
        raise SystemExit(f"{args.probs_json} must be a list")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = args.out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    failed_log = args.out_dir / "stardist_failed_records.jsonl"

    done = 0
    skipped = 0
    failed = 0

    for i, rec in enumerate(recs, start=1):
        x_path = rec.get(args.input_key, "")
        if not x_path:
            failed += 1
            continue

        source = guess_source_name(x_path)
        out_lbl = labels_dir / f"{source}_labels.tif"
        if args.skip_existing_labels and out_lbl.exists():
            skipped += 1
            continue

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump([rec], tf)
            one_manifest = Path(tf.name)

        preds_name = f"preds_{source}.gpkg"
        cmd = [
            args.python,
            str(args.repo_root / "samstars/segmentation/refinement.py"),
            "--probs-json",
            str(one_manifest),
            "--model-dir",
            str(args.model_dir),
            "--model-name",
            args.model_name,
            "--out-dir",
            str(args.out_dir),
            "--input-key",
            args.input_key,
            "--scale-stats",
            str(args.scale_stats),
            "--preds-name",
            preds_name,
            "--write-labels",
            "--chm-key",
            args.chm_key,
        ]
        if args.use_aux:
            cmd.append("--use-aux")

        env = None
        if args.stardist_cpu:
            env = dict(**__import__("os").environ)
            env["CUDA_VISIBLE_DEVICES"] = ""
            # Keep native libs single-threaded to reduce random native crashes.
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
            env.setdefault("TF_NUM_INTEROP_THREADS", "1")

        proc = subprocess.run(cmd, env=env)
        try:
            one_manifest.unlink(missing_ok=True)
        except Exception:
            pass

        if proc.returncode == 0 and out_lbl.exists():
            done += 1
        else:
            failed += 1
            with open(failed_log, "a") as f:
                f.write(json.dumps({"source": source, "x_path": x_path, "returncode": proc.returncode}) + "\n")

        if i % 25 == 0:
            print(f"progress: {i}/{len(recs)}  new_done={done} skipped={skipped} failed={failed}")

    print(f"complete: total={len(recs)} new_done={done} skipped={skipped} failed={failed}")
    if failed:
        print(f"failed log: {failed_log}")


if __name__ == "__main__":
    main()
