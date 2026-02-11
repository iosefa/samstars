#!/usr/bin/env python3
"""
Summarize eval-time across multiple runs (e.g., warmup + repeats).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.summarize_eval_time import _load_fixed_val_stats, _load_seconds


def _expand_paths(arg: str | None) -> List[Path]:
    if not arg:
        return []
    out: List[Path] = []
    for token in [p.strip() for p in arg.split(",") if p.strip()]:
        if any(ch in token for ch in ["*", "?", "["]):
            out.extend(sorted(Path().glob(token)))
        else:
            out.append(Path(token))
    return out


def _parse_times(paths: Iterable[Path]) -> List[float]:
    times = []
    for path in paths:
        if not path.exists():
            print(f"[skip] missing log: {path}", file=sys.stderr)
            continue
        sec = _load_seconds(path)
        if sec is None:
            print(f"[skip] no timing parsed: {path}", file=sys.stderr)
            continue
        times.append(sec)
    return times


def _summarize(times: List[float]) -> dict:
    if not times:
        return {}
    return {
        "n_runs": len(times),
        "seconds_mean": statistics.mean(times),
        "seconds_median": statistics.median(times),
        "seconds_min": min(times),
        "seconds_max": max(times),
        "seconds_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
    }


def _add_normalized(row: dict, area_km2: float, n_crowns: int) -> None:
    if area_km2:
        row["sec_per_km2_mean"] = row["seconds_mean"] / area_km2
        row["sec_per_km2_median"] = row["seconds_median"] / area_km2
    else:
        row["sec_per_km2_mean"] = None
        row["sec_per_km2_median"] = None
    if n_crowns:
        row["sec_per_crown_mean"] = row["seconds_mean"] / n_crowns
        row["sec_per_crown_median"] = row["seconds_median"] / n_crowns
    else:
        row["sec_per_crown_mean"] = None
        row["sec_per_crown_median"] = None


def main() -> None:
    pa = argparse.ArgumentParser(description="Summarize eval-time across multiple runs.")
    pa.add_argument("--size", type=str, required=True)
    pa.add_argument("--fixed-val-manifest", type=Path, required=True)
    pa.add_argument("--detectron-logs", type=str, default=None, help="Comma list/glob")
    pa.add_argument("--sam-logs", type=str, default=None, help="Comma list/glob")
    pa.add_argument("--stardist-logs", type=str, default=None, help="Comma list/glob")
    pa.add_argument("--skip-first", type=int, default=0, help="Skip N earliest logs per list")
    pa.add_argument("--out", type=Path, required=True)
    args = pa.parse_args()

    stats = _load_fixed_val_stats(args.fixed_val_manifest)
    area_km2 = stats["area_km2"]
    n_crowns = stats["n_crowns"]
    n_chips = stats["n_chips"]

    rows = []

    def handle(method: str, stage: str, paths: List[Path]) -> None:
        paths = sorted(paths)
        if args.skip_first:
            paths = paths[args.skip_first :]
        times = _parse_times(paths)
        if not times:
            return
        row = {"size": args.size, "method": method, "stage": stage}
        row.update(_summarize(times))
        row["seconds_runs"] = json.dumps(times)
        _add_normalized(row, area_km2, n_crowns)
        row["area_km2"] = area_km2
        row["n_crowns"] = n_crowns
        row["n_chips"] = n_chips
        rows.append(row)

    handle("detectron2", "eval", _expand_paths(args.detectron_logs))
    handle("sam-star", "sam_eval", _expand_paths(args.sam_logs))
    handle("sam-star", "stardist_eval", _expand_paths(args.stardist_logs))

    # combined SAM-STAR if both present
    sam_rows = [r for r in rows if r["method"] == "sam-star"]
    sam_eval = next((r for r in sam_rows if r["stage"] == "sam_eval"), None)
    sd_eval = next((r for r in sam_rows if r["stage"] == "stardist_eval"), None)
    if sam_eval and sd_eval:
        sam_runs = json.loads(sam_eval.get("seconds_runs", "[]"))
        sd_runs = json.loads(sd_eval.get("seconds_runs", "[]"))
        n_combo = min(len(sam_runs), len(sd_runs))
        combo_runs = [sam_runs[i] + sd_runs[i] for i in range(n_combo)]
        combo = {
            "size": args.size,
            "method": "sam-star",
            "stage": "combined_eval",
            "n_runs": min(sam_eval["n_runs"], sd_eval["n_runs"]),
            "seconds_mean": sam_eval["seconds_mean"] + sd_eval["seconds_mean"],
            "seconds_median": sam_eval["seconds_median"] + sd_eval["seconds_median"],
            "seconds_min": sam_eval["seconds_min"] + sd_eval["seconds_min"],
            "seconds_max": sam_eval["seconds_max"] + sd_eval["seconds_max"],
            "seconds_std": (sam_eval["seconds_std"] ** 2 + sd_eval["seconds_std"] ** 2) ** 0.5,
        }
        combo["seconds_runs"] = json.dumps(combo_runs)
        _add_normalized(combo, area_km2, n_crowns)
        combo["area_km2"] = area_km2
        combo["n_crowns"] = n_crowns
        combo["n_chips"] = n_chips
        rows.append(combo)

    if not rows:
        raise SystemExit("No timings parsed. Provide log files with parseable timing lines.")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
