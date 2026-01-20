#!/usr/bin/env python3
"""
Run simple non-parametric bin tests on per-chip metrics (e.g., AP50) by bin.

Inputs are per-chip CSVs produced by summarize_ap50_by_height_density.py, with
columns like: model, size, chip, bin, ap50, n_gt, n_pred.

Outputs:
  - bin effects per method (Mann-Whitney U or paired Wilcoxon + Holm correction)
  - pairwise bin comparisons per method (same as bin effects; kept for plotting)
  - method comparisons within each bin (paired Wilcoxon + Holm correction)
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _holm_adjust(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m, dtype=float)
    for i, idx in enumerate(order):
        adj[idx] = min((m - i) * pvals[idx], 1.0)
    # enforce monotonicity
    for i in range(1, m):
        if adj[order[i]] < adj[order[i - 1]]:
            adj[order[i]] = adj[order[i - 1]]
    return adj.tolist()


def _eps_sq(H: float, k: int, n: int) -> float:
    if n <= k:
        return float("nan")
    return max((H - k + 1) / (n - k), 0.0)


def _cliffs_delta_from_arrays(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    diff = x[:, None] - y[None, :]
    gt = np.sum(diff > 0)
    lt = np.sum(diff < 0)
    return float((gt - lt) / (x.size * y.size))


def _rank_biserial_from_w(w: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return 1.0 - (2.0 * w) / (n * (n + 1))


def main() -> None:
    pa = argparse.ArgumentParser(description="Non-parametric tests across bins for per-chip metrics.")
    pa.add_argument("--csvs", required=True, help="Comma-separated per-chip CSVs to analyze.")
    pa.add_argument("--metric", default="ap50", help="Metric column to test (default: ap50).")
    pa.add_argument("--size", type=int, default=None, help="Filter to a specific size value.")
    pa.add_argument("--model-col", default="model", help="Model column (default: model).")
    pa.add_argument("--bin-col", default="bin", help="Bin column (default: bin).")
    pa.add_argument("--chip-col", default="chip", help="Chip ID column (default: chip).")
    pa.add_argument("--min-n", type=int, default=2, help="Minimum samples per bin (default: 2).")
    pa.add_argument(
        "--method-pairs",
        default=None,
        help="Optional method pairs 'a:b,c:d' for within-bin comparisons; otherwise all pairs.",
    )
    pa.add_argument("--out-prefix", default=None, help="Write CSV outputs with this prefix.")
    args = pa.parse_args()

    paths = [Path(p.strip()) for p in args.csvs.split(",") if p.strip()]
    dfs = []
    for p in paths:
        if not p.exists():
            raise SystemExit(f"Missing CSV: {p}")
        df = pd.read_csv(p)
        df["_source"] = p.name
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    if args.size is not None and "size" in df.columns:
        df = df[df["size"] == args.size]

    metric = args.metric
    if metric not in df.columns:
        raise SystemExit(f"Metric column '{metric}' not found in CSVs.")

    df = df[[args.model_col, args.bin_col, args.chip_col, metric]].dropna()

    # Bin effects per method (Mann-Whitney U or paired Wilcoxon)
    bin_effect_rows = []
    bin_pair_rows = []
    for method, mdf in df.groupby(args.model_col):
        # collect bins with enough samples
        bin_groups = []
        bin_names = []
        for bin_name, bdf in mdf.groupby(args.bin_col):
            vals = bdf[metric].astype(float).to_numpy()
            if len(vals) >= args.min_n:
                bin_groups.append(vals)
                bin_names.append(bin_name)

        if len(bin_groups) >= 2:
            # pairwise bin comparisons (Mann-Whitney U or paired Wilcoxon)
            pair_rows = []
            pvals = []
            for (b1, g1), (b2, g2) in itertools.combinations(zip(bin_names, bin_groups), 2):
                # paired if the same chips appear in both bins
                s1 = mdf[mdf[args.bin_col] == b1].set_index(args.chip_col)[metric]
                s2 = mdf[mdf[args.bin_col] == b2].set_index(args.chip_col)[metric]
                common = s1.index.intersection(s2.index)
                if len(common) >= args.min_n:
                    x = s1.loc[common].astype(float).to_numpy()
                    y = s2.loc[common].astype(float).to_numpy()
                    try:
                        stat, p_u = stats.wilcoxon(x, y)
                        test = "wilcoxon_paired"
                    except ValueError:
                        # fall back to unpaired test
                        stat, p_u = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                        test = "mannwhitneyu"
                else:
                    x = g1
                    y = g2
                    stat, p_u = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                    test = "mannwhitneyu"

                pair_rows.append(
                    {
                        "model": method,
                        "bin_a": b1,
                        "bin_b": b2,
                        "n_a": int(len(g1)),
                        "n_b": int(len(g2)),
                        "n_common": int(len(common)) if len(common) else 0,
                        "test": test,
                        "stat": float(stat),
                        "p": float(p_u),
                        "cliffs_delta": _cliffs_delta_from_arrays(np.asarray(x), np.asarray(y)),
                    }
                )
                pvals.append(float(p_u))
            adj = _holm_adjust(pvals)
            for row, p_adj in zip(pair_rows, adj):
                row["p_adj_holm"] = p_adj
            bin_pair_rows.extend(pair_rows)
            bin_effect_rows.extend(pair_rows)

    # Method comparisons within each bin (paired Wilcoxon)
    method_pairs = None
    if args.method_pairs:
        method_pairs = []
        for pair in args.method_pairs.split(","):
            if ":" not in pair:
                raise SystemExit("Invalid --method-pairs format; expected 'a:b,c:d'.")
            a, b = pair.split(":")
            method_pairs.append((a.strip(), b.strip()))

    method_rows = []
    for bin_name, bdf in df.groupby(args.bin_col):
        methods = sorted(bdf[args.model_col].unique())
        pairs = method_pairs or list(itertools.combinations(methods, 2))
        pvals = []
        tmp_rows = []
        for m1, m2 in pairs:
            s1 = bdf[bdf[args.model_col] == m1].set_index(args.chip_col)[metric]
            s2 = bdf[bdf[args.model_col] == m2].set_index(args.chip_col)[metric]
            common = s1.index.intersection(s2.index)
            if len(common) < args.min_n:
                continue
            x = s1.loc[common].astype(float).to_numpy()
            y = s2.loc[common].astype(float).to_numpy()
            try:
                w, p = stats.wilcoxon(x, y)
            except ValueError:
                continue
            diff = x - y
            tmp_rows.append(
                {
                    "bin": bin_name,
                    "model_a": m1,
                    "model_b": m2,
                    "n": len(common),
                    "W": float(w),
                    "p": float(p),
                    "test": "wilcoxon_paired",
                    "median_diff": float(np.median(diff)),
                    "rank_biserial": _rank_biserial_from_w(float(w), len(common)),
                }
            )
            pvals.append(float(p))
        adj = _holm_adjust(pvals)
        for row, p_adj in zip(tmp_rows, adj):
            row["p_adj_holm"] = p_adj
        method_rows.extend(tmp_rows)

    # Output
    bin_effect_df = pd.DataFrame(bin_effect_rows)
    bin_pair_df = pd.DataFrame(bin_pair_rows)
    method_df = pd.DataFrame(method_rows)

    if args.out_prefix:
        out_prefix = Path(args.out_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        bin_effect_df.to_csv(f"{out_prefix}_bin_effects.csv", index=False)
        bin_pair_df.to_csv(f"{out_prefix}_bin_pairwise.csv", index=False)
        method_df.to_csv(f"{out_prefix}_method_within_bin.csv", index=False)
        print(f"wrote {out_prefix}_bin_effects.csv")
        print(f"wrote {out_prefix}_bin_pairwise.csv")
        print(f"wrote {out_prefix}_method_within_bin.csv")
    else:
        print("== Bin effects (MWU or paired Wilcoxon) ==")
        print(bin_effect_df.to_string(index=False))
        print("\n== Pairwise bin comparisons (Mann-Whitney U + Holm) ==")
        print(bin_pair_df.to_string(index=False))
        print("\n== Method comparisons within bins (paired Wilcoxon) ==")
        print(method_df.to_string(index=False))


if __name__ == "__main__":
    main()
