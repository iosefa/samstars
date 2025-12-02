#!/usr/bin/env python3
"""
Create/copy SHAP supplementary figures for the manuscript.

Outputs (written to manuscript/figures/output/):
- supp_shap_global_top20.png           (global top-20 feature importance bar)
- supp_shap_band_global.png            (band-level rollup bar)
- supp_shap_modality_global.png        (modality-level rollup bar)
- supp_shap_beeswarm_class_*.png       (per-class beeswarm plots)

Logic:
- If corresponding source PNGs already exist in the repository root (e.g.,
  shap_beeswarm_class_0.png), copy them into the output folder with the
  `supp_` prefix.
- Otherwise, if SHAP artifacts exist (shap_values_sampled.npz,
  shap_features_sampled.csv, shap_feature_names.json), generate the beeswarm
  plots directly using SHAP.

Run from repo root:
  python3 manuscript/figures/scripts/make_supp_shap_plots.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT  = ROOT / "manuscript" / "figures" / "output"


def _set_pub_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "font.size": 12,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
    })

# Source files in repo root (if present, they will be copied)
SRC_FILES = {
    "shap_global_bar_top20.png": "supp_shap_global_top20.png",
    "shap_band_global.png":      "supp_shap_band_global.png",
    "shap_modality_global.png":  "supp_shap_modality_global.png",
}

# Per-class beeswarm files
CLASS_IDS = list(range(6))  # METPOL..OTHER (update if class count changes)
BSWARM_SRC = [f"shap_beeswarm_class_{i}.png" for i in CLASS_IDS]


def copy_existing() -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    copied = []
    # Global bars
    for src, dst in SRC_FILES.items():
        s = ROOT / src
        if s.exists():
            shutil.copy2(s, OUT / dst)
            copied.append(dst)
    # Beeswarms
    for fn in BSWARM_SRC:
        s = ROOT / fn
        if s.exists():
            dst = OUT / ("supp_" + fn)
            shutil.copy2(s, dst)
            copied.append(dst.name)
    return copied


def _manual_beeswarm(ax, shap_vals: np.ndarray, X: np.ndarray, feat_names: list[str], topn=20, order_override: list[int] | None = None):
    """Render a simple beeswarm-like plot without shap library.
    shap_vals: (Ns, F)
    X: (Ns, F)
    """
    Ns, F = shap_vals.shape
    if order_override is not None:
        order = np.array(order_override, dtype=int)
    else:
        mean_abs = np.abs(shap_vals).mean(axis=0)
        order = np.argsort(mean_abs)[-topn:][::-1]
    yticklabels = [feat_names[i] if i < len(feat_names) else f"f{i}" for i in order]
    y_positions = np.arange(len(order))
    # Normalize feature values for color
    Xsel = X[:, order]
    # robust normalization per feature
    Xnorm = np.zeros_like(Xsel, dtype=float)
    for j in range(Xsel.shape[1]):
        col = Xsel[:, j]
        cmin, cmax = np.nanpercentile(col, [2, 98])
        if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
            cmin, cmax = float(np.nanmin(col)), float(np.nanmax(col))
        Xnorm[:, j] = (col - cmin) / max(cmax - cmin, 1e-6)
    # Plot each feature's points with small jitter in y
    from matplotlib import cm as _cm
    from matplotlib import colors as _colors
    cmap = _cm.get_cmap('coolwarm')
    norm = _colors.Normalize(vmin=0.0, vmax=1.0)
    for yi, fi in enumerate(order):
        y = np.full(Ns, yi, dtype=float)
        y += (np.random.default_rng(0).standard_normal(Ns) * 0.06)
        ax.scatter(shap_vals[:, fi], y, s=8, c=cmap(norm(Xnorm[:, yi])), alpha=0.65, linewidths=0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(yticklabels, fontsize=13)
    ax.axvline(0, color='#888', lw=0.8)
    ax.set_xlabel('SHAP value')
    ax.invert_yaxis()


def try_generate_beeswarms() -> list[str]:
    """Generate per-class beeswarm plots if artifacts exist.
    Requires: shap_values_sampled.npz (S: (Ns, F, C)),
              shap_features_sampled.csv (Ns x F),
              shap_feature_names.json (pretty names).
    """
    out = []
    # Prefer ROOT artifacts; fallback to manuscript/results
    npz = ROOT / "shap_values_sampled.npz"
    csv = ROOT / "shap_features_sampled.csv"
    jfe = ROOT / "shap_feature_names.json"
    if not npz.exists():
        npz = ROOT / "manuscript" / "results" / "shap_values_sampled.npz"
    if not csv.exists():
        csv = ROOT / "manuscript" / "results" / "shap_features_sampled.csv"
    if not jfe.exists():
        jfe = ROOT / "manuscript" / "results" / "shap_feature_names.json"
    if not (npz.exists() and csv.exists() and jfe.exists()):
        return out

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap = None
    try:
        import shap as _shap  # type: ignore
        shap = _shap
    except Exception:
        shap = None

    data = np.load(npz)
    S = data["S"]  # (Ns, F, C)
    X = pd.read_csv(csv)
    with open(jfe, "r") as f:
        j = json.load(f)
    pretty = j.get("pretty") or j.get("raw")
    if pretty is None:
        pretty = list(X.columns)

    OUT.mkdir(parents=True, exist_ok=True)
    C = S.shape[-1]
    for ci in range(C):
        try:
            _set_pub_style()
            fig = plt.figure(figsize=(7.5, 5.5))
            if shap is not None:
                shap.summary_plot(S[:, :, ci], X.values, feature_names=pretty, show=False, max_display=20)
            else:
                ax = plt.gca()
                _manual_beeswarm(ax, S[:, :, ci], X.values, pretty, topn=20)
            fig.tight_layout()
            out_path = OUT / f"supp_shap_beeswarm_class_{ci}.png"
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            out.append(out_path.name)
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass
    return out


def try_generate_global_and_dependence() -> list[str]:
    """Generate a global beeswarm and per-class dependence plots (top-1 feature).
    Expects the same artifacts as try_generate_beeswarms().
    """
    out = []
    npz = ROOT / "shap_values_sampled.npz"
    csv = ROOT / "shap_features_sampled.csv"
    jfe = ROOT / "shap_feature_names.json"
    if not npz.exists():
        npz = ROOT / "manuscript" / "results" / "shap_values_sampled.npz"
    if not csv.exists():
        csv = ROOT / "manuscript" / "results" / "shap_features_sampled.csv"
    if not jfe.exists():
        jfe = ROOT / "manuscript" / "results" / "shap_feature_names.json"
    if not (npz.exists() and csv.exists() and jfe.exists()):
        return out

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap = None
    try:
        import shap as _shap  # type: ignore
        shap = _shap
    except Exception:
        shap = None

    data = np.load(npz)
    S = data["S"]  # (Ns, F, C)
    X = pd.read_csv(csv)
    with open(jfe, "r") as f:
        j = json.load(f)
    pretty = j.get("pretty") or j.get("raw")
    if pretty is None:
        pretty = list(X.columns)

    OUT.mkdir(parents=True, exist_ok=True)

    # Global beeswarm: sum shap across classes to get signed (Ns,F)
    try:
        Sg = S.sum(axis=2)
        _set_pub_style()
        fig = plt.figure(figsize=(7.5, 5.5))
        if shap is not None:
            shap.summary_plot(Sg, X.values, feature_names=pretty, show=False, max_display=20)
        else:
            ax = plt.gca()
            _manual_beeswarm(ax, Sg, X.values, pretty, topn=20)
        fig.tight_layout()
        out_path = OUT / "supp_shap_beeswarm_global.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        out.append(out_path.name)
    except Exception:
        try: plt.close("all")
        except Exception: pass

    # Per-class dependence: top-1 feature per class (by mean |SHAP|)
    try:
        absS = np.abs(S)
        C = S.shape[-1]
        for ci in range(C):
            mean_abs = absS[:, :, ci].mean(axis=0)
            top_idx = int(mean_abs.argmax())
            fig = plt.figure(figsize=(5, 4))
            if shap is not None:
                shap.dependence_plot(top_idx, S[:, :, ci], X.values, feature_names=pretty, show=False, interaction_index=None)
            else:
                # Manual dependence: SHAP vs feature value scatter
                ax = plt.gca()
                x = X.values[:, top_idx]
                y = S[:, top_idx, ci]
                ax.scatter(x, y, s=6, alpha=0.6, c=x, cmap='coolwarm', linewidths=0)
                ax.set_xlabel(pretty[top_idx] if top_idx < len(pretty) else f"f{top_idx}")
                ax.set_ylabel('SHAP value')
                ax.axhline(0, color='#888', lw=0.8)
            fig.tight_layout()
            safe = str(top_idx)
            out_path = OUT / f"supp_shap_dependence_class_{ci}_feat_{safe}.png"
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            out.append(out_path.name)
    except Exception:
        try: plt.close("all")
        except Exception: pass

    return out


def try_generate_combo_top20_beeswarm() -> list[str]:
    """Create a 2‑panel figure: Left = top‑20 mean |SHAP| barh; Right = global beeswarm for the same 20 features.
    Uses S: (Ns, F, C), X: (Ns, F), and feature names from shap_feature_names.json.
    """
    out = []
    # Inputs
    npz = ROOT / "shap_values_sampled.npz"
    csv = ROOT / "shap_features_sampled.csv"
    jfe = ROOT / "shap_feature_names.json"
    if not npz.exists():
        npz = ROOT / "manuscript" / "results" / "shap_values_sampled.npz"
    if not csv.exists():
        csv = ROOT / "manuscript" / "results" / "shap_features_sampled.csv"
    if not jfe.exists():
        jfe = ROOT / "manuscript" / "results" / "shap_feature_names.json"
    if not (npz.exists() and csv.exists() and jfe.exists()):
        return out

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap = None
    try:
        import shap as _shap  # type: ignore
        shap = _shap
    except Exception:
        shap = None

    data = np.load(npz)
    S = data["S"]  # (Ns, F, C)
    X = pd.read_csv(csv)
    with open(jfe, "r") as f:
        j = json.load(f)
    names = j.get("pretty") or j.get("raw") or list(X.columns)
    # Compute global mean |SHAP| per feature and select top 20
    mean_abs = np.abs(S).mean(axis=(0,2))  # (F,)
    order = np.argsort(mean_abs)[-20:][::-1]
    feat20 = [names[i] if i < len(names) else f"f{i}" for i in order]
    vals20 = mean_abs[order]
    # Prepare arrays for beeswarm (global SHAP across classes)
    Sg = S.sum(axis=2)            # (Ns, F)
    Sg_sel = Sg[:, order]         # (Ns, 20)
    X_sel  = X.values[:, order]   # (Ns, 20)

    _set_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(32.0, 8.0), dpi=300)
    # Left: barh (largest at top, consistent order with beeswarm)
    ax = axes[0]
    y = np.arange(len(feat20))
    ax.barh(y, vals20, color="#6c8ae2")
    ax.set_yticks(y)
    ax.set_yticklabels(feat20)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("Top 20 Features (Global Mean |SHAP|)")
    ax.invert_yaxis()
    # Right: beeswarm using same 20 features
    ax2 = axes[1]
    # Force manual beeswarm to preserve left‑panel order
    _manual_beeswarm(ax2, Sg_sel, X_sel, feat20, topn=20, order_override=list(range(len(feat20))))
    fig.tight_layout(w_pad=0.8)
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "supp_shap_top20_plus_beeswarm.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    out.append(out_path.name)
    return out


def try_generate_beeswarm_grid() -> list[str]:
    """Create a 6‑panel beeswarm figure, one per class, using SHAP beeswarm images (top‑10).
    Generates each panel at native SHAP size, then assembles them into a mosaic to avoid squashing.
    """
    out = []
    # Inputs
    npz = ROOT / "shap_values_sampled.npz"
    csv = ROOT / "shap_features_sampled.csv"
    jfe = ROOT / "shap_feature_names.json"
    clsjson = ROOT / "shap_class_names.json"
    if not npz.exists():
        npz = ROOT / "manuscript" / "results" / "shap_values_sampled.npz"
    if not csv.exists():
        csv = ROOT / "manuscript" / "results" / "shap_features_sampled.csv"
    if not jfe.exists():
        jfe = ROOT / "manuscript" / "results" / "shap_feature_names.json"
    if not clsjson.exists():
        clsjson = ROOT / "manuscript" / "results" / "shap_class_names.json"
    if not (npz.exists() and csv.exists() and jfe.exists()):
        return out

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    try:
        import shap  # type: ignore
    except Exception:
        return out

    data = np.load(npz)
    S = data["S"]  # (Ns, F, C)
    X = pd.read_csv(csv)
    with open(jfe, "r") as f:
        j = json.load(f)
    pretty = j.get("pretty") or j.get("raw")
    if pretty is None:
        pretty = list(X.columns)
    # Class names mapping
    class_names = ["METPOL","GREROB","ACAKOA","DIOSAN","PROPAL","OTHER"]
    try:
        if clsjson.exists():
            with open(clsjson, "r") as f:
                cn = json.load(f).get("class_names", [])
                if cn and len(cn) == S.shape[-1]:
                    class_names = cn
    except Exception:
        pass

    # 1) Generate per-class PNGs at generous size (top-10)
    OUT.mkdir(parents=True, exist_ok=True)
    panel_paths = []
    for ci in range(min(S.shape[-1], 6)):
        fig = plt.figure(figsize=(10, 8), dpi=300)
        shap.summary_plot(S[:, :, ci], X.values, feature_names=pretty, show=False, max_display=10)
        fp = OUT / f"shap_beeswarm_class_{ci}_top10_panel.png"
        fig.tight_layout()
        fig.savefig(fp, dpi=300, bbox_inches="tight")
        plt.close(fig)
        panel_paths.append(fp)

    # 2) Read back the images and create a mosaic figure sized to the image pixels
    imgs = [mpimg.imread(p) for p in panel_paths]
    # Use first panel size as reference; assume similar sizes
    h_px, w_px = imgs[0].shape[0], imgs[0].shape[1]
    nrows, ncols = 3, 2
    dpi = 300
    fig_w_in = (w_px * ncols) / dpi
    fig_h_in = (h_px * nrows) / dpi
    _set_pub_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w_in, fig_h_in), dpi=dpi)
    letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if idx < len(imgs):
                ax.imshow(imgs[idx])
                ax.set_title(f"{letters[idx]} {class_names[idx]}")
            ax.axis('off')
            idx += 1
    fig.tight_layout(w_pad=0.02, h_pad=0.02)
    out_path = OUT / "supp_shap_beeswarms_6panel.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    out.append(out_path.name)
    return out


def main():
    copied = copy_existing()
    generated = try_generate_beeswarms()
    generated += try_generate_global_and_dependence()
    generated += try_generate_combo_top20_beeswarm()
    generated += try_generate_beeswarm_grid()
    # Also generate a 2‑bar WV vs LiDAR mean |SHAP| plot if artifacts exist
    try:
        # Prefer ROOT artifacts; fallback to manuscript/results
        npz = ROOT / "shap_values_sampled.npz"
        jfe = ROOT / "shap_feature_names.json"
        if not npz.exists():
            npz = ROOT / "manuscript" / "results" / "shap_values_sampled.npz"
        if not jfe.exists():
            jfe = ROOT / "manuscript" / "results" / "shap_feature_names.json"
        if npz.exists() and jfe.exists():
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            data = np.load(npz)
            S = data["S"]  # (Ns, F, C)
            with open(jfe, "r") as f:
                names = json.load(f)
            raw_names = names.get("raw") or names.get("pretty") or []
            if not raw_names:
                raw_names = [f"f{i}" for i in range(S.shape[1])]
            # Per-feature mean |SHAP|
            mean_abs = np.abs(S).mean(axis=(0,2))  # (F,)
            # Group mapping: WorldView = spectral + texture; LiDAR = structure + intensity
            def group_of(feat: str) -> str:
                if feat.startswith(("b0_","b1_","b2_","b3_","b4_","b5_","b6_","b7_")):
                    return "WorldView"
                if feat in {"pai","fhd","ch","mean_intensity","variance_intensity"}:
                    return "LiDAR"
                return "Other"
            groups = [group_of(nm) for nm in raw_names]
            import pandas as pd
            df = pd.DataFrame({"feature": raw_names, "group": groups, "mean_abs": mean_abs})
            # Filter to two macro-groups
            df2 = df[df["group"].isin(["WorldView","LiDAR"])].copy()
            # Compute mean per feature within each group (i.e., size-normalized)
            agg = df2.groupby("group", as_index=False)["mean_abs"].mean()
            # Save CSV and plot
            OUT.mkdir(parents=True, exist_ok=True)
            csv_path = OUT / "supp_table_shap_worldview_vs_lidar.csv"
            agg.to_csv(csv_path, index=False)
            _set_pub_style()
            fig, ax = plt.subplots(figsize=(4.4, 3.6), dpi=300)
            colors = ["#6c8ae2" if g=="WorldView" else "#a6d9a7" for g in agg["group"]]
            ax.bar(agg["group"], agg["mean_abs"], color=colors)
            ax.set_ylabel("Mean |SHAP| per feature")
            ax.set_xlabel("")
            ax.set_title("Average SHAP by Modality")
            # Add headroom for labels
            ymax = float(agg["mean_abs"].max()) if not agg.empty else 1.0
            ax.set_ylim(0, ymax * 1.25)
            for i, v in enumerate(agg["mean_abs"].values):
                ax.annotate(f"{v:.3f}", xy=(i, v), xytext=(0, 6), textcoords="offset points",
                            ha="center", va="bottom", fontsize=12)
            fig.tight_layout()
            out_path = OUT / "supp_shap_worldview_vs_lidar.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            generated.append(out_path.name)
    except Exception:
        # Non-fatal; skip WV vs LiDAR plot if artifacts missing
        pass
    print("Copied:", copied)
    print("Generated:", generated)


if __name__ == "__main__":
    main()
