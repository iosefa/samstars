#!/usr/bin/env python3
"""
Run a trained polygon-refinement model on proposal outputs and write predicted crowns as polygons.

Inputs:
  - A proposal-records JSON produced by `samstars/segmentation/proposals.py`.
  - A StarDist model directory (Keras) containing a `stardist/` subfolder with
    `config.json` and weights (e.g., `weights_best.h5`).

Outputs:
  - `<out_dir>/crowns.gpkg` (layer: crowns) with fields: id, source, label, score
  - Optional per-input label rasters under `<out_dir>/labels/` if `--write-labels`

This is inference-only (no ground truth masks required).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from csbdeep.utils import normalize
from rasterio import features
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import MultiPolygon, Polygon, box, shape
from shapely.strtree import STRtree
from stardist.models import StarDist2D
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from samstars.data.scaling import load_scale_stats, scale_to_01


def _safe_source_name(source: str) -> str:
    """Return a filesystem-safe source name for checkpoint files."""
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(source))
    return safe or "record"


def _parse_csv_values(value: str | None) -> set[str]:
    """Parse a comma-separated CLI value into a set of non-empty strings."""
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _load_skip_sources(*, skip_sources: str | None, skip_sources_file: Path | None) -> set[str]:
    """Load source names to skip from CLI CSV and/or a text file."""
    sources = _parse_csv_values(skip_sources)
    if skip_sources_file is not None:
        with open(skip_sources_file) as f:
            for line in f:
                item = line.split("#", 1)[0].strip()
                if item:
                    sources.add(item)
    return sources


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON via a temporary path so completed markers are atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _write_features_gpkg(
    path: Path,
    features_list: list[dict],
    *,
    schema: dict,
    crs_wkt: str | None,
) -> None:
    """Write polygon features to a GeoPackage via an atomic replacement."""
    import fiona

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.gpkg")
    if tmp.exists():
        tmp.unlink()
    with fiona.open(
        tmp,
        "w",
        driver="GPKG",
        layer="crowns",
        schema=schema,
        crs_wkt=crs_wkt,
    ) as dst:
        for idx, feat in enumerate(features_list, start=1):
            dst.write(
                {
                    "geometry": feat["geometry"].__geo_interface__,
                    "properties": {
                        "id": int(idx),
                        "source": feat["source"],
                        "label": int(feat["label"]),
                        "score": float(feat["score"]),
                    },
                }
            )
    tmp.replace(path)


def _read_features_gpkg(path: Path) -> list[dict]:
    """Read polygon features from a refinement checkpoint GeoPackage."""
    import fiona

    features_list: list[dict] = []
    with fiona.open(path, layer="crowns") as src:
        for feat in src:
            props = dict(feat["properties"])
            score = props.get("score", float("nan"))
            features_list.append(
                {
                    "geometry": shape(feat["geometry"]),
                    "source": str(props.get("source", "")),
                    "label": int(props.get("label", 0)),
                    "score": float(score) if score is not None else float("nan"),
                }
            )
    return features_list


def _write_labels_atomic(path: Path, labels: np.ndarray, *, meta: dict, dtype: str) -> None:
    """Write a label raster via an atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.tif")
    if tmp.exists():
        tmp.unlink()
    out_meta = meta.copy()
    out_meta.update(count=1, dtype=dtype, compress="deflate")
    with rasterio.open(tmp, "w", **out_meta) as out_ds:
        out_ds.write(labels.astype(dtype), 1)
    tmp.replace(path)


def _checkpoint_complete(shard: Path, done_marker: Path, label_path: Path | None) -> bool:
    """Return true only for fully completed per-record checkpoint outputs."""
    if not shard.exists() or not done_marker.exists():
        return False
    if label_path is not None and not label_path.exists():
        return False
    return True


def _bootstrap_checkpoint_from_label(
    *,
    label_path: Path,
    source: str,
    shard_path: Path,
    done_marker: Path,
    schema: dict,
    crs_wkt: str | None,
    min_area_m2: float,
) -> int:
    """Create a per-record checkpoint from an existing label raster."""
    with rasterio.open(label_path) as src:
        labels = src.read(1)
        tfm = src.transform

    record_features = [
        {
            "geometry": poly,
            "source": str(source),
            "label": int(lbl_id),
            "score": float("nan"),
        }
        for lbl_id, poly in _iter_polys_from_labels(
            labels,
            transform=tfm,
            min_area_m2=float(min_area_m2),
        )
    ]
    _write_features_gpkg(shard_path, record_features, schema=schema, crs_wkt=crs_wkt)
    _write_json_atomic(
        done_marker,
        {
            "source": str(source),
            "input": str(label_path),
            "features": len(record_features),
            "bootstrapped_from_label": True,
        },
    )
    return len(record_features)


def _align_to_shape(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Internal helper function."""
    arr = np.asarray(arr)
    th, tw = target_shape
    arr = arr[:th, :tw]
    ah, aw = arr.shape
    if ah < th or aw < tw:
        pad_h = max(0, th - ah)
        pad_w = max(0, tw - aw)
        arr = np.pad(arr, ((0, pad_h), (0, pad_w)), constant_values=0)
    return arr


def _read_float_raster(path: str | Path) -> np.ndarray:
    """Internal helper function."""
    path = str(path)
    if path.endswith(".tif") or path.endswith(".tiff"):
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)
    return np.load(path).astype(np.float32)


def _extract_thresholds(thr) -> tuple[float | None, float | None]:
    """Internal helper function."""
    if thr is None:
        return None, None
    if isinstance(thr, dict):
        prob = thr.get("prob")
        nms = thr.get("nms")
        if prob is None:
            prob = thr.get("prob_thresh")
        if nms is None:
            nms = thr.get("nms_thresh")
        return prob, nms
    prob = getattr(thr, "prob", None)
    nms = getattr(thr, "nms", None)
    return prob, nms


def _load_thresholds_from_json(model_dir: Path, model_name: str) -> tuple[float | None, float | None]:
    """Internal helper function."""
    thr_path = model_dir / model_name / "thresholds.json"
    if not thr_path.exists():
        return None, None
    data = json.load(open(thr_path))
    return _extract_thresholds(data)


def _guess_source_name(input_path: str | Path) -> str:
    """Internal helper function."""
    stem = Path(str(input_path)).stem
    for suf in ("_sam_index", "_sam_prob", "_sam_logit", "_index", "_prob"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem


def _iter_polys_from_labels(
    labels: np.ndarray,
    *,
    transform,
    min_area_m2: float = 0.0,
):
    """Internal helper function."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D, got shape={labels.shape}")
    if not np.any(labels):
        return

    # polygons per label (streaming); rasterio.features.shapes yields geometries for each region.
    for geom_ras, val in features.shapes(
        labels.astype(np.int32),
        mask=(labels != 0),
        transform=transform,
        connectivity=8,
    ):
        lbl_id = int(val)
        if lbl_id == 0:
            continue
        g = shape(geom_ras)
        if g.is_empty:
            continue

        parts = list(g.geoms) if isinstance(g, MultiPolygon) else [g]
        for p in parts:
            if not isinstance(p, Polygon):
                continue
            if p.is_empty:
                continue
            if min_area_m2 > 0 and float(p.area) < float(min_area_m2):
                continue
            yield lbl_id, p


def _first_crs_wkt(recs: list[dict], input_key: str) -> str | None:
    """Internal helper function."""
    for rec in recs:
        x_path = rec.get(input_key, "")
        if not x_path or not Path(x_path).exists():
            continue
        with rasterio.open(x_path) as src:
            return src.crs.to_wkt() if src.crs else None
    return None


def _iter_patch_windows(
    shape: tuple[int, int],
    *,
    patch_size_px: int,
    overlap_px: int,
):
    """Yield overlapping patch windows covering a raster."""
    height, width = shape
    if patch_size_px <= 0:
        raise ValueError("patch_size_px must be > 0")
    if overlap_px < 0:
        raise ValueError("overlap_px must be >= 0")
    if overlap_px >= patch_size_px:
        raise ValueError("overlap_px must be smaller than patch_size_px")

    step = patch_size_px - overlap_px

    def _starts(size: int) -> list[int]:
        if size <= patch_size_px:
            return [0]
        starts = list(range(0, max(1, size - patch_size_px + 1), step))
        last = size - patch_size_px
        if starts[-1] != last:
            starts.append(last)
        return starts

    row_starts = _starts(height)
    col_starts = _starts(width)
    for row_off in row_starts:
        for col_off in col_starts:
            h = min(patch_size_px, height - row_off)
            w = min(patch_size_px, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=w, height=h)


def _window_keep_extent(
    window: Window,
    *,
    full_shape: tuple[int, int],
    transform,
    overlap_m: float,
):
    """Return the geospatial keep extent for a patch window."""
    patch_tfm = window_transform(window, transform)
    left = float(patch_tfm.c)
    top = float(patch_tfm.f)
    right = left + float(window.width) * float(patch_tfm.a)
    bottom = top + float(window.height) * float(patch_tfm.e)
    if bottom > top:
        bottom, top = top, bottom

    row_off = int(window.row_off)
    col_off = int(window.col_off)
    height, width = full_shape
    half_overlap = max(0.0, overlap_m) / 2.0

    shrink_left = half_overlap if col_off > 0 else 0.0
    shrink_right = half_overlap if col_off + int(window.width) < width else 0.0
    shrink_top = half_overlap if row_off > 0 else 0.0
    shrink_bottom = half_overlap if row_off + int(window.height) < height else 0.0

    keep_left = left + shrink_left
    keep_right = right - shrink_right
    keep_bottom = bottom + shrink_bottom
    keep_top = top - shrink_top
    if keep_left >= keep_right or keep_bottom >= keep_top:
        return box(left, bottom, right, top)
    return box(keep_left, keep_bottom, keep_right, keep_top)


def _score_value(score: float) -> float:
    """Return a sortable score value with NaN treated as negative infinity."""
    if score is None or not np.isfinite(score):
        return float("-inf")
    return float(score)


def _iou(poly_a: Polygon, poly_b: Polygon) -> float:
    """Compute polygon IoU."""
    inter = float(poly_a.intersection(poly_b).area)
    if inter <= 0:
        return 0.0
    union = float(poly_a.area + poly_b.area - inter)
    if union <= 0:
        return 0.0
    return inter / union


def _query_tree_indices(tree: STRtree, geom):
    """Return query results as integer indices across Shapely versions."""
    hits = tree.query(geom)
    if len(hits) == 0:
        return []
    first = hits[0]
    if isinstance(first, (int, np.integer)):
        return [int(i) for i in hits]
    geom_to_idx = {id(g): i for i, g in enumerate(tree.geometries)}
    return [geom_to_idx[id(g)] for g in hits]


def _dedup_features_by_iou(
    features_list: list[dict],
    *,
    iou_thr: float,
) -> list[dict]:
    """Keep the higher-score polygon when IoU exceeds the threshold."""
    if iou_thr <= 0 or len(features_list) <= 1:
        return features_list

    geoms = [feat["geometry"] for feat in features_list]
    tree = STRtree(geoms)
    order = sorted(
        range(len(features_list)),
        key=lambda i: _score_value(features_list[i]["score"]),
        reverse=True,
    )
    suppressed = np.zeros(len(features_list), dtype=bool)
    kept: list[dict] = []

    for idx in order:
        if suppressed[idx]:
            continue
        feat = features_list[idx]
        kept.append(feat)
        geom = feat["geometry"]
        for j in _query_tree_indices(tree, geom):
            if j == idx or suppressed[j]:
                continue
            other = features_list[j]
            if _iou(geom, other["geometry"]) >= iou_thr:
                suppressed[j] = True
    return kept


def main() -> None:
    """Run the command-line entry point."""
    pa = argparse.ArgumentParser(description="Run StarDist refinement from proposal records.")
    pa.add_argument("--proposal-records", type=Path, required=True, help="Proposal-record JSON from the proposal stage")
    pa.add_argument("--model-dir", type=Path, required=True, help="Directory containing the StarDist model subdir")
    pa.add_argument("--model-name", type=str, default="stardist", help="Model subdir name (default: stardist)")
    pa.add_argument("--out-dir", type=Path, required=True)

    pa.add_argument("--prob-thr", type=float, default=None)
    pa.add_argument("--nms-thr", type=float, default=None)

    pa.add_argument(
        "--input-key",
        type=str,
        default="index",
        help="Proposal-record key to use as StarDist input (default: index). Use file_name for raw SAM prob.",
    )
    pa.add_argument("--normalize", action="store_true", help="Enable csbdeep percentile normalization")
    pa.add_argument("--no-normalize", action="store_true", help="Disable csbdeep percentile normalization")

    # Aux
    pa.add_argument("--use-aux", action="store_true", help="Use CHM aux channel (stacks [primary, chm01])")
    pa.add_argument("--scale-stats", type=Path, default=None, help="Global scaling stats JSON (required for --use-aux)")
    pa.add_argument("--chm-key", type=str, default="chm", help="Proposal-record key for CHM path")

    # Outputs
    pa.add_argument("--write-labels", action="store_true", help="Write per-input label rasters under out_dir/labels")
    pa.add_argument("--preds-name", type=str, default="crowns.gpkg", help="Output GeoPackage filename")
    pa.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from per-record checkpoints under out_dir/record_shards. "
            "Completed records are skipped and the final GeoPackage is rebuilt at the end."
        ),
    )
    pa.add_argument(
        "--resume-from-labels",
        action="store_true",
        help=(
            "With --resume and --write-labels, bootstrap missing per-record checkpoints "
            "from existing label rasters. Bootstrapped features have NaN scores."
        ),
    )
    pa.add_argument(
        "--skip-sources",
        type=str,
        default=None,
        help="Comma-separated source names to skip before StarDist inference, for example tile_086_031_0.",
    )
    pa.add_argument(
        "--skip-sources-file",
        type=Path,
        default=None,
        help="Optional text file of source names to skip, one per line. Lines may contain # comments.",
    )
    pa.add_argument(
        "--print-source-before-run",
        action="store_true",
        help="Print each source immediately before running or bootstrapping it; useful for native crashes.",
    )
    pa.add_argument(
        "--window-size-m",
        type=float,
        default=None,
        help="Optional internal StarDist patch size in map units; if omitted, predict on the full tile.",
    )
    pa.add_argument(
        "--window-overlap-m",
        type=float,
        default=0.0,
        help="Overlap between internal StarDist patches in map units (default: 0).",
    )

    # Polygon filtering (useful when inputs overlap)
    pa.add_argument(
        "--edge-buffer-px",
        type=float,
        default=0.5,
        help="Drop polygons within this many pixels of the input raster edge (default: 0.5)",
    )
    pa.add_argument("--min-area-m2", type=float, default=0.0, help="Drop polygons smaller than this area (m^2)")
    pa.add_argument(
        "--dedup-iou-thr",
        type=float,
        default=0.0,
        help="Optional final polygon dedup IoU threshold; keep the higher-score polygon when IoU exceeds this value.",
    )
    pa.add_argument(
        "--log",
        action="store_true",
        help="Show verbose refinement logs, including StarDist internal tile progress.",
    )
    args = pa.parse_args()

    if args.normalize and args.no_normalize:
        raise SystemExit("Use only one of --normalize or --no-normalize.")
    if args.resume_from_labels and not args.resume:
        raise SystemExit("--resume-from-labels requires --resume.")
    if args.resume_from_labels and not args.write_labels:
        raise SystemExit("--resume-from-labels requires --write-labels.")
    if args.window_size_m is not None and args.window_size_m <= 0:
        raise SystemExit("--window-size-m must be > 0")
    if args.window_overlap_m < 0:
        raise SystemExit("--window-overlap-m must be >= 0")
    if args.window_size_m is not None and args.window_overlap_m >= args.window_size_m:
        raise SystemExit("--window-overlap-m must be smaller than --window-size-m")
    if not (0.0 <= args.dedup_iou_thr < 1.0):
        raise SystemExit("--dedup-iou-thr must be in [0, 1).")
    if args.normalize:
        do_normalize = True
    elif args.no_normalize:
        do_normalize = False
    else:
        do_normalize = True

    recs = json.load(open(args.proposal_records))
    if not isinstance(recs, list):
        raise SystemExit(f"{args.proposal_records} is not a list of records.")
    skip_sources = _load_skip_sources(
        skip_sources=args.skip_sources,
        skip_sources_file=args.skip_sources_file,
    )

    scale_stats = load_scale_stats(args.scale_stats) if args.scale_stats else None
    if args.use_aux and scale_stats is None:
        raise SystemExit("--scale-stats is required when --use-aux is set")

    chm_lo = chm_hi = None
    if scale_stats:
        chm_lo = float(scale_stats.get("chm_01", {}).get("lo", 0.0))
        chm_hi = float(scale_stats.get("chm_01", {}).get("hi", 1.0))

    model = StarDist2D(None, name=args.model_name, basedir=str(args.model_dir))

    prob_thr = args.prob_thr
    nms_thr = args.nms_thr
    if prob_thr is None and nms_thr is None:
        p1, n1 = _extract_thresholds(getattr(model, "thresholds", None))
        if p1 is None or n1 is None:
            p1, n1 = _load_thresholds_from_json(args.model_dir, args.model_name)
        prob_thr, nms_thr = p1, n1
    if (prob_thr is None) ^ (nms_thr is None):
        raise SystemExit("Provide both --prob-thr and --nms-thr, or neither to use optimized thresholds.")
    if args.log and prob_thr is not None and nms_thr is not None:
        print(f"Using effective thresholds: prob_thresh={prob_thr}, nms_thresh={nms_thr}.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_gpkg = args.out_dir / args.preds_name
    labels_dir = args.out_dir / "labels"
    shards_dir = args.out_dir / "record_shards"
    if args.resume:
        shards_dir.mkdir(parents=True, exist_ok=True)
        if args.write_labels:
            labels_dir.mkdir(parents=True, exist_ok=True)
    else:
        if out_gpkg.exists():
            out_gpkg.unlink()
        tmp_out_gpkg = out_gpkg.with_suffix(".tmp.gpkg")
        if tmp_out_gpkg.exists():
            tmp_out_gpkg.unlink()
        if shards_dir.exists():
            shutil.rmtree(shards_dir)
        shards_dir.mkdir(parents=True, exist_ok=True)
        if args.write_labels:
            if labels_dir.exists():
                shutil.rmtree(labels_dir)
            labels_dir.mkdir(parents=True, exist_ok=True)

    schema = {
        "geometry": "Polygon",
        "properties": {
            "id": "int",
            "source": "str",
            "label": "int",
            "score": "float",
        },
    }

    wrote = 0
    crs_wkt = _first_crs_wkt(recs, args.input_key)
    skipped = 0
    skipped_by_request = 0

    pbar = tqdm(
        recs,
        desc="StarDist refinement",
        unit="record",
        dynamic_ncols=True,
    )
    for rec in pbar:
            x_path = rec.get(args.input_key, "")
            if not x_path or not Path(x_path).exists():
                continue

            source = _guess_source_name(x_path)
            pbar.set_postfix_str(source, refresh=False)
            if source in skip_sources or _safe_source_name(source) in skip_sources:
                skipped_by_request += 1
                continue
            safe_source = _safe_source_name(source)
            shard_path = shards_dir / f"{safe_source}.gpkg"
            done_marker = shards_dir / f"{safe_source}.done.json"
            label_path = labels_dir / f"{source}_labels.tif" if args.write_labels else None
            if args.resume and _checkpoint_complete(shard_path, done_marker, label_path):
                skipped += 1
                continue
            current_source_path = args.out_dir / "current_refinement_source.txt"
            current_source_path.write_text(f"{source}\n")
            if args.print_source_before_run:
                print(f"processing source: {source}", flush=True)
            if args.resume_from_labels and label_path is not None and label_path.exists():
                try:
                    _bootstrap_checkpoint_from_label(
                        label_path=label_path,
                        source=source,
                        shard_path=shard_path,
                        done_marker=done_marker,
                        schema=schema,
                        crs_wkt=crs_wkt,
                        min_area_m2=float(args.min_area_m2),
                    )
                    skipped += 1
                    continue
                except Exception as exc:
                    if args.log:
                        print(f"Could not bootstrap checkpoint from {label_path}: {exc}")

            with rasterio.open(x_path) as src:
                x = src.read(1).astype(np.float32)
                tfm = src.transform
                res_m = float(max(abs(tfm.a), abs(tfm.e)))
                edge_tol_m = float(max(0.0, args.edge_buffer_px) * res_m)
                extent = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                border = extent.boundary

            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            if args.use_aux:
                chm_path = rec.get(args.chm_key, "")
                if not chm_path or not Path(chm_path).exists():
                    continue
                chm = _read_float_raster(chm_path)
                chm = _align_to_shape(chm, x.shape)
                if scale_stats is not None:
                    chm = scale_to_01(chm, chm_lo, chm_hi)
                x_in = np.stack([x, chm.astype(np.float32)], axis=-1).astype(np.float32)
            else:
                x_in = x[..., None].astype(np.float32)

            if do_normalize:
                x_in = normalize(x_in, 1, 99.8, axis=(0, 1)).astype(np.float32)

            predict_kwargs = {
                "show_tile_progress": bool(args.log),
            }
            if prob_thr is not None and nms_thr is not None:
                predict_kwargs["prob_thresh"] = prob_thr
                predict_kwargs["nms_thresh"] = nms_thr

            windowed = args.window_size_m is not None
            window_polys: list[tuple[Polygon, int, float]] = []

            if not windowed:
                predict_kwargs["n_tiles"] = model._guess_n_tiles(x_in)
                labels, details = model.predict_instances(x_in, **predict_kwargs)

                score_by_label = {}
                if isinstance(details, dict) and "prob" in details:
                    probs = np.asarray(details["prob"], dtype=float)
                    score_by_label = {i + 1: float(p) for i, p in enumerate(probs)}

                for lbl_id, poly in _iter_polys_from_labels(
                    labels,
                    transform=tfm,
                    min_area_m2=float(args.min_area_m2),
                ):
                    if edge_tol_m > 0 and poly.distance(border) < edge_tol_m:
                        continue
                    score = float(score_by_label.get(int(lbl_id), float("nan")))
                    window_polys.append((poly, int(lbl_id), score))

                if args.write_labels:
                    with rasterio.open(x_path) as src:
                        meta = src.meta.copy()
                    _write_labels_atomic(label_path, labels, meta=meta, dtype="uint16")
            else:
                patch_size_px = max(1, int(round(float(args.window_size_m) / res_m)))
                overlap_px = max(0, int(round(float(args.window_overlap_m) / res_m)))

                for window in _iter_patch_windows(
                    x.shape,
                    patch_size_px=patch_size_px,
                    overlap_px=overlap_px,
                ):
                    row_off = int(window.row_off)
                    col_off = int(window.col_off)
                    h = int(window.height)
                    w = int(window.width)
                    x_patch = x_in[row_off : row_off + h, col_off : col_off + w]
                    patch_kwargs = dict(predict_kwargs)
                    patch_kwargs["n_tiles"] = model._guess_n_tiles(x_patch)
                    labels, details = model.predict_instances(x_patch, **patch_kwargs)

                    score_by_label = {}
                    if isinstance(details, dict) and "prob" in details:
                        probs = np.asarray(details["prob"], dtype=float)
                        score_by_label = {i + 1: float(p) for i, p in enumerate(probs)}

                    patch_tfm = window_transform(window, tfm)
                    keep_extent = _window_keep_extent(
                        window,
                        full_shape=x.shape,
                        transform=tfm,
                        overlap_m=float(args.window_overlap_m),
                    )

                    for lbl_id, poly in _iter_polys_from_labels(
                        labels,
                        transform=patch_tfm,
                        min_area_m2=float(args.min_area_m2),
                    ):
                        if edge_tol_m > 0 and poly.distance(border) < edge_tol_m:
                            continue
                        if not poly.representative_point().within(keep_extent):
                            continue
                        score = float(score_by_label.get(int(lbl_id), float("nan")))
                        window_polys.append((poly, int(lbl_id), score))

                if args.write_labels:
                    with rasterio.open(x_path) as src:
                        meta = src.meta.copy()
                    shapes = [
                        (poly, int(i + 1))
                        for i, (poly, _lbl_id, _score) in enumerate(window_polys)
                    ]
                    labels_full = features.rasterize(
                        shapes=shapes,
                        out_shape=x.shape,
                        transform=tfm,
                        fill=0,
                        dtype="uint32",
                    )
                    _write_labels_atomic(label_path, labels_full, meta=meta, dtype="uint32")

            record_features = [
                {
                    "geometry": poly,
                    "source": str(source),
                    "label": int(lbl_id),
                    "score": float(score),
                }
                for poly, lbl_id, score in window_polys
            ]
            _write_features_gpkg(shard_path, record_features, schema=schema, crs_wkt=crs_wkt)
            _write_json_atomic(
                done_marker,
                {
                    "source": str(source),
                    "input": str(x_path),
                    "features": len(record_features),
                },
            )

    out_features: list[dict] = []
    for marker in sorted(shards_dir.glob("*.done.json")):
        shard_path = marker.with_suffix("").with_suffix(".gpkg")
        if not shard_path.exists():
            continue
        out_features.extend(_read_features_gpkg(shard_path))

    out_features = _dedup_features_by_iou(
        out_features,
        iou_thr=float(args.dedup_iou_thr),
    )

    _write_features_gpkg(out_gpkg, out_features, schema=schema, crs_wkt=crs_wkt)
    wrote = len(out_features)

    print(f"✓ wrote {wrote} crown polygons to {out_gpkg}")
    if args.resume:
        print(f"✓ skipped {skipped} completed refinement records")
    if skipped_by_request:
        print(f"✓ skipped {skipped_by_request} requested sources")


if __name__ == "__main__":
    main()
