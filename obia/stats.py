#!/usr/bin/env python3
from __future__ import annotations

import os
import concurrent.futures as cf
from functools import partial
from typing import Dict, List, Tuple, Any

import numpy as np
import geopandas as gpd
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm
from scipy.stats import skew, kurtosis

from pyforestscan.handlers import read_lidar
from pyforestscan.calculate import (
    assign_voxels,
    calculate_pad,
    calculate_fhd,
    calculate_chm,
    calculate_pai,
)

from utils.utils import mask_image_with_polygon, crop_image_to_bbox

# ───────────────────────── column helpers ────────────────────────────────────

def _create_empty_stats_columns(
    spectral_bands: List[int],
    textural_bands: List[int],
    calc_mean: bool,
    calc_variance: bool,
    calc_min: bool,
    calc_max: bool,
    calc_skewness: bool,
    calc_kurtosis: bool,
    calc_contrast: bool,
    calc_dissimilarity: bool,
    calc_homogeneity: bool,
    calc_ASM: bool,
    calc_energy: bool,
    calc_correlation: bool,
    calc_pai: bool,
    calc_fhd: bool,
    calc_ch: bool,
    calc_mean_intensity: bool,
    calc_variance_intensity: bool,
) -> List[str]:
    """Identical to original – builds full set of output columns."""

    columns = ["segment_id"]

    spectral_stats = {
        "mean": calc_mean,
        "variance": calc_variance,
        "min": calc_min,
        "max": calc_max,
        "skewness": calc_skewness,
        "kurtosis": calc_kurtosis,
    }

    textural_stats = {
        "contrast": calc_contrast,
        "dissimilarity": calc_dissimilarity,
        "homogeneity": calc_homogeneity,
        "ASM": calc_ASM,
        "energy": calc_energy,
        "correlation": calc_correlation,
    }

    for band in spectral_bands:
        for stat, flag in spectral_stats.items():
            if flag:
                columns.append(f"b{band}_{stat}")

    for band in textural_bands:
        for stat, flag in textural_stats.items():
            if flag:
                columns.append(f"b{band}_{stat}")

    pc_stats = {
        "pai": calc_pai,
        "fhd": calc_fhd,
        "ch": calc_ch,
        "mean_intensity": calc_mean_intensity,
        "variance_intensity": calc_variance_intensity,
    }
    for stat, flag in pc_stats.items():
        if flag:
            columns.append(stat)

    columns.append("geometry")
    return columns

# ───────────────────────── pixel‑based helpers (unchanged) ───────────────────

def calculate_spectral_stats(
    image: np.ndarray,
    statistics_bands: List[int],
    *,
    calc_mean: bool = True,
    calc_variance: bool = True,
    calc_min: bool = True,
    calc_max: bool = True,
    calc_skewness: bool = True,
    calc_kurtosis: bool = True,
) -> Dict[str, float]:
    """Original implementation – unmodified."""

    stats_dict: Dict[str, float] = {}
    for band in statistics_bands:
        band_data = image[band]
        valid = ~np.isnan(band_data)
        flat = band_data[valid]
        pre = f"b{band}"
        if flat.size == 0:
            if calc_mean:
                stats_dict[f"{pre}_mean"] = np.nan
            if calc_variance:
                stats_dict[f"{pre}_variance"] = np.nan
            if calc_min:
                stats_dict[f"{pre}_min"] = np.nan
            if calc_max:
                stats_dict[f"{pre}_max"] = np.nan
            if calc_skewness:
                stats_dict[f"{pre}_skewness"] = np.nan
            if calc_kurtosis:
                stats_dict[f"{pre}_kurtosis"] = np.nan
            continue
        if calc_mean:
            stats_dict[f"{pre}_mean"] = float(np.mean(flat))
        if calc_variance:
            stats_dict[f"{pre}_variance"] = float(np.var(flat))
        if calc_min:
            stats_dict[f"{pre}_min"] = float(np.min(flat))
        if calc_max:
            stats_dict[f"{pre}_max"] = float(np.max(flat))
        if calc_skewness:
            stats_dict[f"{pre}_skewness"] = float(skew(flat))
        if calc_kurtosis:
            stats_dict[f"{pre}_kurtosis"] = float(kurtosis(flat))
    return stats_dict


def calculate_textural_stats(
    image: np.ndarray,
    textural_bands: List[int],
    *,
    calc_contrast: bool = True,
    calc_dissimilarity: bool = True,
    calc_homogeneity: bool = True,
    calc_ASM: bool = True,
    calc_energy: bool = True,
    calc_correlation: bool = True,
) -> Dict[str, float]:
    """Original implementation – unmodified."""

    stats: Dict[str, float] = {}
    for band in textural_bands:
        band_im = image[:, :, band]
        valid = ~np.isnan(band_im)
        if not np.any(valid):
            for name, flag in {
                "contrast": calc_contrast,
                "dissimilarity": calc_dissimilarity,
                "homogeneity": calc_homogeneity,
                "ASM": calc_ASM,
                "energy": calc_energy,
                "correlation": calc_correlation,
            }.items():
                if flag:
                    stats[f"b{band}_{name}"] = np.nan
            continue

        arr = band_im.copy()
        arr[~valid] = 0
        if not np.issubdtype(arr.dtype, np.integer):
            mn, mx = np.min(arr), np.max(arr)
            arr = np.zeros_like(arr) if mx == mn else ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        try:
            glcm = graycomatrix(
                arr,
                distances=[2],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=256,
                symmetric=True,
                normed=True,
            )
        except ValueError:
            for name, flag in {
                "contrast": calc_contrast,
                "dissimilarity": calc_dissimilarity,
                "homogeneity": calc_homogeneity,
                "ASM": calc_ASM,
                "energy": calc_energy,
                "correlation": calc_correlation,
            }.items():
                if flag:
                    stats[f"b{band}_{name}"] = np.nan
            continue

        lookup = {
            "contrast": calc_contrast,
            "dissimilarity": calc_dissimilarity,
            "homogeneity": calc_homogeneity,
            "ASM": calc_ASM,
            "energy": calc_energy,
            "correlation": calc_correlation,
        }
        for name, flag in lookup.items():
            if flag:
                stats[f"b{band}_{name}"] = float(np.mean(graycoprops(glcm, name)))
    return stats

# ───────────────────────── point‑cloud helpers (unchanged) ───────────────────

def calculate_structural_stats(
    pointcloud: Any,
    voxel_resolution: Tuple[float, float, float],
    *,
    calc_pai: bool = True,
    calc_fhd: bool = True,
    calc_ch: bool = True,
) -> Dict[str, float]:
    """Original implementation – unmodified."""

    stats: Dict[str, float] = {}
    try:
        voxels, _ = assign_voxels(pointcloud, voxel_resolution)
    except ValueError:
        if calc_pai:
            stats["pai"] = np.nan
        if calc_fhd:
            stats["fhd"] = np.nan
        if calc_ch:
            stats["ch"] = np.nan
        return stats

    if np.sum(voxels) == 0:
        if calc_pai:
            stats["pai"] = np.nan
        if calc_fhd:
            stats["fhd"] = np.nan
        if calc_ch:
            stats["ch"] = np.nan
        return stats
    if calc_pai:
        pad = calculate_pad(voxels, voxel_resolution[-1])
        stats["pai"] = float(np.mean(calculate_pai(pad, voxel_height=1)))
    if calc_fhd:
        stats["fhd"] = float(np.mean(calculate_fhd(voxels)))
    if calc_ch:
        ch, _ = calculate_chm(pointcloud, voxel_resolution, interpolation=None)
        stats["ch"] = float(np.mean(ch))
    return stats


def calculate_radiometric_stats(
    pointcloud: Any,
    *,
    calc_mean_intensity: bool = True,
    calc_variance_intensity: bool = True,
) -> Dict[str, float]:
    """Original implementation – unmodified."""

    stats: Dict[str, float] = {}
    intens = None
    if isinstance(pointcloud, np.ndarray) and pointcloud.dtype.names and "Intensity" in pointcloud.dtype.names:
        intens = pointcloud["Intensity"]
    elif isinstance(pointcloud, dict):
        intens = pointcloud.get("Intensity")

    if intens is None or intens.size == 0:
        if calc_mean_intensity:
            stats["mean_intensity"] = np.nan
        if calc_variance_intensity:
            stats["variance_intensity"] = np.nan
        return stats

    if calc_mean_intensity:
        stats["mean_intensity"] = float(np.mean(intens))
    if calc_variance_intensity:
        stats["variance_intensity"] = float(np.var(intens))
    return stats

# ───────────────────────── internal worker util ──────────────────────────────

def _process_segment(
    seg_tuple: Tuple[int, gpd.GeoSeries],
    *,
    image_wrapper: Any,
    spectral_bands: List[int],
    textural_bands: List[int],
    voxel_resolution: Tuple[float, float, float] | None,
    ept: str | None,
    ept_srs: str | None,
    flags: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute all statistics for **one** segment."""

    idx, segment = seg_tuple
    geom = segment.geometry
    seg_id = segment["segment_id"]

    # Crop & mask
    cropped, transform = crop_image_to_bbox(image_wrapper, geom)
    masked = mask_image_with_polygon(cropped, geom, transform)

    out: Dict[str, Any] = {"segment_id": seg_id, "geometry": geom}

    # --- spectral
    if flags["calc_spectral"]:
        out.update(
            calculate_spectral_stats(
                masked,
                spectral_bands,
                **flags["spectral_kwargs"],
            )
        )

    # --- textural
    if flags["calc_textural"] and textural_bands:
        out.update(
            calculate_textural_stats(
                masked,
                textural_bands,
                **flags["textural_kwargs"],
            )
        )

    # --- point cloud (if available) ------------------------------------
    if ept is not None:
        xmin, ymin, xmax, ymax = geom.bounds
        pcs = read_lidar(
            ept,
            ept_srs,
            ([xmin, xmax], [ymin, ymax]),
            crop_poly=True,
            poly=geom.wkt,
        )
        if not pcs:
            # fill NaNs so downstream code doesn't KeyError
            if flags["calc_structural"]:
                out.update({k: np.nan for k in ("pai", "fhd", "ch") if k not in out})
            if flags["calc_radiometric"]:
                out.update({k: np.nan for k in ("mean_intensity", "variance_intensity") if k not in out})
        else:
            pc = pcs[0]
            if flags["calc_structural"]:
                out.update(
                    calculate_structural_stats(
                        pc,
                        voxel_resolution,
                        **flags["structural_kwargs"],
                    )
                )
            if flags["calc_radiometric"]:
                out.update(
                    calculate_radiometric_stats(
                        pc,
                        **flags["radiometric_kwargs"],
                    )
                )
    return out

# ───────────────────────── public API (parallel) ─────────────────────────────

def create_objects(
    segments: gpd.GeoDataFrame,
    image: Any,
    *,
    ept: str | None = None,
    ept_srs: str | None = None,
    spectral_bands: List[int] | None = None,
    textural_bands: List[int] | None = None,
    voxel_resolution: Tuple[float, float, float] | None = None,
    calculate_spectral: bool = True,
    calculate_textural: bool = True,
    calculate_structural: bool = False,
    calculate_radiometric: bool = False,
    # pixel/stat flags (unchanged defaults)
    calc_mean: bool = True,
    calc_variance: bool = True,
    calc_min: bool = True,
    calc_max: bool = True,
    calc_skewness: bool = True,
    calc_kurtosis: bool = True,
    calc_contrast: bool = True,
    calc_dissimilarity: bool = True,
    calc_homogeneity: bool = True,
    calc_ASM: bool = True,
    calc_energy: bool = True,
    calc_correlation: bool = True,
    calc_pai: bool = True,
    calc_fhd: bool = True,
    calc_ch: bool = True,
    calc_mean_intensity: bool = True,
    calc_variance_intensity: bool = True,
    # NEW (optional) -----------------------------------------------------------------
    backend: str = "threads",  # "threads" | "processes" | "serial"
    max_workers: int | None = None,
) -> gpd.GeoDataFrame:
    """Parallel implementation – defaults mimic original behaviour"""

    if not (calculate_spectral or calculate_textural or calculate_structural or calculate_radiometric):
        raise ValueError("At least one calculation flag must be True.")

    if spectral_bands is None:
        spectral_bands = list(range(image.img_data.shape[0]))
    if textural_bands is None:
        textural_bands = list(range(image.img_data.shape[0]))

    # Build column order once
    cols = _create_empty_stats_columns(
        spectral_bands,
        textural_bands,
        calc_mean,
        calc_variance,
        calc_min,
        calc_max,
        calc_skewness,
        calc_kurtosis,
        calc_contrast,
        calc_dissimilarity,
        calc_homogeneity,
        calc_ASM,
        calc_energy,
        calc_correlation,
        calc_pai,
        calc_fhd,
        calc_ch,
        calc_mean_intensity,
        calc_variance_intensity,
    )

    # Bundle flags for quick passing to workers
    flags = {
        "calc_spectral": calculate_spectral,
        "calc_textural": calculate_textural,
        "calc_structural": calculate_structural,
        "calc_radiometric": calculate_radiometric,
        "spectral_kwargs": {
            "calc_mean": calc_mean,
            "calc_variance": calc_variance,
            "calc_min": calc_min,
            "calc_max": calc_max,
            "calc_skewness": calc_skewness,
            "calc_kurtosis": calc_kurtosis,
        },
        "textural_kwargs": {
            "calc_contrast": calc_contrast,
            "calc_dissimilarity": calc_dissimilarity,
            "calc_homogeneity": calc_homogeneity,
            "calc_ASM": calc_ASM,
            "calc_energy": calc_energy,
            "calc_correlation": calc_correlation,
        },
        "structural_kwargs": {
            "calc_pai": calc_pai,
            "calc_fhd": calc_fhd,
            "calc_ch": calc_ch,
        },
        "radiometric_kwargs": {
            "calc_mean_intensity": calc_mean_intensity,
            "calc_variance_intensity": calc_variance_intensity,
        },
    }

    # Decide executor ---------------------------------------------------------
    if backend == "serial":
        iterable = (
            _process_segment(
                seg,
                image_wrapper=image,
                spectral_bands=spectral_bands,
                textural_bands=textural_bands,
                voxel_resolution=voxel_resolution,
                ept=ept,
                ept_srs=ept_srs,
                flags=flags,
            )
            for seg in segments.iterrows()
        )
        results = [r for r in tqdm(iterable, total=len(segments), desc="segments")]

    else:
        if backend not in {"threads", "processes"}:
            raise ValueError("backend must be 'threads', 'processes', or 'serial'")
        Executor = cf.ThreadPoolExecutor if backend == "threads" else cf.ProcessPoolExecutor
        with Executor(max_workers=max_workers or os.cpu_count()) as pool:
            futures = [
                pool.submit(
                    _process_segment,
                    seg,
                    image_wrapper=image,
                    spectral_bands=spectral_bands,
                    textural_bands=textural_bands,
                    voxel_resolution=voxel_resolution,
                    ept=ept,
                    ept_srs=ept_srs,
                    flags=flags,
                )
                for seg in segments.iterrows()
            ]
            results = [
                f.result() for f in tqdm(cf.as_completed(futures), total=len(futures), desc="segments")
            ]

    gdf = gpd.GeoDataFrame(results, columns=cols, crs=segments.crs)
    return gdf
