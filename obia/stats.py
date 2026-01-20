import numpy as np
import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.ops import transform as shp_transform

from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm
from pyforestscan.handlers import read_lidar
from pyforestscan.calculate import assign_voxels, calculate_pad, calculate_fhd, calculate_chm, calculate_pai

from utils.utils import mask_image_with_polygon, crop_image_to_bbox

from scipy.stats import skew, kurtosis


def _create_empty_stats_columns(spectral_bands, textural_bands, calc_mean, calc_variance, calc_min, calc_max,
                                calc_skewness, calc_kurtosis,
                                calc_contrast, calc_dissimilarity, calc_homogeneity, calc_ASM, calc_energy,
                                calc_correlation,
                                calc_pai, calc_fhd, calc_ch, calc_mean_intensity, calc_variance_intensity):
    """
    Generate a list of columns for statistics based on the input parameters. This function is
    used to create the necessary column headers for storing spectral, textural, and pointcloud
    statistics, dynamically adjusted based on which statistics have been selected to calculate.
    The resulting list of columns is structured to accommodate both spectral and textural band
    data, as well as point cloud statistics, ensuring that only the required statistics are
    included based on the calculation flags provided.

    :param spectral_bands: A list of indices representing the spectral bands to include in the
        statistics. Each index corresponds to a band for which statistics may be calculated.
    :param textural_bands: A list of indices representing the textural bands to include in the
        statistics. Each index corresponds to a band for which statistics may be calculated.
    :param calc_mean: A boolean indicating whether to include mean statistics for each band.
    :param calc_variance: A boolean indicating whether to include variance statistics for each
        band.
    :param calc_min: A boolean indicating whether to include minimum value statistics for each
        band.
    :param calc_max: A boolean indicating whether to include maximum value statistics for each
        band.
    :param calc_skewness: A boolean indicating whether to include skewness statistics for each
        band.
    :param calc_kurtosis: A boolean indicating whether to include kurtosis statistics for each
        band.
    :param calc_contrast: A boolean indicating whether to include contrast statistics for each
        textural band.
    :param calc_dissimilarity: A boolean indicating whether to include dissimilarity statistics
        for each textural band.
    :param calc_homogeneity: A boolean indicating whether to include homogeneity statistics for
        each textural band.
    :param calc_ASM: A boolean indicating whether to include Angular Second Moment (ASM)
        statistics for each textural band.
    :param calc_energy: A boolean indicating whether to include energy statistics for each
        textural band.
    :param calc_correlation: A boolean indicating whether to include correlation statistics for
        each textural band.
    :param calc_pai: A boolean indicating whether to include Plant Area Index (PAI) statistics
        in the pointcloud stats.
    :param calc_fhd: A boolean indicating whether to include Foliage Height Diversity (FHD)
        statistics in the pointcloud stats.
    :param calc_ch: A boolean indicating whether to include canopy height (CH) statistics in
        the pointcloud stats.
    :param calc_mean_intensity: A boolean indicating whether to include mean intensity
        statistics in the pointcloud stats.
    :param calc_variance_intensity: A boolean indicating whether to include intensity variance
        statistics in the pointcloud stats.
    :return: A list of column names representing the structure of statistics to be collected.
        The columns include various statistical measures for each selected band and additional
        statistics for point clouds if specified.
    """
    columns = ['segment_id']

    spectral_stats = {
        "mean": calc_mean,
        "variance": calc_variance,
        "min": calc_min,
        "max": calc_max,
        "skewness": calc_skewness,
        "kurtosis": calc_kurtosis
    }

    textural_stats = {
        "contrast": calc_contrast,
        "dissimilarity": calc_dissimilarity,
        "homogeneity": calc_homogeneity,
        "ASM": calc_ASM,
        "energy": calc_energy,
        "correlation": calc_correlation
    }

    for band_index in spectral_bands:
        for stat, is_calculated in spectral_stats.items():
            if is_calculated:
                columns.append(f"b{band_index}_{stat}")

    for band_index in textural_bands:
        for stat, is_calculated in textural_stats.items():
            if is_calculated:
                columns.append(f"b{band_index}_{stat}")

    pointcloud_stats = {
        "pai": calc_pai,
        "fhd": calc_fhd,
        "ch": calc_ch,
        "mean_intensity": calc_mean_intensity,
        "variance_intensity": calc_variance_intensity
    }

    for stat, is_calculated in pointcloud_stats.items():
        if is_calculated:
            columns.append(stat)

    columns.append('geometry')

    return columns


def calculate_spectral_stats(
        image, statistics_bands,
        calc_mean=True, calc_variance=True, calc_min=True, calc_max=True, calc_skewness=True, calc_kurtosis=True
):
    """
    Calculates a range of statistical measures for specific bands of a spectral image. The statistics calculated are
    mean, variance, minimum, maximum, skewness, and kurtosis, based on the boolean flags provided. Missing or NaN
    values are ignored when computing statistics. If all values are NaN for a band, NaN is returned for each statistic.

    :param image: A multi-band spectral image where each band is represented in a separate 2D array.
    :type image: np.ndarray
    :param statistics_bands: List of band indices in the image for which statistics should be calculated.
    :type statistics_bands: List[int]
    :param calc_mean: Indicates whether to calculate the mean for the specified bands.
    :type calc_mean: bool
    :param calc_variance: Indicates whether to calculate the variance for the specified bands.
    :type calc_variance: bool
    :param calc_min: Indicates whether to calculate the minimum value for the specified bands.
    :type calc_min: bool
    :param calc_max: Indicates whether to calculate the maximum value for the specified bands.
    :type calc_max: bool
    :param calc_skewness: Indicates whether to calculate the skewness for the specified bands.
    :type calc_skewness: bool
    :param calc_kurtosis: Indicates whether to calculate the kurtosis for the specified bands.
    :type calc_kurtosis: bool
    :return: A dictionary containing the calculated statistics for each specified band, with keys formatted as
             'b{band_index}_statistic'.
    :rtype: dict
    """
    stats_dict = {}
    for band_index in statistics_bands:
        band_data = image[band_index, :, :]
        valid_mask = ~np.isnan(band_data)
        band_flat = band_data[valid_mask]

        band_prefix = f"b{band_index}"

        if band_flat.size == 0:
            if calc_mean:
                stats_dict[f"{band_prefix}_mean"] = np.nan
            if calc_variance:
                stats_dict[f"{band_prefix}_variance"] = np.nan
            if calc_min:
                stats_dict[f"{band_prefix}_min"] = np.nan
            if calc_max:
                stats_dict[f"{band_prefix}_max"] = np.nan
            if calc_skewness:
                stats_dict[f"{band_prefix}_skewness"] = np.nan
            if calc_kurtosis:
                stats_dict[f"{band_prefix}_kurtosis"] = np.nan
        else:
            if calc_mean:
                stats_dict[f"{band_prefix}_mean"] = np.mean(band_flat)
            if calc_variance:
                stats_dict[f"{band_prefix}_variance"] = np.var(band_flat)
            if calc_min:
                stats_dict[f"{band_prefix}_min"] = np.min(band_flat)
            if calc_max:
                stats_dict[f"{band_prefix}_max"] = np.max(band_flat)
            if calc_skewness:
                stats_dict[f"{band_prefix}_skewness"] = skew(band_flat)
            if calc_kurtosis:
                stats_dict[f"{band_prefix}_kurtosis"] = kurtosis(band_flat)
    return stats_dict


def calculate_textural_stats(
        image, textural_bands,
        calc_contrast=True, calc_dissimilarity=True, calc_homogeneity=True,
        calc_ASM=True, calc_energy=True, calc_correlation=True
):
    """
    Calculate textural statistics from a multi-band image based on specified bands.
    This function computes various textural properties such as contrast,
    dissimilarity, homogeneity, ASM, energy, and correlation. The calculations
    are performed on the specified bands of the input image, with options to
    select which statistics to compute. The results for each statistic are
    stored in a dictionary with the band index as the prefix to the keys.

    :param image: The multi-band image from which textural statistics are calculated.
    :type image: numpy.ndarray
    :param textural_bands: Indices of the bands to be used for calculating textural statistics.
    :type textural_bands: list of int
    :param calc_contrast: Flag indicating whether to calculate contrast.
    :type calc_contrast: bool, optional
    :param calc_dissimilarity: Flag indicating whether to calculate dissimilarity.
    :type calc_dissimilarity: bool, optional
    :param calc_homogeneity: Flag indicating whether to calculate homogeneity.
    :type calc_homogeneity: bool, optional
    :param calc_ASM: Flag indicating whether to calculate ASM (angular second moment).
    :type calc_ASM: bool, optional
    :param calc_energy: Flag indicating whether to calculate energy.
    :type calc_energy: bool, optional
    :param calc_correlation: Flag indicating whether to calculate correlation.
    :type calc_correlation: bool, optional
    :return: A dictionary containing the computed textural statistics for each band.
    :rtype: dict
    """
    stats_dict = {}

    for band_index in textural_bands:
        band_data = image[:, :, band_index]
        valid_mask = ~np.isnan(band_data)

        if not np.any(valid_mask):
            if calc_contrast:
                stats_dict[f"b{band_index}_contrast"] = np.nan
            if calc_dissimilarity:
                stats_dict[f"b{band_index}_dissimilarity"] = np.nan
            if calc_homogeneity:
                stats_dict[f"b{band_index}_homogeneity"] = np.nan
            if calc_ASM:
                stats_dict[f"b{band_index}_ASM"] = np.nan
            if calc_energy:
                stats_dict[f"b{band_index}_energy"] = np.nan
            if calc_correlation:
                stats_dict[f"b{band_index}_correlation"] = np.nan
            continue

        if np.isnan(band_data[valid_mask]).any():
            if calc_contrast:
                stats_dict[f"b{band_index}_contrast"] = np.nan
            if calc_dissimilarity:
                stats_dict[f"b{band_index}_dissimilarity"] = np.nan
            if calc_homogeneity:
                stats_dict[f"b{band_index}_homogeneity"] = np.nan
            if calc_ASM:
                stats_dict[f"b{band_index}_ASM"] = np.nan
            if calc_energy:
                stats_dict[f"b{band_index}_energy"] = np.nan
            if calc_correlation:
                stats_dict[f"b{band_index}_correlation"] = np.nan
            continue

        band_clean = band_data.copy()
        band_clean[~valid_mask] = 0

        band_prefix = f"b{band_index}"

        if np.issubdtype(band_clean.dtype, np.integer):
            glcm_input = band_clean.astype(np.uint8)
        else:
            band_min, band_max = np.min(band_clean), np.max(band_clean)
            if band_max == band_min:
                glcm_input = np.zeros(band_clean.shape, dtype=np.uint8)
            else:
                glcm_input = ((band_clean - band_min) / (band_max - band_min) * 255).astype(np.uint8)

        try:
            glcm = graycomatrix(
                glcm_input,
                distances=[2],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=256,
                symmetric=True,
                normed=True
            )
        except ValueError:
            if calc_contrast:
                stats_dict[f"{band_prefix}_contrast"] = np.nan
            if calc_dissimilarity:
                stats_dict[f"{band_prefix}_dissimilarity"] = np.nan
            if calc_homogeneity:
                stats_dict[f"{band_prefix}_homogeneity"] = np.nan
            if calc_ASM:
                stats_dict[f"{band_prefix}_ASM"] = np.nan
            if calc_energy:
                stats_dict[f"{band_prefix}_energy"] = np.nan
            if calc_correlation:
                stats_dict[f"{band_prefix}_correlation"] = np.nan
            continue

        if calc_contrast:
            stats_dict[f"{band_prefix}_contrast"] = np.mean(graycoprops(glcm, 'contrast'))
        if calc_dissimilarity:
            stats_dict[f"{band_prefix}_dissimilarity"] = np.mean(graycoprops(glcm, 'dissimilarity'))
        if calc_homogeneity:
            stats_dict[f"{band_prefix}_homogeneity"] = np.mean(graycoprops(glcm, 'homogeneity'))
        if calc_ASM:
            stats_dict[f"{band_prefix}_ASM"] = np.mean(graycoprops(glcm, 'ASM'))
        if calc_energy:
            stats_dict[f"{band_prefix}_energy"] = np.mean(graycoprops(glcm, 'energy'))
        if calc_correlation:
            stats_dict[f"{band_prefix}_correlation"] = np.mean(graycoprops(glcm, 'correlation'))

    return stats_dict


def calculate_structural_stats(
        pointcloud, voxel_resolution, calc_pai=True, calc_fhd=True, calc_ch=True
):
    """
    Calculate structural statistics from a point cloud.

    This function computes structural statistics of a 3D point cloud, such as
    plant area index (PAI), foliage height diversity (FHD), and canopy height (CH),
    based on the given voxel resolution. The computation of each statistic is
    conditional and can be controlled via corresponding boolean flags. The results
    are returned in a dictionary format.

    :param pointcloud: A 3D point cloud to analyze.
    :type pointcloud: array-like
    :param voxel_resolution: Resolution for voxel grid used in computations.
    :type voxel_resolution: tuple or list
    :param calc_pai: Flag to determine if plant area index (PAI) should be calculated.
    :type calc_pai: bool, optional
    :param calc_fhd: Flag to determine if foliage height diversity (FHD) should be calculated.
    :type calc_fhd: bool, optional
    :param calc_ch: Flag to determine if canopy height (CH) should be calculated.
    :type calc_ch: bool, optional
    :return: Dictionary containing the calculated statistics.
    :rtype: dict
    """
    stats_dict = {}
    try:
        voxels, extent = assign_voxels(pointcloud, voxel_resolution)
    except ValueError:
        print('oh no')
        if calc_pai:
            stats_dict['pai'] = np.nan
        if calc_fhd:
            stats_dict['fhd'] = np.nan
        if calc_ch:
            stats_dict['ch'] = np.nan
        return stats_dict
    if np.sum(voxels) == 0:
        if calc_pai:
            stats_dict['pai'] = np.nan
        if calc_fhd:
            stats_dict['fhd'] = np.nan
        if calc_ch:
            stats_dict['ch'] = np.nan
        return stats_dict
    if calc_pai:
        pad = calculate_pad(voxels, voxel_resolution[-1])
        pai = calculate_pai(pad, min_height=1, voxel_height=1)
        stats_dict['pai'] = np.mean(pai)
    if calc_fhd:
        fhd = calculate_fhd(voxels)
        stats_dict['fhd'] = np.mean(fhd)
    if calc_ch:
        ch, extent = calculate_chm(pointcloud, voxel_resolution, interpolation=None)
        stats_dict['ch'] = np.mean(ch)
    return stats_dict


def calculate_radiometric_stats(
        pointcloud, calc_mean_intensity=True, calc_variance_intensity=True
):
    """
    Calculate radiometric statistics from a point cloud data structure.

    This function evaluates the mean and variance of intensity values
    present in a point cloud dataset. The point cloud can be represented
    either as a structured NumPy array or a dictionary, with intensity
    data stored under the 'Intensity' field or key. Users can specify
    whether to calculate the mean or variance of the intensity values,
    or both, by setting the corresponding flags. If intensity data is
    not available, the function returns NaN for the requested statistics.

    :param pointcloud: A point cloud data containing intensity values,
       represented as a structured numpy array or a dictionary.
    :type pointcloud: numpy.ndarray or dict
    :param calc_mean_intensity: Flag to determine whether to calculate
       the mean of intensity values. Defaults to True.
    :type calc_mean_intensity: bool, optional
    :param calc_variance_intensity: Flag to determine whether to
       calculate the variance of intensity values. Defaults to True.
    :type calc_variance_intensity: bool, optional
    :return: A dictionary containing the calculated statistics:
       keys 'mean_intensity' and/or 'variance_intensity' with their
       corresponding values or NaN if the intensity data is missing.
    :rtype: dict
    """
    stats_dict = {}
    if isinstance(pointcloud, np.ndarray) and pointcloud.dtype.names:
        if 'Intensity' in pointcloud.dtype.names:
            intensities = pointcloud['Intensity']
        else:
            intensities = None
    elif isinstance(pointcloud, dict):
        intensities = pointcloud.get('Intensity', None)
    else:
        intensities = None

    if intensities is None:
        if calc_mean_intensity:
            stats_dict['mean_intensity'] = np.nan
        if calc_variance_intensity:
            stats_dict['variance_intensity'] = np.nan
        return stats_dict

    if intensities.size == 0:
        if calc_mean_intensity:
            stats_dict['mean_intensity'] = np.nan
        if calc_variance_intensity:
            stats_dict['variance_intensity'] = np.nan
        return stats_dict

    if calc_mean_intensity:
        stats_dict['mean_intensity'] = np.mean(intensities)
    if calc_variance_intensity:
        stats_dict['variance_intensity'] = np.var(intensities)
    return stats_dict


def create_objects(
        segments, image, ept=None, ept_srs=None, spectral_bands=None, textural_bands=None, voxel_resolution=None,
        calculate_spectral=True, calculate_textural=True, calculate_structural=False, calculate_radiometric=False,
        calc_mean=True, calc_variance=True, calc_min=True, calc_max=True, calc_skewness=True, calc_kurtosis=True,
        calc_contrast=True, calc_dissimilarity=True, calc_homogeneity=True, calc_ASM=True, calc_energy=True, calc_correlation=True,
        calc_pai=True, calc_fhd=True, calc_ch=True, calc_mean_intensity=True, calc_variance_intensity=True,
        ept_geom_srs=None
):
    """
    :param segments: GeoDataFrame containing the segmented regions to be analyzed.
    :param image: Object containing image data and metadata to be used for analysis. Should have 'img_data' attribute as 3D NumPy array and 'crs' attribute.
    :param ept: Optional; Path to the EPT (Entwine Point Tiles) point cloud data. Defaults to None.
    :param ept_srs: Optional; Spatial reference system for the EPT data. Required if ept is provided. Defaults to None.
    :param spectral_bands: Optional; List of spectral bands to be used in the analysis. Defaults to all available bands.
    :param textural_bands: Optional; List of textural bands to be used in the analysis. Defaults to all available bands.
    :param voxel_resolution: Optional; Voxel resolution for 3D point cloud data analysis. Required if ept is provided. Defaults to None.
    :param ept_geom_srs: Optional; CRS for reprojecting segment geometry when querying EPT. Defaults to None (no reprojection).
    :param calculate_spectral: Boolean; Whether to calculate spectral statistics. Defaults to True.
    :param calculate_textural: Boolean; Whether to calculate textural statistics. Defaults to True.
    :param calculate_structural: Boolean; Whether to calculate structural statistics using point cloud data. Defaults to False.
    :param calculate_radiometric: Boolean; Whether to calculate radiometric statistics using point cloud data. Defaults to False.
    :param calc_mean: Boolean; Whether to calculate the mean of the pixel values. Defaults to True.
    :param calc_variance: Boolean; Whether to calculate the variance of the pixel values. Defaults to True.
    :param calc_min: Boolean; Whether to calculate the minimum of the pixel values. Defaults to True.
    :param calc_max: Boolean; Whether to calculate the maximum of the pixel values. Defaults to True.
    :param calc_skewness: Boolean; Whether to calculate the skewness of the pixel values. Defaults to True.
    :param calc_kurtosis: Boolean; Whether to calculate the kurtosis of the pixel values. Defaults to True.
    :param calc_contrast: Boolean; Whether to calculate the contrast for textural analysis. Defaults to True.
    :param calc_dissimilarity: Boolean; Whether to calculate the dissimilarity for textural analysis. Defaults to True.
    :param calc_homogeneity: Boolean; Whether to calculate the homogeneity for textural analysis. Defaults to True.
    :param calc_ASM: Boolean; Whether to calculate the Angular Second Moment (ASM) for textural analysis. Defaults to True.
    :param calc_energy: Boolean; Whether to calculate the energy for textural analysis. Defaults to True.
    :param calc_correlation: Boolean; Whether to calculate the correlation for textural analysis. Defaults to True.
    :param calc_pai: Boolean; Whether to calculate the Plant Area Index (PAI) from point cloud data. Defaults to True.
    :param calc_fhd: Boolean; Whether to calculate the Foliage Height Diversity (FHD) from point cloud data. Defaults to True.
    :param calc_ch: Boolean; Whether to calculate the Canopy Height (CH) from point cloud data. Defaults to True.
    :param calc_mean_intensity: Boolean; Whether to calculate the mean intensity from point cloud data. Defaults to True.
    :param calc_variance_intensity: Boolean; Whether to calculate the variance of intensity from point cloud data. Defaults to True.
    :return: GeoDataFrame containing the calculated statistics for each segment.
    """
    if not (calculate_spectral or calculate_textural or calculate_structural or calculate_radiometric):
        raise ValueError(
            "At least one of 'calculate_spectral', 'calculate_textural', 'calculate_structural', or 'calculate_radiometric' must be True."
        )

    n_bands = image.rasterio_obj.count
    if spectral_bands is None:
        spectral_bands = list(range(n_bands))
    if textural_bands is None:
        textural_bands = list(range(n_bands))

    #
    # # ------------- KEY ADDITION -------------
    # # Filter out any band indices that exceed the number of bands in the image.
    # valid_spectral_bands = [b for b in spectral_bands if b < image.img_data.shape[0]]
    # if len(valid_spectral_bands) < len(spectral_bands):
    #     print(
    #         "Warning: Some spectral band indices were invalid and have been removed. "
    #         f"(Requested: {spectral_bands}, valid: {valid_spectral_bands})"
    #     )
    # spectral_bands = valid_spectral_bands
    #
    # valid_textural_bands = [b for b in textural_bands if b < image.img_data.shape[0]]
    # if len(valid_textural_bands) < len(textural_bands):
    #     print(
    #         "Warning: Some textural band indices were invalid and have been removed. "
    #         f"(Requested: {textural_bands}, valid: {valid_textural_bands})"
    #     )
    # textural_bands = valid_textural_bands
    # # ------------- END KEY ADDITION ----------

    columns = _create_empty_stats_columns(
        spectral_bands, textural_bands,
        calc_mean, calc_variance, calc_min, calc_max, calc_skewness, calc_kurtosis,
        calc_contrast, calc_dissimilarity, calc_homogeneity, calc_ASM, calc_energy, calc_correlation,
        calc_pai, calc_fhd, calc_ch, calc_mean_intensity, calc_variance_intensity
    )

    results = []
    transformer = None
    if ept_geom_srs is not None and segments.crs is not None:
        src_crs = CRS.from_user_input(segments.crs)
        dst_crs = CRS.from_user_input(ept_geom_srs)
        if src_crs != dst_crs:
            transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    for i, (idx, segment) in enumerate(tqdm(segments.iterrows(), total=len(segments), desc="Processing Segments")):
        geom = segment.geometry
        segment_id = segment['segment_id']

        cropped_img_data, cropped_transform = crop_image_to_bbox(image, geom)
        masked_img_data = mask_image_with_polygon(cropped_img_data, geom, cropped_transform)

        row = {
            'segment_id': segment_id,
            'geometry': geom
        }

        spectral_statistics = calculate_spectral_stats(
            masked_img_data, spectral_bands,
            calc_mean=calc_mean, calc_variance=calc_variance, calc_min=calc_min,
            calc_max=calc_max, calc_skewness=calc_skewness, calc_kurtosis=calc_kurtosis
        )
        row.update(spectral_statistics)

        # --- 2) Textural stats
        if calculate_textural and textural_bands:
            textural_statistics = calculate_textural_stats(
                masked_img_data,
                textural_bands,
                calc_contrast=calc_contrast,
                calc_dissimilarity=calc_dissimilarity,
                calc_homogeneity=calc_homogeneity,
                calc_ASM=calc_ASM,
                calc_energy=calc_energy,
                calc_correlation=calc_correlation
            )
            row.update(textural_statistics)

        if ept is not None:
            if ept_srs is None:
                raise ValueError("Error: 'ept_srs' must be provided when 'ept' is not None.")
            if voxel_resolution is None:
                raise ValueError("Error: 'voxel_resolution' must be provided when 'ept' is not None.")
            geom_ept = geom
            if transformer is not None:
                geom_ept = shp_transform(transformer.transform, geom)
            xmin, ymin, xmax, ymax = geom_ept.bounds
            bounds = ([xmin, xmax], [ymin, ymax])

            pointclouds = read_lidar(ept, ept_srs, bounds, crop_poly=True, poly=geom_ept.wkt)
            if not pointclouds:
                print("no point clouds found for segment", segment_id)
                if calculate_structural:
                    row['pai'] = np.nan
                    row['fhd'] = np.nan
                    row['ch'] = np.nan
                if calculate_radiometric:
                    row['mean_intensity'] = np.nan
                    row['variance_intensity'] = np.nan
            else:
                pointcloud = pointclouds[0]

                if calculate_structural:
                    structural_statistics = calculate_structural_stats(
                        pointcloud,
                        voxel_resolution,
                        calc_pai=calc_pai,
                        calc_fhd=calc_fhd,
                        calc_ch=calc_ch
                    )
                    row.update(structural_statistics)

                if calculate_radiometric:
                    radiometric_statistics = calculate_radiometric_stats(
                        pointcloud,
                        calc_mean_intensity=calc_mean_intensity,
                        calc_variance_intensity=calc_variance_intensity
                    )
                    row.update(radiometric_statistics)

        results.append(row)

    gdf = gpd.GeoDataFrame(results, columns=columns, crs=segments.crs)
    return gdf


# #!/usr/bin/env python3
# from __future__ import annotations
#
# import os
# import concurrent.futures as cf
# from functools import partial
# from typing import Dict, List, Tuple, Any
#
# import numpy as np
# import geopandas as gpd
# from skimage.feature import graycomatrix, graycoprops
# from tqdm import tqdm
# from scipy.stats import skew, kurtosis
#
# from pyforestscan.handlers import read_lidar
# from pyforestscan.calculate import (
#     assign_voxels,
#     calculate_pad,
#     calculate_fhd,
#     calculate_chm,
#     calculate_pai,
# )
#
# from utils.utils import mask_image_with_polygon, crop_image_to_bbox
#
# # ───────────────────────── column helpers ────────────────────────────────────
#
# def _create_empty_stats_columns(
#     spectral_bands: List[int],
#     textural_bands: List[int],
#     calc_mean: bool,
#     calc_variance: bool,
#     calc_min: bool,
#     calc_max: bool,
#     calc_skewness: bool,
#     calc_kurtosis: bool,
#     calc_contrast: bool,
#     calc_dissimilarity: bool,
#     calc_homogeneity: bool,
#     calc_ASM: bool,
#     calc_energy: bool,
#     calc_correlation: bool,
#     calc_pai: bool,
#     calc_fhd: bool,
#     calc_ch: bool,
#     calc_mean_intensity: bool,
#     calc_variance_intensity: bool,
# ) -> List[str]:
#     """Identical to original – builds full set of output columns."""
#
#     columns = ["segment_id"]
#
#     spectral_stats = {
#         "mean": calc_mean,
#         "variance": calc_variance,
#         "min": calc_min,
#         "max": calc_max,
#         "skewness": calc_skewness,
#         "kurtosis": calc_kurtosis,
#     }
#
#     textural_stats = {
#         "contrast": calc_contrast,
#         "dissimilarity": calc_dissimilarity,
#         "homogeneity": calc_homogeneity,
#         "ASM": calc_ASM,
#         "energy": calc_energy,
#         "correlation": calc_correlation,
#     }
#
#     for band in spectral_bands:
#         for stat, flag in spectral_stats.items():
#             if flag:
#                 columns.append(f"b{band}_{stat}")
#
#     for band in textural_bands:
#         for stat, flag in textural_stats.items():
#             if flag:
#                 columns.append(f"b{band}_{stat}")
#
#     pc_stats = {
#         "pai": calc_pai,
#         "fhd": calc_fhd,
#         "ch": calc_ch,
#         "mean_intensity": calc_mean_intensity,
#         "variance_intensity": calc_variance_intensity,
#     }
#     for stat, flag in pc_stats.items():
#         if flag:
#             columns.append(stat)
#
#     columns.append("geometry")
#     return columns
#
# # ───────────────────────── pixel‑based helpers (unchanged) ───────────────────
#
# def calculate_spectral_stats(
#     image: np.ndarray,
#     statistics_bands: List[int],
#     *,
#     calc_mean: bool = True,
#     calc_variance: bool = True,
#     calc_min: bool = True,
#     calc_max: bool = True,
#     calc_skewness: bool = True,
#     calc_kurtosis: bool = True,
# ) -> Dict[str, float]:
#     """Original implementation – unmodified."""
#
#     stats_dict: Dict[str, float] = {}
#     for band in statistics_bands:
#         band_data = image[band]
#         valid = ~np.isnan(band_data)
#         flat = band_data[valid]
#         pre = f"b{band}"
#         if flat.size == 0:
#             if calc_mean:
#                 stats_dict[f"{pre}_mean"] = np.nan
#             if calc_variance:
#                 stats_dict[f"{pre}_variance"] = np.nan
#             if calc_min:
#                 stats_dict[f"{pre}_min"] = np.nan
#             if calc_max:
#                 stats_dict[f"{pre}_max"] = np.nan
#             if calc_skewness:
#                 stats_dict[f"{pre}_skewness"] = np.nan
#             if calc_kurtosis:
#                 stats_dict[f"{pre}_kurtosis"] = np.nan
#             continue
#         if calc_mean:
#             stats_dict[f"{pre}_mean"] = float(np.mean(flat))
#         if calc_variance:
#             stats_dict[f"{pre}_variance"] = float(np.var(flat))
#         if calc_min:
#             stats_dict[f"{pre}_min"] = float(np.min(flat))
#         if calc_max:
#             stats_dict[f"{pre}_max"] = float(np.max(flat))
#         if calc_skewness:
#             stats_dict[f"{pre}_skewness"] = float(skew(flat))
#         if calc_kurtosis:
#             stats_dict[f"{pre}_kurtosis"] = float(kurtosis(flat))
#     return stats_dict
#
#
# def calculate_textural_stats(
#     image: np.ndarray,
#     textural_bands: List[int],
#     *,
#     calc_contrast: bool = True,
#     calc_dissimilarity: bool = True,
#     calc_homogeneity: bool = True,
#     calc_ASM: bool = True,
#     calc_energy: bool = True,
#     calc_correlation: bool = True,
# ) -> Dict[str, float]:
#     """Original implementation – unmodified."""
#
#     stats: Dict[str, float] = {}
#     for band in textural_bands:
#         band_im = image[:, :, band]
#         valid = ~np.isnan(band_im)
#         if not np.any(valid):
#             for name, flag in {
#                 "contrast": calc_contrast,
#                 "dissimilarity": calc_dissimilarity,
#                 "homogeneity": calc_homogeneity,
#                 "ASM": calc_ASM,
#                 "energy": calc_energy,
#                 "correlation": calc_correlation,
#             }.items():
#                 if flag:
#                     stats[f"b{band}_{name}"] = np.nan
#             continue
#
#         arr = band_im.copy()
#         arr[~valid] = 0
#         if not np.issubdtype(arr.dtype, np.integer):
#             mn, mx = np.min(arr), np.max(arr)
#             arr = np.zeros_like(arr) if mx == mn else ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
#         else:
#             arr = arr.astype(np.uint8)
#         try:
#             glcm = graycomatrix(
#                 arr,
#                 distances=[2],
#                 angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
#                 levels=256,
#                 symmetric=True,
#                 normed=True,
#             )
#         except ValueError:
#             for name, flag in {
#                 "contrast": calc_contrast,
#                 "dissimilarity": calc_dissimilarity,
#                 "homogeneity": calc_homogeneity,
#                 "ASM": calc_ASM,
#                 "energy": calc_energy,
#                 "correlation": calc_correlation,
#             }.items():
#                 if flag:
#                     stats[f"b{band}_{name}"] = np.nan
#             continue
#
#         lookup = {
#             "contrast": calc_contrast,
#             "dissimilarity": calc_dissimilarity,
#             "homogeneity": calc_homogeneity,
#             "ASM": calc_ASM,
#             "energy": calc_energy,
#             "correlation": calc_correlation,
#         }
#         for name, flag in lookup.items():
#             if flag:
#                 stats[f"b{band}_{name}"] = float(np.mean(graycoprops(glcm, name)))
#     return stats
#
# # ───────────────────────── point‑cloud helpers (unchanged) ───────────────────
#
# def calculate_structural_stats(
#     pointcloud: Any,
#     voxel_resolution: Tuple[float, float, float],
#     *,
#     calc_pai: bool = True,
#     calc_fhd: bool = True,
#     calc_ch: bool = True,
# ) -> Dict[str, float]:
#     """Original implementation – unmodified."""
#
#     stats: Dict[str, float] = {}
#     try:
#         voxels, _ = assign_voxels(pointcloud, voxel_resolution)
#     except ValueError:
#         if calc_pai:
#             stats["pai"] = np.nan
#         if calc_fhd:
#             stats["fhd"] = np.nan
#         if calc_ch:
#             stats["ch"] = np.nan
#         return stats
#
#     if np.sum(voxels) == 0:
#         if calc_pai:
#             stats["pai"] = np.nan
#         if calc_fhd:
#             stats["fhd"] = np.nan
#         if calc_ch:
#             stats["ch"] = np.nan
#         return stats
#     if calc_pai:
#         pad = calculate_pad(voxels, voxel_resolution[-1])
#         stats["pai"] = float(np.mean(calculate_pai(pad, voxel_height=1)))
#     if calc_fhd:
#         stats["fhd"] = float(np.mean(calculate_fhd(voxels)))
#     if calc_ch:
#         ch, _ = calculate_chm(pointcloud, voxel_resolution, interpolation=None)
#         stats["ch"] = float(np.mean(ch))
#     return stats
#
#
# def calculate_radiometric_stats(
#     pointcloud: Any,
#     *,
#     calc_mean_intensity: bool = True,
#     calc_variance_intensity: bool = True,
# ) -> Dict[str, float]:
#     """Original implementation – unmodified."""
#
#     stats: Dict[str, float] = {}
#     intens = None
#     if isinstance(pointcloud, np.ndarray) and pointcloud.dtype.names and "Intensity" in pointcloud.dtype.names:
#         intens = pointcloud["Intensity"]
#     elif isinstance(pointcloud, dict):
#         intens = pointcloud.get("Intensity")
#
#     if intens is None or intens.size == 0:
#         if calc_mean_intensity:
#             stats["mean_intensity"] = np.nan
#         if calc_variance_intensity:
#             stats["variance_intensity"] = np.nan
#         return stats
#
#     if calc_mean_intensity:
#         stats["mean_intensity"] = float(np.mean(intens))
#     if calc_variance_intensity:
#         stats["variance_intensity"] = float(np.var(intens))
#     return stats
#
# # ───────────────────────── internal worker util ──────────────────────────────
#
# def _process_segment(
#     seg_tuple: Tuple[int, gpd.GeoSeries],
#     *,
#     image_wrapper: Any,
#     spectral_bands: List[int],
#     textural_bands: List[int],
#     voxel_resolution: Tuple[float, float, float] | None,
#     ept: str | None,
#     ept_srs: str | None,
#     flags: Dict[str, Any],
# ) -> Dict[str, Any]:
#     """Compute all statistics for **one** segment."""
#
#     idx, segment = seg_tuple
#     geom = segment.geometry
#     seg_id = segment["segment_id"]
#
#     # Crop & mask
#     cropped, transform = crop_image_to_bbox(image_wrapper, geom)
#     masked = mask_image_with_polygon(cropped, geom, transform)
#
#     out: Dict[str, Any] = {"segment_id": seg_id, "geometry": geom}
#
#     # --- spectral
#     if flags["calc_spectral"]:
#         out.update(
#             calculate_spectral_stats(
#                 masked,
#                 spectral_bands,
#                 **flags["spectral_kwargs"],
#             )
#         )
#
#     # --- textural
#     if flags["calc_textural"] and textural_bands:
#         out.update(
#             calculate_textural_stats(
#                 masked,
#                 textural_bands,
#                 **flags["textural_kwargs"],
#             )
#         )
#
#     # --- point cloud (if available) ------------------------------------
#     if ept is not None:
#         xmin, ymin, xmax, ymax = geom.bounds
#         pcs = read_lidar(
#             ept,
#             ept_srs,
#             ([xmin, xmax], [ymin, ymax]),
#             crop_poly=True,
#             poly=geom.wkt,
#         )
#         if not pcs:
#             # fill NaNs so downstream code doesn't KeyError
#             if flags["calc_structural"]:
#                 out.update({k: np.nan for k in ("pai", "fhd", "ch") if k not in out})
#             if flags["calc_radiometric"]:
#                 out.update({k: np.nan for k in ("mean_intensity", "variance_intensity") if k not in out})
#         else:
#             pc = pcs[0]
#             if flags["calc_structural"]:
#                 out.update(
#                     calculate_structural_stats(
#                         pc,
#                         voxel_resolution,
#                         **flags["structural_kwargs"],
#                     )
#                 )
#             if flags["calc_radiometric"]:
#                 out.update(
#                     calculate_radiometric_stats(
#                         pc,
#                         **flags["radiometric_kwargs"],
#                     )
#                 )
#     return out
#
# # ───────────────────────── public API (parallel) ─────────────────────────────
#
# def create_objects(
#     segments: gpd.GeoDataFrame,
#     image: Any,
#     *,
#     ept: str | None = None,
#     ept_srs: str | None = None,
#     spectral_bands: List[int] | None = None,
#     textural_bands: List[int] | None = None,
#     voxel_resolution: Tuple[float, float, float] | None = None,
#     calculate_spectral: bool = True,
#     calculate_textural: bool = True,
#     calculate_structural: bool = False,
#     calculate_radiometric: bool = False,
#     # pixel/stat flags (unchanged defaults)
#     calc_mean: bool = True,
#     calc_variance: bool = True,
#     calc_min: bool = True,
#     calc_max: bool = True,
#     calc_skewness: bool = True,
#     calc_kurtosis: bool = True,
#     calc_contrast: bool = True,
#     calc_dissimilarity: bool = True,
#     calc_homogeneity: bool = True,
#     calc_ASM: bool = True,
#     calc_energy: bool = True,
#     calc_correlation: bool = True,
#     calc_pai: bool = True,
#     calc_fhd: bool = True,
#     calc_ch: bool = True,
#     calc_mean_intensity: bool = True,
#     calc_variance_intensity: bool = True,
#     # NEW (optional) -----------------------------------------------------------------
#     backend: str = "threads",  # "threads" | "processes" | "serial"
#     max_workers: int | None = None,
# ) -> gpd.GeoDataFrame:
#     """Parallel implementation – defaults mimic original behaviour"""
#
#     if not (calculate_spectral or calculate_textural or calculate_structural or calculate_radiometric):
#         raise ValueError("At least one calculation flag must be True.")
#
#     band_count = image.ds.count
#     if spectral_bands is None:
#         spectral_bands = list(range(band_count))
#     if textural_bands is None:
#         textural_bands = spectral_bands.copy()
#
#     # Build column order once
#     cols = _create_empty_stats_columns(
#         spectral_bands,
#         textural_bands,
#         calc_mean,
#         calc_variance,
#         calc_min,
#         calc_max,
#         calc_skewness,
#         calc_kurtosis,
#         calc_contrast,
#         calc_dissimilarity,
#         calc_homogeneity,
#         calc_ASM,
#         calc_energy,
#         calc_correlation,
#         calc_pai,
#         calc_fhd,
#         calc_ch,
#         calc_mean_intensity,
#         calc_variance_intensity,
#     )
#
#     # Bundle flags for quick passing to workers
#     flags = {
#         "calc_spectral": calculate_spectral,
#         "calc_textural": calculate_textural,
#         "calc_structural": calculate_structural,
#         "calc_radiometric": calculate_radiometric,
#         "spectral_kwargs": {
#             "calc_mean": calc_mean,
#             "calc_variance": calc_variance,
#             "calc_min": calc_min,
#             "calc_max": calc_max,
#             "calc_skewness": calc_skewness,
#             "calc_kurtosis": calc_kurtosis,
#         },
#         "textural_kwargs": {
#             "calc_contrast": calc_contrast,
#             "calc_dissimilarity": calc_dissimilarity,
#             "calc_homogeneity": calc_homogeneity,
#             "calc_ASM": calc_ASM,
#             "calc_energy": calc_energy,
#             "calc_correlation": calc_correlation,
#         },
#         "structural_kwargs": {
#             "calc_pai": calc_pai,
#             "calc_fhd": calc_fhd,
#             "calc_ch": calc_ch,
#         },
#         "radiometric_kwargs": {
#             "calc_mean_intensity": calc_mean_intensity,
#             "calc_variance_intensity": calc_variance_intensity,
#         },
#     }
#
#     # Decide executor ---------------------------------------------------------
#     if backend == "serial":
#         iterable = (
#             _process_segment(
#                 seg,
#                 image_wrapper=image,
#                 spectral_bands=spectral_bands,
#                 textural_bands=textural_bands,
#                 voxel_resolution=voxel_resolution,
#                 ept=ept,
#                 ept_srs=ept_srs,
#                 flags=flags,
#             )
#             for seg in segments.iterrows()
#         )
#         results = [r for r in tqdm(iterable, total=len(segments), desc="segments")]
#
#     else:
#         if backend not in {"threads", "processes"}:
#             raise ValueError("backend must be 'threads', 'processes', or 'serial'")
#         Executor = cf.ThreadPoolExecutor if backend == "threads" else cf.ProcessPoolExecutor
#         with Executor(max_workers=max_workers or os.cpu_count()) as pool:
#             futures = [
#                 pool.submit(
#                     _process_segment,
#                     seg,
#                     image_wrapper=image,
#                     spectral_bands=spectral_bands,
#                     textural_bands=textural_bands,
#                     voxel_resolution=voxel_resolution,
#                     ept=ept,
#                     ept_srs=ept_srs,
#                     flags=flags,
#                 )
#                 for seg in segments.iterrows()
#             ]
#             results = [
#                 f.result() for f in tqdm(cf.as_completed(futures), total=len(futures), desc="segments")
#             ]
#
#     gdf = gpd.GeoDataFrame(results, columns=cols, crs=segments.crs)
#     return gdf
