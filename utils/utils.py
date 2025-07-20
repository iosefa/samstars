import geopandas as gpd
import numpy as np
import rasterio

from affine import Affine
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from typing import List, Tuple


def label_segments(segments: gpd.GeoDataFrame, labelled_points: gpd.GeoDataFrame) -> Tuple[
    gpd.GeoDataFrame, List[str]]:
    """
    :param segments: A GeoDataFrame representing the segments to be labeled.
    :param labelled_points: A GeoDataFrame representing the labeled points used for segment labeling.
    :return: A tuple containing a GeoDataFrame with labeled segments and a list of segment IDs for mixed segments.
    """
    mixed_segments = []
    labelled_segments = segments.copy()
    intersections = gpd.sjoin(labelled_segments, labelled_points, how='inner', predicate='intersects')

    for polygon_id, group in intersections.groupby(intersections.index):
        classes = group['class'].unique()

        if len(classes) == 1:
            labelled_segments.loc[polygon_id, 'feature_class'] = classes[0]
        else:
            segment_id = group['segment_id'].values[0]
            mixed_segments.append(segment_id)

    labelled_segments = labelled_segments[labelled_segments['feature_class'].notna()]

    return labelled_segments, mixed_segments


def crop_image_to_bbox(image, geom):
    """
    Crop the image data to the bounding box of the given geometry.

    :param image: The Image object containing the image data and rasterio object.
    :param geom: The geometry (Polygon) used to derive the bounding box for cropping.
    :return: Cropped image data as a NumPy array and the updated transform.
    """
    xmin, ymin, xmax, ymax = geom.bounds
    window = from_bounds(xmin, ymin, xmax, ymax, transform=image.transform)
    cropped_img_data = image.rasterio_obj.read(window=window)
    cropped_transform = image.rasterio_obj.window_transform(window)

    return cropped_img_data, cropped_transform


def mask_image_with_polygon(cropped_img_data, polygon, cropped_transform):
    """
    Masks all pixels outside the polygon for the given cropped image.

    :param cropped_img_data: The cropped image data as a NumPy array.
    :param polygon: The geometry (Polygon) used for masking.
    :param cropped_transform: The affine transform for the cropped image.
    :return: Masked image data as a NumPy array.
    """
    height, width = cropped_img_data.shape[1], cropped_img_data.shape[2]
    mask = geometry_mask([polygon], transform=cropped_transform, invert=True, out_shape=(height, width))
    mask_expanded = np.expand_dims(mask, axis=0)  # Add an extra dimension for bands
    masked_img_data = np.where(mask_expanded, cropped_img_data, np.nan)

    return masked_img_data


def _write_geotiff(pil_image, output_path, crs, transform):
    """
    Writes the given PIL image as a GeoTIFF file to the specified output path. This function
    handles both single-band and multi-band image data. It transforms the PIL image data into
    a format suitable for raster processing and writes it using the GeoTIFF driver in
    `rasterio`. The Coordinate Reference System (CRS) and affine transform associated with
    the image must be provided to correctly georeference the output GeoTIFF.

    :param pil_image: The PIL image to be written as a GeoTIFF. It can be a single-band or
        multi-band image.
    :type pil_image: PIL.Image.Image
    :param output_path: The file path where the GeoTIFF will be saved.
    :type output_path: str
    :param crs: Coordinate Reference System for the output GeoTIFF. This should be provided
        in a format understood by `rasterio`, such as an EPSG code.
    :type crs: any
    :param transform: The affine transform to be applied to the image. This should align the
        pixel grid with the georeferenced coordinate system.
    :type transform: Affine
    :return: None
    """
    data = np.array(pil_image)
    data = data.astype(np.uint8)

    bands, height, width = data.shape if len(data.shape) == 3 else (1, *data.shape)

    new_image = rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs=crs,
        transform=transform)
    if bands > 1:
        for i in range(bands):
            new_image.write(data[i], i + 1)
    else:
        new_image.write(data, 1)
    new_image.close()
    print(f"Done Writing GeoTIFF at {output_path}")
