"""General utility functions for SAMStars."""

import numpy as np
from PIL.Image import fromarray
import rasterio
import numpy as np
import cv2
import rasterio

from scipy.ndimage import uniform_filter


import json
import geopandas as gpd
import numpy as np

from affine import Affine
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import Polygon
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


def save_deepforest_predictions_to_gpkg(
    df,               # DataFrame from DeepForest predict_image(...)
    tile_name,        # e.g. "img_007.jpg"
    transforms_json,  # path to transforms.json
    output_gpkg       # path to output .gpkg
):
    """
    Convert DeepForest predictions (pixel coords) to georeferenced polygons
    and save them in a GeoPackage.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: ["xmin","ymin","xmax","ymax","label","score"].
        Optional "image_path" if you prefer to verify or filter by tile_name.
    tile_name : str
        The tile's filename (e.g. "img_007.jpg") that matches the key in transforms.json.
    transforms_json : str
        File path to the transforms.json created by tile_and_process(...).
        We'll load the tile's Affine transform + CRS from there.
    output_gpkg : str
        Path to the output GeoPackage file to create/overwrite.
    """
    # 1) Load transforms.json
    with open(transforms_json, "r") as f:
        transforms_dict = json.load(f)

    if tile_name not in transforms_dict:
        # todo: raise warning or something
        print(f"Tile '{tile_name}' not found in transforms.json. Skipping.")
        return

    # 2) Reconstruct the tile's Affine + CRS
    # transforms_dict[tile_name] = { "transform": [a,b,c,d,e,f], "crs": "EPSG:..." }
    tinfo = transforms_dict[tile_name]
    a, b, c, d, e, f = tinfo["transform"]
    tile_affine = Affine(a, b, c, d, e, f)
    crs_str = tinfo["crs"]  # e.g. "EPSG:32610"

    # 3) Convert each bounding box to a polygon in map coords
    records = []
    for idx, row in df.iterrows():
        # read pixel coords
        xmin_pix = row["xmin"]
        ymin_pix = row["ymin"]
        xmax_pix = row["xmax"]
        ymax_pix = row["ymax"]

        # top-left corner => (col=xmin_pix, row=ymin_pix)
        x1, y1 = tile_affine * (xmin_pix, ymin_pix)
        # top-right corner => (col=xmax_pix, row=ymin_pix)
        x2, y2 = tile_affine * (xmax_pix, ymin_pix)
        # bottom-right corner => (col=xmax_pix, row=ymax_pix)
        x3, y3 = tile_affine * (xmax_pix, ymax_pix)
        # bottom-left corner => (col=xmin_pix, row=ymax_pix)
        x4, y4 = tile_affine * (xmin_pix, ymax_pix)

        # build polygon (4-corner bounding box)
        geom = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)])

        # store relevant columns
        rec = {
            "label": row.get("label", "Tree"),   # default label if not present
            "score": row.get("score", None),
            "geometry": geom
        }
        records.append(rec)

    if not records:
        # todo: raise warning or something
        print(f"No predictions to save for tile {tile_name}")
        return

    gdf = gpd.GeoDataFrame(records, crs=crs_str)
    # 4) Save to GPKG
    gdf.to_file(output_gpkg, driver="GPKG")


def open_binary_geotiff_as_mask(mask_path):
    """
    :param mask_path: Path to the binary GeoTIFF file to be opened.
    :return: A tuple containing the binary mask array, bounding box, affine transform, and profile of the raster.
    """
    with rasterio.open(mask_path) as src:
        mask_array = src.read(1).astype(bool)  # Read the mask as a binary array
        transform = src.transform  # Get the affine transform of the raster
        width, height = src.width, src.height  # Get the dimensions of the raster
        profile = src.profile  # Capture the profile to use later

        # Calculate the bounding box from the transform and dimensions
        left, top = transform * (0, 0)
        right, bottom = transform * (width, height)
        bbox = (left, bottom, right, top)

    return mask_array, bbox, transform, profile


def rescale_to_8bit(image, min=2, max=98):
    """
    Rescales an image to an 8-bit unsigned integer representation by stretching
    and clamping its intensity values between specified percentile ranges. This
    transformation enhances the dynamic range of the image and is useful for
    various image processing tasks.

    :param image: A NumPy array representing the input image to be rescaled.
                  This image should have intensity values that need adjustment.
    :param min: The lower percentile cutoff for rescaling, with a default
                value of 2, indicating that the intensity values below this
                percentile will be clipped.
    :param max: The upper percentile cutoff for rescaling, with a default
                value of 98, indicating that the intensity values above this
                percentile will be clipped.
    :return: A NumPy array of the same shape as the input image, with the
             intensity values rescaled to the range [0, 255] as 8-bit
             unsigned integers.
    """
    p_min, p_max = np.percentile(image, (min, max))

    if p_min == p_max:
        return np.zeros(image.shape, dtype=np.uint8)

    scaled_image = 255 * (image - p_min) / (p_max - p_min)

    scaled_image = np.clip(scaled_image, 0, 255)

    return scaled_image.astype(np.uint8)


def apply_histogram_equalization(image):
    """
    Apply histogram equalization to the input image. Histogram equalization
    enhances the contrast of an image by effectively spreading out the most
    frequent intensity values. This function supports both grayscale and
    RGB images, converting them to grayscale before applying the equalization
    process. The equalized grayscale image is then stacked into a 3-channel
    image.

    :param image:
        The image to be equalized. It can be a 2D array for grayscale or a
        3D array for RGB images.
    :type image: numpy.ndarray

    :return:
        A new image where the histogram has been equalized. The returned
        image will be a 3-channel RGB image irrespective of whether the
        input was grayscale or RGB.
    :rtype: numpy.ndarray
    """
    if image.ndim == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image

    equalized_img = cv2.equalizeHist(image_gray)

    return np.stack((equalized_img,) * 3, axis=-1)


def apply_clahe(image):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to an
    image for improving contrast in both grayscale and color images.

    CLAHE works by dividing the image into small tiles and enhancing each
    tile separately. It is particularly useful for improving the visibility
    of features in images with varying lighting conditions or shadows.

    :param image: The input image to which CLAHE should be applied. This can
                  be either a grayscale or a color image. If the image has
                  three dimensions, it is assumed to be in the BGR color
                  format where CLAHE is applied to each channel separately.
    :return: The image with applied CLAHE, having enhanced contrast compared
             to the original input.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    if image.ndim == 3:
        channels = cv2.split(image)
        clahe_channels = [clahe.apply(ch) for ch in channels]
        image_clahe = cv2.merge(clahe_channels)
    else:
        image_clahe = clahe.apply(image)

    return image_clahe


def rgb_to_gray(rgb):
    """ITU-R 601 grayscale (expects float32 array in [0,1])."""
    coeffs = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (rgb * coeffs).sum(axis=-1)


def variance_of_laplacian(gray, win):
    """Local variance of the 3×3 Laplacian, window = win×win pixels."""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    mean   = uniform_filter(lap,        size=win)
    mean2  = uniform_filter(lap * lap,  size=win)
    return mean2 - mean ** 2           # σ² = E[X²] − E[X]²


def laplacian(in_path, out_path, win):
    """Laplacian."""
    with rasterio.open(in_path) as src:
        meta = src.meta.copy()

        # 1. Read visible bands (Blue, Green, Red for WorldView)
        vis_bands = [2, 3, 5]
        arr = src.read(vis_bands).astype(np.float32)

        # 2. Normalise each band to [0,1]  ——  *** FIX HERE ***
        band_min = arr.min(axis=(1, 2), keepdims=True)
        band_rng = np.ptp(arr, axis=(1, 2), keepdims=True) + 1e-8
        arr = (arr - band_min) / band_rng

        # 3. Grayscale → Laplacian-variance sharpness
        arr = np.transpose(arr, (1, 2, 0))          # (rows, cols, bands)
        gray  = rgb_to_gray(arr)
        sharp = variance_of_laplacian(gray, win)

        # 4. Stretch to 0–1 for display
        lo, hi = np.percentile(sharp, [2, 98])
        sharp  = np.clip((sharp - lo) / (hi - lo), 0, 1)

        # 5. Write out
        meta.update(dtype="float32", count=1, nodata=None)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(sharp.astype("float32"), 1)

class Image:
    """
        Image
        A class for handling geographic raster image data, providing utilities to manage its properties and to convert it into visualization formats.

        Attributes
        ----------
        img_data : numpy.ndarray
            The image data array.
        crs : str or dict
            Coordinate reference system of the image.
        transform : affine.Affine
            Affine transformation matrix.
        affine_transformation : affine.Affine
            Deprecated attribute for affine transformation.
        rasterio_obj : rasterio.io.DatasetReader
            Rasterio dataset object for the image.

        Methods
        -------
        __init__(self, img_data, crs, affine_transformation, transform, rasterio_obj)
            Constructs all the necessary attributes for the Image object.
        to_image(self, bands)
            Converts specified bands of the image into an RGB image.
    """
    def __init__(self, img_data, crs, affine_transformation, transform, rasterio_obj):
        """Initialize the instance."""
        self.img_data = img_data
        self.crs = crs
        self.affine_transformation = affine_transformation
        self.transform = transform
        self.rasterio_obj = rasterio_obj

        # 🔧 Back‑compat for older scripts
        self.ds = rasterio_obj          # alias
    def to_image(self, bands, p_min=2, p_max=98, stretch_type=None):
        """
        always applies a linear stretch
        :param bands: A list or tuple of three integers representing the indices of the bands to use for the RGB image.
        :param p_min: Minimum percentile for rescaling.
        :param p_max: Maximum percentile for rescaling.
        :param stretch_type: Type of contrast stretching to apply ("linear", "histogram_equalization", "clahe").
        :return: A rescaled and contrast-stretched RGB image.
        """
        if not isinstance(bands, (list, tuple)) or len(bands) != 3:
            raise ValueError("'bands' should be a list or tuple of exactly three elements")

        num_bands = self.img_data.shape[2]
        rgb_data = np.empty((self.img_data.shape[0], self.img_data.shape[1], 3), dtype=np.float32)

        for i, band in enumerate(bands):
            if band >= num_bands or band < 0:
                raise IndexError(f"Band index {band} out of range. Available bands indices: 0 to {num_bands - 1}.")
            rgb_data[:, :, i] = self.img_data[:, :, band]

        rgb_data_rescaled = rescale_to_8bit(rgb_data, min=p_min, max=p_max)

        if stretch_type == "histogram_equalization":
            rgb_data_rescaled = apply_histogram_equalization(rgb_data_rescaled)
        elif stretch_type == "clahe":
            rgb_data_rescaled = apply_clahe(rgb_data_rescaled)
        elif stretch_type != None:
            raise ValueError(f"Unknown stretch_type: {stretch_type}")

        return fromarray(rgb_data_rescaled.astype(np.uint8))


def open_geotiff(image_path, bands=None):
    """
    :param image_path: Path to the GeoTIFF file to be opened.
    :type image_path: str
    :param bands: List of band indices to be read from the GeoTIFF file. If None, all bands are read.
    :type bands: list of int, optional
    :return: An Image object containing the raster data, coordinate reference system, affine transformation matrix, and the original rasterio object.
    :rtype: Image
    """
    rasterio_obj = rasterio.open(image_path)

    crs = rasterio_obj.crs
    transform = rasterio_obj.transform
    affine_transformation = [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f]

    x_size = rasterio_obj.width
    y_size = rasterio_obj.height
    num_bands = rasterio_obj.count

    if bands is None:
        bands = list(range(1, num_bands + 1))

    data = np.empty((y_size, x_size, len(bands)), dtype=np.float32)

    for i, b in enumerate(bands):
        band_array = rasterio_obj.read(b)
        data[:, :, i] = band_array

    return Image(data, crs, affine_transformation, transform, rasterio_obj)


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
