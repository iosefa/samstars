#!/usr/bin/env python3
import json
import os

import geopandas as gpd
import pdal
from pyforestscan.handlers import read_lidar


def write_las(arrays, output_file, in_srs=None, out_srs=None, compress=True):
    """
    Writes point cloud data to a LAS or LAZ file.

    :param arrays: The point cloud data arrays.
    :param output_file: The path of the output file.
    :param srs: Optional; Spatial Reference System to reproject the data.
    :param compress: Optional; Boolean flag to compress the output. Defaults to True.
    :raises ValueError: If 'compress' is True and output file extension is not .laz.
    :raises ValueError: If 'compress' is False and output file extension is not .las.

    :return: None
    """
    output_extension = os.path.splitext(output_file)[1].lower()

    if compress:
        if output_extension != '.laz':
            raise ValueError("If 'compress' is True, output file must have a .laz extension.")
        output_format = "writers.las"
    else:
        if output_extension != '.las':
            raise ValueError("If 'compress' is False, output file must have a .las extension.")
        output_format = "writers.las"

    pipeline_steps = []

    if out_srs:
        pipeline_steps.append({
            "type": "filters.reprojection",
            "in_srs": in_srs,
            "out_srs": out_srs
        })

    pipeline_steps.append({
        "type": output_format,
        "filename": output_file,
        "minor_version": "4",
        "extra_dims": "all"
    })

    pipeline_def = {
        "pipeline": pipeline_steps
    }

    pipeline_json = json.dumps(pipeline_def)

    if not isinstance(arrays, list):
        arrays = [arrays]

    pipeline = pdal.Pipeline(pipeline_json, arrays=arrays)
    pipeline.execute()


# -------------------
# CONFIG / PATHS
# -------------------
ept = "/Users/iosefa/repos/obia/docs/example_data/full/ept/ept.json"
ept_srs = "EPSG:32605"
final_srs = "EPSG:6635"  # reproject outputs back to this CRS

# Inputs
gdf_path = "data/set_c/set_c_extent.gpkg"

# Outputs
las_out = "data/set_c/set_c.las"

def build_bounds_str(gdf: gpd.GeoDataFrame) -> str:
    minx, miny, maxx, maxy = gdf.total_bounds
    return f"([{minx},{maxx}], [{miny},{maxy}])"

gdf = gpd.read_file(gdf_path)
bounds_str = build_bounds_str(gdf)
poly_union = gdf.union_all()

pc = read_lidar(
    ept,
    ept_srs,
    bounds_str,
    crop_poly=True,
    poly=poly_union.wkt,
)[0]

write_las(pc, las_out, in_srs=ept_srs, out_srs=final_srs, compress=False)
