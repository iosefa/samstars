import geopandas as gpd
from obia.stats import create_objects
from utils.utils import open_geotiff

def main():

    img = open_geotiff("img.tif")
    print("Raster size:", img.ds.width, "×", img.ds.height)

    segments = gpd.read_file("crowns.gpkg")

    # subset = segments.sample(n=500, random_state=42)  # drop random_state for true randomness
    # segments = subset.reset_index(drop=True)

    if "segment_id" not in segments.columns:
        segments["segment_id"] = range(1, len(segments) + 1)

    gdf = create_objects(
        segments,
        img,
        ept="ept/ept.json",
        ept_srs="EPSG:32605",
        voxel_resolution=(500, 500, 1),
        textural_bands=[0],
        calculate_structural=True,
        calculate_radiometric=True
    )

    gdf.to_file("objects.gpkg")


if __name__ == "__main__":
    main()
