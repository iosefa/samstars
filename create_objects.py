import geopandas as gpd
from obia.stats import create_objects        # parallel version
from utils.utils import open_geotiff         # lazy reader

def main():

    img = open_geotiff("img_cog.tif")
    print("Raster size:", img.ds.width, "×", img.ds.height)
    print("Band indexes:", img.band_indexes)


    segments = gpd.read_file("crowns.gpkg")


    gdf = create_objects(
        segments,
        img,
        ept="ept/ept.json",
        ept_srs="EPSG:6635",
        voxel_resolution=(500, 500, 2),
        textural_bands=[0],
        calculate_structural=True,
        calculate_radiometric=True,
        backend="threads",
        max_workers=8
    )

    gdf.to_file("objects.gpkg")

if __name__ == "__main__":      # safe on macOS, Windows, WSL, Linux
    main()