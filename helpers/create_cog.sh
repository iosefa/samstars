gdal_translate 	/mnt/d/20JAN07_M1BS_mosaic.tif tmp.tif \
    -co TILED=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 \
    -co COMPRESS=DEFLATE -co PREDICTOR=2 -co COPY_SRC_OVERVIEWS=NO -co BIGTIFF=YES

gdaladdo -r average tmp.tif 2 4 8 16 32

gdal_translate tmp.tif img_cog.tif -of GTiff \
    -co TILED=YES -co COPY_SRC_OVERVIEWS=YES -co BIGTIFF=YES