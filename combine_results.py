import geopandas as gpd
import geojson
import rasterio
import numpy as np
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio import mask
from rasterio.merge import merge

def combine_rasters(pc):
	class_images = []
	transforms  = []
	for index, rows in pc.iterrows():
		with rasterio.open('./raster_results/'+rows.identificatie + '.tif') as src:
			pcr, out_transform = rasterio.mask.mask(src, [rows.geometry], invert = True)
			transforms.append(out_transform)
			class_images.append(pcr)
	for index, rows in pc.iterrows():
		new_array = rasterio.open('./final_results/'+rows.identificatie + '.tif',
                          'w', driver='GTiff',
                          height =class_images[index].shape[1], width = class_images[index].shape[2],
                          count=1, dtype = str(class_images[index][0][0].dtype),
                          crs=28992, nodata = 0,
                          transform=transforms[index])
		new_array.write(class_images[index][0],1)
		new_array.close()		


def main():
	pc = gpd.read_file('./data/classification_features/data_brink.geojson')
	combine_rasters(pc)
if __name__ == '__main__':
    main()