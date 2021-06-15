import json
import geopandas as gpd 
import rasterio
import csv
from shapely.affinity import translate
from rasterio.mask import mask
from sklearn.cluster import KMeans
import numpy as np
from rasterio.features import shapes
from rasterio.crs import CRS
from rasterio.transform import from_origin

#Read fuetures to be classified
def read_geopandas():
	path_pol = './data/classification_features/data.geojson'
	bag = gpd.read_file(path_pol)
	return bag

#Read images to perform classification
def read_images(bag):
	images = []
	path_img = './data/images/'
	for i,row in bag.iterrows():
		image_path = row.identificatie
		img = rasterio.open(path_img + image_path	+'.tif')
		images.append(img)
	return images

#Get bounds of an image	
def get_image_bounds(images):
	bounds = []
	for img in images:
		green = img.read(1)
		blue = img.read(2)
		red = img.read(3)
		rgb = np.dstack((green, blue, red))
		w = rgb.shape[0]
		l = rgb.shape[1]
		bounds.append((w,l))
	return bounds	

#Get bounds of a feature	
def get_bounds(bag):
	bounds = []
	for index, row in bag.iterrows():   
		geom = gpd.GeoSeries(row.geometry)
		width = int((geom.bounds.maxx[0] - geom.bounds.minx[0])/0.25)
		lenght = int((geom.bounds.maxy[0] - geom.bounds.miny[0])/0.25)
		bound = (width, lenght)
		bounds.append(bound)
	return bounds

def get_ids(bag):
	ids = []
	for index, row in bag.iterrows():
		ids.append(row.identificatie)
	return ids

#Trasnlate thefootprint according to the alignment.
def dtranslate(bag,mult):
	translations = []
	identificaties	= []
	with open('./data/translations/translations.json') as f:
		translations = json.load(f)
	tmp = []
	for index, poi in bag.iterrows():
		identificatie  = bag.loc[index,'identificatie']
		xoff = translations[identificatie][0] * mult 
		yoff = translations[identificatie][1] * mult 
		new_point = translate(bag.loc[index, 'geometry'], xoff=xoff, yoff=yoff)
		tmp.append({'geometry':new_point, 'identificatie':bag.loc[index, 'identificatie']})
	buildings = gpd.GeoDataFrame(tmp)		
	return buildings

#Clip the images with the buildings
def mask_buildings_images(buildings, images):
	lists= []
	for index, row in buildings.iterrows():
		geom = gpd.GeoSeries(row.geometry)
		out, _= mask(images[index], geom.geometry, invert=False)
		lists.append(out)
	return lists 

#Flatten bands of a nD numpy array
def flatten_bands(bands):
    n = []
    for im in bands:
        temp = im.flatten()
        n.append(temp)
    x = np.array(n)
    x= x.transpose()
    return x

#K-means clusterization
def clusterization(masks,bounds):
	clusters = []
	for out in masks:
		redn = (out[0])
		greenn = (out[1])
		bluen = (out[2])
		images = [redn, greenn, bluen]
		x = flatten_bands(images)
		cluster3 = KMeans(n_clusters=3, random_state=170).fit_predict(x)
		clusters.append(cluster3)
	images = []
	for index, cluster in enumerate(clusters):
		im = cluster.reshape(bounds[index][0],bounds[index][1])
		images.append(im)
	return images

#Write raster clusterization
def save_raster_clusterization(img,buildings,clusters):
	with open('./data/translations/translations.json') as f:
		translations = json.load(f)
	for r, image in enumerate(clusters):
		trans = translations[buildings["identificatie"][r]]
		transform = from_origin(img[r].transform[2]-trans[0],img[r].transform[5]-trans[1],0.25,0.25)
		new_array = rasterio.open('./raster_results/'+buildings["identificatie"][r]+'.tif', 
								'w', driver='GTiff',
								height = image.shape[0], width = image.shape[1],
	                           	count=1, dtype = str(image[0][0].dtype),
	                          	crs=28992, nodata = 0,
	            		        transform=transform)
		new_array.write(image,1)
		new_array.close()

def polygonize_raster(raster,images,ids, bag):
	for index,r in enumerate(raster):
		crs = CRS.from_epsg(28992)
		transform = from_origin(images[index].bounds[0],images[index].bounds[3],0.25,0.25)
		mask = None
		results = ({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(shapes(r, mask=mask, transform=transform,connectivity =4)))
		geoms = list(results)
		gpd_polygonized_raster  = gpd.GeoDataFrame.from_features(geoms)
		parth_rest = './results/'
		gpd_polygonized_raster['identificatie'] = ids[index]
		gpd_polygonized_raster1 = dtranslate(gpd_polygonized_raster, -1.0)
		gpd_polygonized_raster1['group'] = gpd_polygonized_raster['raster_val']
		gpd_polygonized_raster1.crs = 28992
		bag.crs = 28992
		gpd_polygonized_raster1['geometry'] = gpd_polygonized_raster1.geometry.buffer(0)
		gpd_polygonized_raster1 = gpd.clip(gpd_polygonized_raster1, bag.iloc[[index]])
		gpd_polygonized_raster1 = gpd_polygonized_raster1[gpd_polygonized_raster1.geometry.apply(lambda x : x.type!='GeometryCollection' )]
		gpd_polygonized_raster1.to_file(parth_rest+ ids[index] + '.geojson', driver='GeoJSON', schema=None)

#Asign obstacles
def assign_obstacles(clusters):
	clust = []
	for img in clusters:
		unique, counts = np.unique(img, return_counts=True)
		while (counts[0]< counts[1] or counts[1]< counts[2]):
				if counts[1] > counts[0]:
					img = np.where(img != 1 , img, 3) 
					img = np.where(img != 0 , img, 1)  
					img = np.where(img != 3 , img, 0)
				unique, counts = np.unique(img, return_counts=True)
				if counts[2] > counts[1]:
					img = np.where(img != 2 , img, 3) 
					img = np.where(img != 1 , img, 2)  
					img = np.where(img != 3 , img, 1) 
		clust.append(img) 
	return clust

def main():
    bag =  read_geopandas()
    images = read_images(bag)
    bounds = get_bounds(bag)
    img_bounds = get_image_bounds(images)	
    buildings_t = dtranslate(bag, 1.0)
    masks = mask_buildings_images(buildings_t, images)
    clusters = clusterization(masks,img_bounds)
    ids = get_ids(bag)
    clust = assign_obstacles(clusters)
    save_raster_clusterization(images,bag,clust)
    #polygonize_raster(clusters, images, ids, bag)
if __name__ == '__main__':
    main()
