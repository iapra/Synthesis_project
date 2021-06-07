import json
import geopandas as gpd 
import rasterio    
from rasterio.mask import mask
from rasterio.transform import from_origin
import cv2 
import numpy as np
import csv

## Functions
def read_input(input):
	 #-- read the needed parameters from the file 'params.json' (must be in same folder)
    try:
        jparams = json.load(open(input))
    except:
        print("ERROR: something is wrong with the params.json file.")
        sys.exit()
    input = {}
    if jparams["type_request"] == "identificatie":
    	buildings = jparams["values"]
    	input =  {"identificatie": buildings}
    	return align_by_id(buildings)
    elif jparams["type_request"] == "boundingbox":
    	bbox = [jparams["xmin"],jparams["ymin"],jparams["xmax"],jparams["xmax"]]
    	input  = {"boundingbox": bbox }
    	return align_by_bbox(bbox)

def read_geojson_id(buildings):
    ids = []
    for id in buildings:
        data = gpd.read_file('./data/alignment_features/'+ id + '.json')
        ids.append(data)
    return ids

def get_big_building(buildings):
    ids = read_geojson_id(buildings)
    comlete_shape = []
    for i,id in enumerate(ids):
        index = id.sindex
        dissolved_id = id.dissolve()
        single = dissolved_id.explode()
        indexs = single.sindex
        sjoined_listings = gpd.sjoin(single, id, op="intersects")
        complete = sjoined_listings[["identificatie_right", "geometry"]]
        complete_id =  complete[complete['identificatie_right']==buildings[i]]
        complete_id = gpd.GeoDataFrame(complete_id)
        comlete_shape.append(complete_id)
    return comlete_shape    

def create_translations(cs, polygons,trans):
    x = np.linspace(-3,3,41)#41
    y = np.linspace(-3,3,41)
    for i in x:
        for j in y:
            pol = cs.translate(xoff = i, yoff=j)
            polygons.append(pol)
            trans.append((i,j))

def read_images(buildings):
    images = []
    for id in buildings:
        img = rasterio.open('./data/images/'+ id + '.tif')
        images.append(img)
    return images

def align_by_id(buildings):
    cs = get_big_building(buildings)
    images = read_images(buildings)
    translations= []
    identificaties = []
    for i, id in enumerate(cs):
        polygons = []
        trans = []
        img = images[i]
        identificaties.append(id.identificatie_right[0].values[0])
        create_translations(id, polygons, trans)
        bagb = id.buffer(10)
        bagi, _ = mask(img, bagb.geometry, invert=False)
        rgb = np.dstack((bagi[0], bagi[1], bagi[2]))
        w = rgb.shape[0]
        l = rgb.shape[1]
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        edges_high_thresh = cv2.Canny(gray, 90, 180)
        lines =[]
        for pol in polygons:
            line = pol.exterior
            lines.append(line)
        rasters = []
        transform = from_origin(img.bounds[0],img.bounds[3],0.25,0.25)
        for line in lines:
            raster_bag = rasterio.features.rasterize(line.geometry.buffer(0.125), out_shape=(w, l), transform=transform, fill = 2)
            rasters.append(raster_bag)
        edges_high_thresh = np.where(edges_high_thresh==255, 1, edges_high_thresh)
        rasters2 = []
        for raster in rasters:
            rasteri = np.where(raster==2, 0, raster)
            rasters2.append(rasteri)
        suma = []
        for raster in rasters2:
            tyre = edges_high_thresh  * raster
            s = tyre.sum()
            suma.append(s)
        maxi = 0
        pos = 0
        for h,j in enumerate(suma):
            if j >= maxi:
                maxi = j
                pos = h
        translations.append(trans[pos])
        trans = {}
        for index, iden in enumerate(identificaties):
            trans[iden] = translations[index]
        with open('./data/translations/translations.json', 'w') as json_file:
                json.dump(trans, json_file)
def main():
    read_input('params.json')

if __name__ == '__main__':
    main()