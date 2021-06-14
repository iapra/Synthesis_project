##Import images and raster from bag, 3dbag and aerial image
import sys
import json
import geojson
import geopandas as gpd
import requests
from owslib.wms import WebMapService
import rasterio 
from rasterio import MemoryFile
from rasterio.plot import show
from rasterio.transform import from_origin
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
    	import_ids(input)
    elif jparams["type_request"] == "boundingbox":
    	bbox = [jparams["xmin"],jparams["ymin"],jparams["xmax"],jparams["xmax"]]
    	input  = {"boundingbox": bbox }
    	import_bbox(input)

def import_ids(input):
	feat = input["identificatie"]
	features = join_ids(feat)
	n_features = len(features)
	#wfs request
	url = 'https://geodata.nationaalgeoregister.nl/bag/wfs/v1_1?request=GetFeature&service=WFS&version=2.0.0&typeName=bag:pand&FEATUREID='+features+'&count='+ str(n_features)+'&outputFormat=application%2Fjson%3B%20subtype%3Dgeojson'
	r = requests.get(url)
	data = gpd.GeoDataFrame.from_features(geojson.loads(r.content))

	#get bounding boxes 
	bboxes = []
	for i, row in data.iterrows():
		bbox = row.geometry.bounds
		bbox_big = (bbox[0]-25, bbox[1]-25,bbox[2]+25,bbox[3]+25, row.identificatie)
		bboxes.append(bbox_big)

	#import wms

	import_wms(bboxes)

	#write features in geojson
	data.crs = 28992
	data.to_file("./data/classification_features/data.geojson", driver="GeoJSON")

	# get features for alignment
	get_alignment_features(bboxes)

def get_wms_details(url, version="1.3.0"):
    wms = WebMapService(url, version=version)
    return wms

def import_wms(bboxes):
	images =[]
	count = 0
	for building in bboxes:
		# Aerial RGB Image taken from PDOK
		w = (building[2]-building[0])/0.25
		l = (building[3]-building[1])/0.25
		rgb_url = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?"
		wms = get_wms_details(rgb_url)
		img = wms.getmap(layers=['Actueel_ortho25'],
			srs='EPSG:28992',
    		format='image/png',
    		#building bounding box
    		bbox=(building[0],building[1],building[2],building[3] ),
    		size=(w, l))
		transform = from_origin(building[0],building[3],0.25,0.25)
		with MemoryFile(img) as memfile:
			with memfile.open() as dataset:
				data_array = dataset.read()
			new_array = rasterio.open('./data/images/'+building[4]+'.tif', 
							'w', driver='GTiff',
							height = l, width = w,
                           	count=3, dtype=str(data_array.dtype),
                          	crs=28992,
            		        transform=transform)
			new_array.write(data_array)
			new_array.close()
		count = count + 1


def join_ids(feat):
	rel = []
	name = "pand.bag:"
	for obj in feat:
			s = name + obj
			rel.append(s)
	s = ",".join([item for item in rel])
	return s

def get_alignment_features(bbox_big):
	for building	in bbox_big:
		bbox = building[:-1]
		s = ",".join([str(item) for item in bbox])
		url = 'https://geodata.nationaalgeoregister.nl/bag/wfs/v1_1?request=GetFeature&service=WFS&version=2.0.0&typeName=bag:pand&bbox='+str(s)+'&outputFormat=application%2Fjson%3B%20subtype%3Dgeojson'
		r = requests.get(url)
		al_data = gpd.GeoDataFrame.from_features(geojson.loads(r.content), crs = 28992)		
		al_data.to_file("./data/alignment_features/"+ building[4] + ".json", driver="GeoJSON")

def import_bbox(input):
	return 0