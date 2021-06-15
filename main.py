import json
import os
from numpy.core.defchararray import array
from numpy.lib.function_base import append
from plyfile import PlyData
import numpy as np
import pandas as pd
import geopandas as gpd
from os import listdir
from os.path import isfile, join
import csv
import time

# -- *all* code goes into 'roof_obstacles'
import roof_obstacles
import image_classification
import combine_results
#from deeplearning.predict.main import solar_panel_test

input_ply = "./data/brink/50_bdg_rad5.ply"
input_json = "./data/brink/50_bdg.json"
output_files = "./fileout/out"
#input_pcs = "./data/pointclouds/"

# -- Structures to get the input elements:
vertices = []
faces = []
json_boundaries = []
json_vertices = []

def read_json(in_file):
    f = open(in_file)
    # loading json returns JSON object as a dictionary
    data = json.load(f)   
    f.close()

    for i in data["CityObjects"]:
        each_boundary = []
        x = 0
        for lists in data["CityObjects"][i]["geometry"][2]["boundaries"][0]:
            # print(lists[0])
            if data["CityObjects"][i]["geometry"][2]["semantics"]["values"][0][x] == 1: # if it is a roof surface, it is added to the list
                each_boundary.append(lists[0])
            x += 1
        json_boundaries.append(each_boundary)

    for coord in data["vertices"]:
        json_vertices.append(coord)

    # Add transform to json_vertices
    for vtx in json_vertices:
        vtx[0] = (vtx[0] * data["transform"]["scale"][0]) + data["transform"]["translate"][0]
        vtx[1] = (vtx[1] * data["transform"]["scale"][1]) + data["transform"]["translate"][1]
        vtx[2] = (vtx[2] * data["transform"]["scale"][2]) + data["transform"]["translate"][2]

def read_ply(input_ply):
    plydata = PlyData.read(input_ply)                       # read file
    data = plydata.elements[0].data                         # read data
    data_pd = pd.DataFrame(data)                            # Convert to DataFrame, because DataFrame can parse structured data

    # This keeps all properties (x, y, z, nx, ny, nz)
    point_cloud = np.zeros(data_pd.shape, dtype=np.float64)     # Initialize the array of stored data
    property_names = data[0].dtype.names                        # read the name of the property
    for i, name in enumerate(property_names):                   # Read data according to property, so as to ensure that the data read is the same data type.
        point_cloud[:, i] = data_pd[name]
    return point_cloud

def npy_to_one_ply(all_pc):
    all_data = []
    all_files = os.listdir(all_pc)
    for pc in all_files:
        data = np.load(all_pc + pc)
        for each in data:
            all_data.append(each)
    roof_obstacles.write_ply(all_data, './fileout/ply_pc/all_bdg.ply')

def npy_to_ply(all_pc):
    all_files = os.listdir(all_pc)
    for pc in all_files:
        data = np.load(all_pc + pc)
        roof_obstacles.write_ply(data, './fileout/ply_pc/' + str(pc[:16]) + '.ply')

def write_json(in_file, outfile, csv_table, dict_buildings):
    inp_file = open(in_file, "r")
    json_obj = json.load(inp_file)
    inp_file.close()

    with open(csv_table, newline='') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        next(r, None)                                   # skip the headers
        for row in r:
            identificatie = row[0]
            city_obj, roof_area = dict_buildings[identificatie]
            obstacle_area = row[1]
            available_area = (float(roof_area) - float(obstacle_area))
            has_solar_panel = "False"                   # -- Replace with row[2]
            json_obj["CityObjects"][city_obj]["attributes"]["obstacle_area (m^2)"] = obstacle_area
            json_obj["CityObjects"][city_obj]["attributes"]["available_area (m^2)"] = available_area
            json_obj["CityObjects"][city_obj]["attributes"]["has_solar_panel"] = has_solar_panel

    inp_file = open(outfile, "w")
    json.dump(json_obj, inp_file)
    inp_file.close()

def main():
    start = time.perf_counter() 
    # -- READ PLY: store the input 3D points in np array
    point_cloud = read_ply(input_ply)

    # -- READ JSON: store the input in arrays
    read_json(input_json)

    # -- Detect obstacles
    print("\ninit geometry based detection")
    geojson_part1, dict_buildings = roof_obstacles.detect_obstacles(point_cloud, json_vertices, json_boundaries, output_files, input_json)
    
    # -- Image classification
    print("\ninit image classification")
    image_classification.main()
    # -- Merge part 1 and part 2
    print("\ninit combine geometry and image class")
    combine_results.main(geojson_part1)
        
    # -- Solar panel Detection
    #print("init solar panel detection")
    #roof = pd.read_csv('roof_semantics.csv')
    #solar = []
    #for index, row in roof.iterrows():
    #    solar_bool = solar_panel_test(row.identificatie)
    #    solar.append(solar_bool)
    #roof['has_solar_panels']= solar
    #df.to_csv('roof_semantics.csv',index=False)

    # -- CityJSON output
    csv = 'roof_semantics.csv'
    write_json(input_json, output_files + '.json', csv, dict_buildings)  

    # -- Time for computation
    end = time.perf_counter() 
    print("Time elapsed during the calculation:", (end - start)/60, " min")

if __name__ == '__main__':
    main()
