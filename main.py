import json
import os
from numpy.core.defchararray import array
from numpy.lib.function_base import append
from plyfile import PlyData
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

# -- *all* code goes into 'roof_obstacles'
import roof_obstacles

input_ply = "./data/extract1_n_rad7.ply"
input_json = "./data/extract1.json"
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

def main():
    # -- READ PLY: store the input 3D points in np array
    point_cloud = read_ply(input_ply)

    # -- READ JSON: store the input in arrays
    read_json(input_json)

    # -- Detect obstacles
    geojson_part1 = roof_obstacles.detect_obstacles(point_cloud, json_vertices, json_boundaries, output_files, input_json)
    
    # -- Image classification

    # -- Merge part 1 and part 2
    
    # -- CityJSON output
    #write_json(input_json, output_file, dict_buildings)  



if __name__ == '__main__':
    main()
