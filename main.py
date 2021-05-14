import sys
import json
import os
from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd
import pywavefront
import json
# -- *all* code goes into 'roof_obstacles'
import roof_obstacles

input_ply = "./data/one_building.ply"
input_obj = "./data/3d_one_building.obj"
output_file = "./data/out.json"
input_json = "./data/3dbag_v21031_7425c21b_5910.json"

# Structures to get the input elements:
vertices = []
faces = []
json_boundaries = []
json_vertices = []

def yield_file(in_file):
    f = open(in_file)
    buf = f.read()
    f.close()
    for b in buf.split('\n'):
        if b.startswith('v '):
            yield ['v', [float(x) for x in b.split(" ")[1:]]]
        elif b.startswith('f '):
            triangles = b.split(' ')[1:]
            yield ['f', [int(t.split("/")[0]) for t in triangles]]
        else:
            yield ['', ""]

def read_obj(in_file):
    for k, v in yield_file(in_file):
        if k == 'v':
            vertices.append(v)
        elif k == 'f':
            faces.append(v)

    if not len(faces) or not len(vertices):
        return None

def read_json(in_file):
    f = open(in_file)
    data = json.load(f)   # returns JSON object as a dictionary
    f.close()

    for i in data["CityObjects"]:
        each_boundary = []
        for lists in data["CityObjects"][i]["geometry"][2]["boundaries"][0]:
            each_boundary.append(lists[0])
        json_boundaries.append(each_boundary)

    for coord in data["vertices"]:
        json_vertices.append(coord)

    # Add transform to json_vertices
    for vtx in json_vertices:
        vtx[0] = (vtx[0] * data["transform"]["scale"][0]) + data["transform"]["translate"][0]
        vtx[1] = (vtx[1] * data["transform"]["scale"][1]) + data["transform"]["translate"][1]
        vtx[2] = (vtx[2] * data["transform"]["scale"][2]) + data["transform"]["translate"][2]

    # print(json_vertices)
    # # print(data["vertices"])
    # print(len(json_boundaries))
    # return data


def main():
    # -- READ PLY: store the input 3D points in np array
    plydata = PlyData.read(input_ply)                       # read file
    data = plydata.elements[0].data                         # read data
    data_pd = pd.DataFrame(data)                            # Convert to DataFrame, because DataFrame can parse structured data

    # THIS KEEPS ALL PROPERTIES (nb of returns, etc)
    # data_np = np.zeros(data_pd.shape, dtype=np.float)   # Initialize the array of stored data
    # property_names = data[0].dtype.names                # read the name of the property
    # for i, name in enumerate(property_names):           # Read data according to property, so as to ensure that the data read is the same data type.
    #     data_np[:, i] = data_pd[name]

    # THIS KEEPS ONLY x,y,z
    point_cloud = np.zeros((data_pd.shape[0], 3), dtype=float)  # Initialize the array of stored data
    property_names = data[0].dtype.names
    for i, name in enumerate(property_names):
        if (i > 2): continue
        point_cloud[:, i] = data_pd[name]


    # -- READ OBJ: store the input in arrays
    read_obj(input_obj)
    print(read_json(input_json))

    # -- READ json:
    # with open (input_json) as obj:
    #     data = json.loads(obj)

    # -- detect obstacles
    roof_obstacles.detect_obstacles(point_cloud, vertices, faces, output_file)


if __name__ == '__main__':
    main()
