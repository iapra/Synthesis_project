import sys
import json 
import os
from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd

#-- *all* code goes into 'roof_obstacles'
import roof_obstacles

input_ply = "./data/one_building.ply"
input_obj = "./data/3d_one_building.obj"
#input_json = "../data/3d_one_building.json"
output_file = "./data/out.json"

def main():
    #-- READ PLY: store the input 3D points in np array
    plydata = PlyData.read(input_ply)                   # read file
    data = plydata.elements[0].data                     # read data
    data_pd = pd.DataFrame(data)                        # Convert to DataFrame, because DataFrame can parse structured data

    # THIS KEEPS ALL PROPERTIES (nb of returns, etc)
    # data_np = np.zeros(data_pd.shape, dtype=np.float)   # Initialize the array of stored data
    # print (data_pd.shape)
    # property_names = data[0].dtype.names                # read the name of the property
    # for i, name in enumerate(property_names):           # Read data according to property, so as to ensure that the data read is the same data type.
    #     data_np[:, i] = data_pd[name]
    # print(data_np)

    #THIS KEEPS ONLY x,y,z
    data_np = np.zeros((data_pd.shape[0], 3), dtype=np.float)   # Initialize the array of stored data
    print (data_np.shape)
    property_names = data[0].dtype.names
    for i, name in enumerate(property_names): 
        if (i > 2): continue
        data_np[:, i] = data_pd[name]
    print(data_np)


    # with open("input_ply") as csvfile:
    #     r = csv.reader(csvfile, delimiter=' ')
    #     header = next(r)
    #     for line in r:
    #         p = list(map(float, line)) #-- convert each str to a float
    #         assert(len(p) == 3)
    #         list_pts_3d.append(p)



    #-- READ OBJ: store the input in ?


    #-- READ json:
    # with open (input_json) as obj:
    #     data = json.loads(obj)


    #-- detect obstacles
    roof_obstacles.detect_obstacles(output_file)     

if __name__ == '__main__':
    main()
