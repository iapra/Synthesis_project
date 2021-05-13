import sys
import math
import csv
import random
import json 
import time

#-- *all* code goes into 'roof_obstacles'
import roof_obstacles

input_ply = "../data/one_building.ply"
input_obj = "../data/3d_one_building.obj"
output_file = "../data/out.json"

def main():
    #-- READ PLY: store the input 3D points in list
    list_pts_3d = []
    # with open("input_ply") as csvfile:
    #     r = csv.reader(csvfile, delimiter=' ')
    #     header = next(r)
    #     for line in r:
    #         p = list(map(float, line)) #-- convert each str to a float
    #         assert(len(p) == 3)
    #         list_pts_3d.append(p)



    #-- READ OBJ: store the input in ?





    #-- detect obstacles
    roof_obstacles.detect_obstacles(output_file)     

if __name__ == '__main__':
    main()
