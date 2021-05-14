# SYNTHESIS PROJECT.2021

import math
import numpy as np
import scipy.spatial

#-- to speed up the nearest neighbour us a kd-tree
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
# kd = scipy.spatial.KDTree(list_pts)
# d, i = kd.query(p, k=1)

def area(p1,p2,p3):
    return abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1])
                + p3[0] * (p1[1] - p2[1])) / 2.0)

def isInside(p1,p2,p3, p):
# A function to check whether point P(x, y)
# lies inside the triangle formed by
# A(x1, y1), B(x2, y2) and C(x3, y3)
    A = area (p1,p2,p3)
    if A == 0: return False
    A1 = area (p, p2, p3)
    A2 = area (p1, p, p3)   
    A3 = area (p1, p2, p)
    if(A == A1 + A2 + A3):
        return True
    else:
        return False


def detect_obstacles(point_cloud, vertices, faces, output_file):
    print(len(vertices))
    print(len(faces))

    kd_pc = scipy.spatial.KDTree(point_cloud)
    # Loop through triangles and select points above it (in a local subset)
    k = 0
    for triangle in faces:
        k += 1
        assert (len(triangle) == 3)

        subset = []
        for point in point_cloud:
            # Points' subset
            p1 = vertices[triangle[0]-1]
            p2 = vertices[triangle[1]-1]
            p3 = vertices[triangle[2]-1]
            if isInside(p1, p2, p3, point): 
                subset.append(point)
        
        # Triangle is vertical
        if len(subset) == 0: continue
        else:
            print ("For triangle number ", k, " Number of point in = ", len(subset))
            # Distance points to surface: discard points closer than .. threshold to define
            for p in subset:
                # COMPUTE DISTANCE HERE
                continue
    
   

    # Obstacle points convex-hull

    # Projection on mesh

    # Solar potential area computation

    # Store new attribute per triangle


    return


