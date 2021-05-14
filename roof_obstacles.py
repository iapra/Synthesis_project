# SYNTHESIS PROJECT.2021

import math
import numpy as np
import scipy.spatial
from plyfile import PlyData, PlyElement

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
    all_subsets = []
    tuples = []
    for triangle in faces:
        assert (len(triangle) == 3)

        #inside = 0 
        outside = 0
        subset = []
        # Distance points to surface: discard points closer than .. threshold to define
        for point in point_cloud:
            # Points' subset
            # print(point)
            p1 = vertices[triangle[0]-1]
            p2 = vertices[triangle[1]-1]
            p3 = vertices[triangle[2]-1]
            if isInside(p1, p2, p3, point): 
                subset.append(point)
                # inside += 1
            else:
                outside += 1

        # if len(subset) != 0:
        #     all_subsets.append(subset)
        all_subsets.append(subset)

        k += 1
        print("For triangle number ", k, " Number of point in = ", len(subset), " Number of points out = ", outside)

    # print(all_subsets)
    # turn the subset into a np.array for the write part
    #     for i in subset:
    #         x = tuple(i)
    #         tuples.append(x)
    #     a = np.array(tuples, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    #     break

    # change the index to switch between subsets
    for i in all_subsets[0]:
        x = tuple(i)
        tuples.append(x)
    a = np.array(tuples, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    # write PLY
    el = PlyElement.describe(a, 'vertex')

    with open('points_test.ply', mode='wb') as f:
        PlyData([el], text=True).write(f)

    # Obstacle points convex-hull

    # Projection on mesh

    # Solar potential area computation

    # Store new attribute per triangle

    return


