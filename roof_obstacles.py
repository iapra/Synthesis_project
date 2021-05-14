# SYNTHESIS PROJECT.2021

import math
import numpy as np
from numpy.lib.function_base import corrcoef
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
# A function to check whether point p lies inside the triangle p1,p2,p3
    A = area (p1,p2,p3)
    if A == 0: return False
    A1 = area (p, p2, p3)
    A2 = area (p1, p, p3)   
    A3 = area (p1, p2, p)
    if(A == A1 + A2 + A3):
        return True
    else:
        return False

def plane_equation (v1, v2, v3):
# Function to find plane equation
    p1 = np.array([v1[0], v1[1], v1[2]])
    p2 = np.array([v2[0], v2[1], v2[2]])
    p3 = np.array([v3[0], v3[1], v3[2]])
    v1 = p3 - p1
    v2 = p2 - p1
    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp
    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)
    equation_coef = [a, b, c, d]
    return equation_coef

def shortest_distance(p, equation_coef):
# Function to find distance from point p to plane
    a, b, c, d = equation_coef[0], equation_coef[1], equation_coef[2], -equation_coef[3]
    x, y, z = p[0], p[1], p[2]
    numerator = abs((a * x + b * y + c * z + d))
    denum = (math.sqrt(a * a + b * b + c * c))
    #print(numerator/denum)
    return numerator/denum


def detect_obstacles(point_cloud, vertices, faces, output_file):
    print(len(vertices))
    print(len(faces))

    kd_pc = scipy.spatial.KDTree(point_cloud)
    # Loop through triangles and select points above it (in a local subset)
    k = 1
    all_subsets = []
    tuples = []

    for triangle in faces:
        assert (len(triangle) == 3)
        p1 = vertices[triangle[0]-1]
        p2 = vertices[triangle[1]-1]
        p3 = vertices[triangle[2]-1]

        subset = []
        obstacle_pts = []
        too_high = []

        for point in point_cloud:
            if isInside(p1, p2, p3, point): 
                subset.append(point)
            
        # Triangle is vertical
        if len(subset) == 0: continue

        else:
        # Distance points to surface: discard points closer than .. threshold to define
            for p in subset:
                threshold = 0.1
                dist = shortest_distance(p, plane_equation(p1,p2,p3))
                if dist > threshold:
                    obstacle_pts.append(p)
                if dist > 4:
                    too_high.append(p)
                    if len(too_high) > 2:
                        break
                else: continue

        all_subsets.append(subset)
        if len(obstacle_pts) == 0:
            continue
        if len(too_high) > 2:
            print ("Triangle ", k, "--- This triangle is probably the ground !")
        else:
            print ("Triangle ", k, "--- Number of point in = ", len(subset), " - Number of obstacle points = ", len(obstacle_pts))
        
        k += 1

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


