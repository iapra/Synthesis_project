# SYNTHESIS PROJECT.2021

import math
import numpy as np
from numpy.lib.function_base import corrcoef
import scipy.spatial
from plyfile import PlyData, PlyElement
#from sklearn.cluster import KMeans

from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import Birch
from matplotlib import pyplot

#-- to speed up the nearest neighbour us a kd-tree
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
# kd = scipy.spatial.KDTree(list_pts)
# d, i = kd.query(p, k=1)

def area_2d(p1,p2,p3):
    return abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1])
                + p3[0] * (p1[1] - p2[1])) / 2.0)

def isInside(p1,p2,p3, p):
# A function to check whether point p lies inside the triangle p1,p2,p3
    A = area_2d (p1,p2,p3)
    if A <= 0.05: return False
    A1 = area_2d (p, p2, p3)
    A2 = area_2d (p1, p, p3)   
    A3 = area_2d (p1, p2, p)
    if(A == A1 + A2 + A3):
        return True
    else:
        return False

#unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x = np.linalg.det([[1,a[1],a[2]],
                        [1,b[1],b[2]],
                        [1,c[1],c[2]]])
    y = np.linalg.det([[a[0],1,a[2]],
                        [b[0],1,b[2]],
                        [c[0],1,c[2]]])
    z = np.linalg.det([[a[0],a[1],1],
                        [b[0],b[1],1],
                        [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)

#area of polygon poly (embedded in 3D)
def area_polygon_3d(poly):
    if len(poly) < 3: # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)

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

def get_height_difference(vertices):
    z= []
    for vertex in vertices:
        z.append(vertex[2])
    height_diff = max(z) - min(z)
    return height_diff


def detect_obstacles(point_cloud, vertices, faces, output_file):
    print("Number of vertices: ", len(vertices))
    print("Number of faces: ", len(faces))

    # ALL FOLLOWING SHOULD BE FOR 1 BUILDING // TO ADAPT
    # Loop through triangles and select points above it (in a local subset)
    k = 0
    #all_subsets = []
    obstacle_pts = []
    tuples = []
    height_building = get_height_difference(vertices)
    print("Building's height is ", height_building)

    projected_area_2d = 0.00
    area_3d = 0.00

    for triangle in faces:
        subset = []
        too_high = []
        obstacle_triangle = [] # this helps us to find if the triangle is roof or not

        assert (len(triangle) == 3)
        p1 = vertices[triangle[0]-1]
        p2 = vertices[triangle[1]-1]
        p3 = vertices[triangle[2]-1]

        for point in point_cloud:
            if isInside(p1, p2, p3, point): 
                subset.append(point)
            
        # Triangle is vertical
        if len(subset) == 0: 
            #all_subsets.append(obstacle_pts)
            continue
        
        # Distance points to surface: discard points closer than threshold to define
        else:
            threshold = 0.5
            for p in subset:
                dist = shortest_distance(p, plane_equation(p1,p2,p3))
                if dist > threshold:
                    obstacle_triangle.append(p)
                if dist > (height_building-2):
                    # Triangle is probably the ground
                    # too_high.append(p)
                    # break
                    too_high.append(p)
                    if len(too_high) > 0:
                        obstacle_triangle.clear()
                        break
                else: continue

        for p in obstacle_triangle:
            obstacle_pts.append(p)

        #all_subsets.append(obstacle_pts)

        if len(obstacle_triangle) != 0 and len(too_high) == 0:
            projected_area_2d += area_2d(p1,p2,p3)
            area_3d += area_polygon_3d([p1,p2,p3])
            #print ("Triangle id ", k, "--- Number of point in = ", len(subset), " - Number of obstacle points = ", len(obstacle_pts))
        k += 1
    print("Projected roof area in 2D: ", projected_area_2d)
    print("Roof area in 3D: ", area_3d)

    # turn the subset into a np.array for the write part
    #     for i in subset:
    #         x = tuple(i)
    #         tuples.append(x)
    #     a = np.array(tuples, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    #     break

    # visualise all points detected as obstacles
    for p in obstacle_pts:
        #for obstacle_point in obstacle_arr:
        point = tuple(p)
        tuples.append(point)
    a = np.array(tuples, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    print("Number of ostacle points found: ", len(obstacle_pts))

    # write PLY
    el = PlyElement.describe(a, 'vertex')

    with open('points_test.ply', mode='wb') as f:
        PlyData([el], text=True).write(f)

    # Clustering
    kd_pc = scipy.spatial.KDTree(obstacle_pts)
    # data = np.vstack(obstacle_pts).T
    # clustering = KMeans(n_clusters=5, random_state=170)
    # km = clustering.fit_predict(data)
    # for ele in km:
    #     print (ele)

    # define dataset
    nb_cluster = round(len(obstacle_pts)/5)
    X, _ = make_classification(n_samples=len(obstacle_pts), 
            n_features=nb_cluster, n_informative=nb_cluster, 
            n_redundant=0, n_clusters_per_class=1)
    # define the model
    model = Birch(threshold=0.01, n_clusters=nb_cluster)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(X[:,0], X[:,1], c=yhat)
    # show the plot
    pyplot.show()

    # Obstacle points convex-hull

    # Area calculation

    # Solar potential area computation

    # Store new attribute per building

    return


