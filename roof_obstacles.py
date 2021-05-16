# SYNTHESIS PROJECT.2021

import math
import numpy as np
import pandas as pd
from numpy.lib.function_base import corrcoef
import scipy.spatial
from plyfile import PlyData, PlyElement
from collections import UserString, deque

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
    #if A <= 0.02: return False
    tolerance = 0.1
    A1 = area_2d (p, p2, p3)
    A2 = area_2d (p1, p, p3)   
    A3 = area_2d (p1, p2, p)
    if((A >= (A1 + A2 + A3) - tolerance) and (A <= (A1 + A2 + A3) + tolerance)) or ((A1 + A2 + A3) >= (A - tolerance) and (A1 + A2 + A3) <= (A + tolerance)):
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

def write_obj(vertices, faces, fileout):
    vertices_out = []
    faces_out = []
    countf = 1
    for building in faces:
        for f in building:
           # print(vertices[f[0]][0])
            #vertices_list = (vertices[f[0]][0], vertices[f[0]][1], vertices[f[0]][2])
            vertices_out.append(vertices[f[0]])
            vertices_out.append(vertices[f[1]])
            vertices_out.append(vertices[f[2]])
            face = [countf, countf+1, countf+2]
            faces_out.append(face)
            countf += 3
    # write file
    with open(fileout, "w") as file:
        for v in vertices_out:
            file.write("v ")
            file.write(str(v[0]))
            file.write(" ")
            file.write(str(v[1]))
            file.write(" ")
            file.write(str(v[2]))
            file.write("\n")
        file.write("Oo\n")
        for f in faces_out:
            file.write("f ")
            file.write(str(f[0]))
            file.write(" ")
            file.write(str(f[1]))
            file.write(" ")
            file.write(str(f[2]))
            file.write("\n")
        file.close()

def write_ply (obstacle_pts, fileout):
    tuples = []
    for p in obstacle_pts:
            #for obstacle_point in obstacle_arr:
            point = tuple(p)
            tuples.append(point)
    a = np.array(tuples, dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
    print("Number of points in PLY file: ", len(obstacle_pts))
    # write PLY
    el = PlyElement.describe(a, 'vertex')
    with open(fileout, mode='wb') as f:
        PlyData([el], text=True).write(f)

def write_txt_cluster(dict_obstacles, obstacle_pts, fileout):
    with open(fileout, "w") as f:
        f.write("x y z obstacle \n")
        count = 0
        for key in dict_obstacles:
            for val in dict_obstacles[key]:
                f.write(str(obstacle_pts[val][0]))
                f.write(" ")
                f.write(str(obstacle_pts[val][1]))
                f.write(" ")
                f.write(str(obstacle_pts[val][2]))
                f.write(" ")
                f. write(key)
                f.write("\n")
            count += 1
        f.close()

def detect_obstacles(point_cloud, vertices, faces, output_file):
    print("Number of vertices: ", len(vertices))
    print("Number of faces: ", len(faces))

    # ALL FOLLOWING SHOULD BE FOR 1 BUILDING // TO ADAPT
    # Loop through triangles and select points above it (in a local subset)
    k = 0
    obstacle_pts = []
    subset_all = []

    projected_area_2d = 0.00
    area_3d = 0.00

    # ROOF EXPORT IN OBJ FOR VISUALISATION: Check that faces are only roofs and LOD2
    write_obj(vertices, faces, './fileout/roofs_out.obj')

    # OBSTACLE EXTRACTION
    set_point = set()
    for building in faces:
        obstacle_building = []
        #height_building = get_height_difference(faces)
        #print("Building's height is ", height_building)
        for triangle in building:
            subset = []
            assert (len(triangle) == 3)
            p1 = vertices[triangle[0]]
            p2 = vertices[triangle[1]]
            p3 = vertices[triangle[2]]
            projected_area_2d += area_2d(p1,p2,p3)
            area_3d += area_polygon_3d([p1,p2,p3])

            id_point = 0
            for point in point_cloud:
                if isInside(p1, p2, p3, point) and id_point not in set_point: 
                    # TODO add a condition that the point should not be under the plane
                    set_point.add(id_point)
                    subset.append(point)
                    subset_all.append(point)
                id_point += 1
                
            # Triangle is vertical?
            if len(subset) == 0: 
                continue
        
            # Distance points to surface: discard points closer than threshold to define
            else:
                threshold = 0.4
                for p in subset:
                    dist = shortest_distance(p, plane_equation(p1,p2,p3))
                    #print(dist)
                    if dist > threshold: 
                        obstacle_building.append(p)
                        obstacle_pts.append(p)
                        # TODO check that there are no duplicated points in obstacle_pts
                    else: continue
            k += 1
        print("Projected roof area in 2D: ", projected_area_2d)
        print("Roof area in 3D: ", area_3d)

    # visualise all points detected as obstacles
    #write_ply(obstacle_pts, './fileout/points_obtacle.ply')

    # Manual clustering
    kd = scipy.spatial.KDTree(obstacle_pts)
    tops_id = set()
    tops = []
    stack = deque()
    stacked_points_id = set()

    pid = 0
    for p in obstacle_pts:
        stacked_points_id.add(pid)
        stack.append(pid)
        top = pid
        while (len(stack) > 0):
            assert(len(stack) == 1)
            current_id = stack[-1]
            stack.pop()

            # We get the higher point in the radius search
            higher_id = current_id
            subset_id = kd.query_ball_point(obstacle_pts[current_id], r = 5)
            #print(len(subset_id))

            for id in subset_id:
                if obstacle_pts[id][2] > obstacle_pts[higher_id][2]:
                    higher_id = id
            
            if higher_id not in stacked_points_id:
                stack.append(higher_id)
                stacked_points_id.add(higher_id)
            else: 
                top = higher_id
        pid += 1
        # We add the top to the top of the obstacles
        if top not in tops_id:
            tops_id.add(top)
            tops.append(obstacle_pts[top])
        else: continue

    print("Number of clusters: ", len(tops))

    # KD tree for the tops
    kd_tops = scipy.spatial.KDTree(tops)

    # search for each point its closest top-point
    dict_obstacles = {}
    for id in range(len(tops)):
        dict_obstacles[str(id)] = list()
    id_ = 0
    for p in obstacle_pts:
        # Nearest neighbour search
        dist, id = kd_tops.query(p, k=1)
        dict_obstacles[str(id)].append(id_)
        id_ += 1

    # Check and visualise clusters
    write_txt_cluster(dict_obstacles, obstacle_pts, "./fileout/out_test.txt")

    # Create np-array of each cluster
    clusters_arr = []
    for key in dict_obstacles:
        array_point3d = []
        for val in dict_obstacles[key]:
            point_arr = [obstacle_pts[val][0], obstacle_pts[val][1], obstacle_pts[val][2]]
            array_point3d.append(point_arr)
        if len(array_point3d) > 3:                  # add condition of number of points in the cluster
            clusters_arr.append(array_point3d)
    #print(clusters_arr)

    # Obstacle points convex-hull
    
    hulls = []
    for cluster_ in clusters_arr:
        #print(cluster_)
        cluster = np.array(cluster_)
        try:
            hull = scipy.spatial.ConvexHull(cluster[:,:2])
            hull_arr = []
            for vertex_id in hull.vertices:
                hull_arr.append(cluster[vertex_id])
            hulls.append(hull_arr)
        except:
            continue
            #print ("Didnt work :(")

   # print(hulls)

    # Check for one hull
    hull_vertices = []
    hull_faces = []
    id_p = 0
    for hull in hulls:
        face_ = []
        for vertex in hull:
            v = [vertex[0], vertex[1],6]
            hull_vertices.append(v)
            face_.append(id_p)
            id_p += 1
        hull_faces.append(face_)
    #print(hull_faces)

    # write file
    with open("./fileout/hulls.obj", "w") as file:
        for v in hull_vertices:
            file.write("v ")
            file.write(str(v[0]))
            file.write(" ")
            file.write(str(v[1]))
            file.write(" ")
            file.write(str(v[2]))
            file.write("\n")
        file.write("Oo\n")
        for f in hull_faces:
            i = 0
            file.write("f ")
            while i < len(f):
                file.write(str(f[i]+1))
                file.write(" ")
                i += 1
            file.write("\n")
        file.close()


    # Area calculation of hulls
    for hull in hulls:
        a = area_polygon_3d(hull)
        #print (a)


    # Solar potential area computation

    # Store new attribute per building

    return


