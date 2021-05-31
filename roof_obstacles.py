# SYNTHESIS PROJECT.2021

import math
from geojson.geometry import MultiPolygon
import numpy as np
from numpy.core.einsumfunc import _parse_possible_contraction
import pandas as pd
from numpy.lib.function_base import corrcoef
import scipy.spatial
from plyfile import PlyData, PlyElement
from collections import UserString, deque
from numpy import unique, where
from sklearn.datasets import make_classification
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns
# from sklearn.datasets.samples_generator import make_blobs
from geojson import Point, Polygon, Feature, FeatureCollection, dump
import json
import os
import sys
import shutil
# import alphashape
import geopandas as gpd

# -- to speed up the nearest neighbour us a kd-tree
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
# kd = scipy.spatial.KDTree(list_pts)
# d, i = kd.query(p, k=1)

def write_json(in_file, outfile, dict):
    # We first copy the input file
    if not os.path.isfile(in_file):
        print("Source file does not exist.")
        sys.exit(3)
    try:
        shutil.copy(in_file, outfile)
    except (IOError):
        print("Could not copy file.")
        sys.exit(3)
    # print(dict)

    # We edit the copy by adding an attribute
    with open(outfile, 'r+') as f:
        data = json.load(f)
        # d = {"HEY": 2000}
        # d = {data["CityObjects"]["1891794"]["attributes"]:2000}
        data["CityObjects"]["1891794"]["attributes"]["HEY "] = 2000
        # json.dump(d, data["CityObjects"]["1891794"]["attributes"])
        for i in data["CityObjects"]:  # i id the building id
            # f.write("HEY ")
            for attribute_field in data["CityObjects"][i]["attributes"]:
                print(attribute_field)  # key = attribute name
                print(data["CityObjects"][i]["attributes"][attribute_field])  # attribute value
    f.close()


def get_buildingID(json_in, index):
    with open(json_in, 'r') as f:
        data = json.load(f)
        building_id = list(data["CityObjects"].keys())[index]
        return (building_id)


def area_2d(p1, p2, p3):
    return abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1])
                + p3[0] * (p1[1] - p2[1])) / 2.0)


def isInside(p1, p2, p3, p):
    # A function to check whether point p lies inside the triangle p1,p2,p3
    A = area_2d(p1, p2, p3)
    # if A <= 0.02: return False
    tolerance = 0.1
    A1 = area_2d(p, p2, p3)
    A2 = area_2d(p1, p, p3)
    A3 = area_2d(p1, p2, p)
    if ((A >= (A1 + A2 + A3) - tolerance) and (A <= (A1 + A2 + A3) + tolerance)) or (
            (A1 + A2 + A3) >= (A - tolerance) and (A1 + A2 + A3) <= (A + tolerance)):
        return True
    else:
        return False


def isAbove(p1, p2, p3, p):
    det_above = np.linalg.det([[p1[0], p1[1], p1[2], 1],
                               [p2[0], p2[1], p2[2], 1],
                               [p3[0], p3[1], p3[2], 1],
                               [p[0], p[1], p[2], 1]])
    if det_above < 0:
        return True
    else:
        return False


# unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x = np.linalg.det([[1, a[1], a[2]],
                       [1, b[1], b[2]],
                       [1, c[1], c[2]]])
    y = np.linalg.det([[a[0], 1, a[2]],
                       [b[0], 1, b[2]],
                       [c[0], 1, c[2]]])
    z = np.linalg.det([[a[0], a[1], 1],
                       [b[0], b[1], 1],
                       [c[0], c[1], 1]])
    magnitude = (x ** 2 + y ** 2 + z ** 2) ** .5
    return (x / magnitude, y / magnitude, z / magnitude)


# area of polygon poly (embedded in 3D)
def area_polygon_3d(poly):
    if len(poly) < 3:  # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i + 1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result / 2)


def plane_equation(v1, v2, v3):
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
    # print(numerator/denum)
    return numerator / denum


def distance_2p(p1, p2):
    math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[1]) ** 2)


def get_height_difference(vertices):
    z = []
    for vertex in vertices:
        z.append(vertex[2])
    height_diff = max(z) - min(z)
    return height_diff


def normal_check(in_point):
    z1 = 0
    z2 = 0
    z3 = 1
    nx = in_point[3]
    ny = in_point[4]
    nz = in_point[5]

    # angle between two vectors in radians
    angle = np.arccos([((z1 * nx) + (z2 * ny) + (z3 * nz))
                       / (math.sqrt((z1 * z1) + (z2 * z2) + (z3 * z3)) * math.sqrt((nx * nx) + (ny * ny) + (nz * nz)))])

    if 50 < math.degrees(angle):  # this can be changed
        return False
    else:
        return True


def get_slope(p1, p2):
    # print(p1)
    # print(p2)
    length = math.sqrt((p2[0] - p1[[0]]) ** 2 + (p2[1] - p1[[1]]) ** 2 + (p2[2] - p1[[2]]) ** 2)
    length_2d = math.sqrt((p2[0] - p1[[0]]) ** 2 + (p2[1] - p1[[1]]) ** 2)
    try:
        return math.acos(length_2d / length)
    except:
        return 0


def write_obj(vertices, faces, fileout):
    vertices_out = []
    faces_out = []
    countf = 1
    for building in faces:
        for f in building:
            # print(vertices[f[0]][0])
            # vertices_list = (vertices[f[0]][0], vertices[f[0]][1], vertices[f[0]][2])
            vertices_out.append(vertices[f[0]])
            vertices_out.append(vertices[f[1]])
            vertices_out.append(vertices[f[2]])
            face = [countf, countf + 1, countf + 2]
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


def write_ply(obstacle_pts, fileout):
    tuples = []
    for p in obstacle_pts:
        new_p = [p[0], p[1], p[2]]
        # for obstacle_point in obstacle_arr:
        point = tuple(new_p)
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
                f.write(key)
                f.write("\n")
            count += 1
        f.close()

def dissolve_geojson(geojson):
    data = gpd.read_file(geojson)
    data_columns = data[['CityObject', 'geometry']]
    dissolved = data_columns.dissolve(by='CityObject')
    dissolved.to_file("./fileout/dissolved.geojson", driver='GeoJSON')


def detect_obstacles(point_cloud, vertices, faces, output_file, input_json):
    extract_nb = "extract1_n_rad7"  # variable to name properly the output files

    print("Number of vertices: ", len(vertices))
    print("Number of buildings: ", len(faces))

    dict_buildings = {}
    dict_obstacles_total = {}
    obstacle_pts_total = []
    tops_total = []
    hulls = []
    features = []
    hexagons = []

    # ROOF EXPORT IN OBJ FOR VISUALISATION: Check that faces are only roofs and LOD2
    write_obj(vertices, faces, './fileout/roofs_out.obj')

    # OBSTACLE EXTRACTION
    set_point = set()
    building_nb = 1
    projected_area_2d = 0.00
    area_3d = 0.00
    for building in faces:
        obstacle_pts = []
        rel_height = 0
        max_height = 0
        max_point = []
        min_point = []
        for triangle in building:
            subset = []
            assert (len(triangle) == 3)
            p1 = vertices[triangle[0]]
            p2 = vertices[triangle[1]]
            p3 = vertices[triangle[2]]
            projected_area_2d += area_2d(p1, p2, p3)
            area_3d += area_polygon_3d([p1, p2, p3])

            id_point = 0
            for point in point_cloud:
                if isInside(p1, p2, p3, point) and isAbove(p1, p2, p3, point) and normal_check(
                        point) and id_point not in set_point:
                    set_point.add(id_point)
                    subset.append(point)
                id_point += 1

            # Triangle is vertical?
            if len(subset) == 0:
                continue

            # Distance points to surface: discard points closer than threshold to define
            else:
                threshold = 0.4
                for p in subset:
                    dist = shortest_distance(p, plane_equation(p1, p2, p3))
                    if dist > threshold:
                        if len(max_point) == 0: max_point.append(p)
                        if len(min_point) == 0: min_point.append(p)
                        obstacle_pts.append(p)
                        obstacle_pts_total.append(p)
                        rel_height += dist
                        if p[2] > max_point[-1][2]:
                            max_height = dist
                            max_point.append(point)
                        if p[2] < min_point[-1][2]:
                            min_point.append(point)
                    else:
                        continue

        # Clean obstacle points
        obstacle_pts_final = []
        obstacle_pts_final2d = []
        try:
            kd_first = scipy.spatial.KDTree(obstacle_pts)
        except:
            continue

        for point_ in obstacle_pts:
            # Nearest neighbour search
            _dist, _id = kd_first.query(point_, k=2)
            if _dist[1] < 0.8:
                obstacle_pts_final.append(point_)
                obstacle_pts_final2d.append(point_[0:2])
            else:
                continue
        # print(len(obstacle_pts))
        # print(len(obstacle_pts_final))
        if len(obstacle_pts_final) < 2: continue
        if len(obstacle_pts_final2d) < 2: continue
        rel_height /= len(obstacle_pts)

        point_set = []
        for p in obstacle_pts_final2d:
            if tuple(p) not in point_set:
                point_set.append(tuple(p))


        for obs in point_set:
            x = obs[0]
            y = obs[1]
            one_hexagon = []
            # one_hexagon.append(obs)
            one_hexagon.append((x + 0.3, y))
            one_hexagon.append((x + 0.15, y + 0.3))
            one_hexagon.append((x - 0.15, y + 0.3))
            one_hexagon.append((x - 0.3, y))
            one_hexagon.append((x - 0.15, y - 0.3))
            one_hexagon.append((x + 0.15, y - 0.3))
            hexagons.append(one_hexagon)

    # Visualise convex-hulls -> to obj file
    hulls_vertices = []
    hulls_faces = []
    id_p = 0
    for hexs in hexagons:
        face_ = []
        for vertex in hexs:
            v = [vertex[0], vertex[1], 10]
            hulls_vertices.append(v)
            face_.append(id_p)
            id_p += 1
        hulls_faces.append(face_)

    # write file (hulls obj)
    with open("./fileout/hulls.obj", "w") as file:
        for v in hulls_vertices:
            file.write("v ")
            file.write(str(v[0]))
            file.write(" ")
            file.write(str(v[1]))
            file.write(" ")
            file.write(str(v[2]))
            file.write("\n")
        file.write("Oo\n")
        for f in hulls_faces:
            i = 0
            file.write("f ")
            while i < len(f):
                file.write(str(f[i] + 1))
                file.write(" ")
                i += 1
            file.write("\n")
        file.close()



        # Check and visualise clusters
        # write_txt_cluster(dict_obstacles, obstacle_pts_final, "./fileout/cluster.txt")

        # # K-means
        # nb_cluster = round(len(obstacle_pts_final2d)/7)
        # k_means = KMeans(init='random', n_clusters=3, n_init=10)
        # k_means.fit(obstacle_pts_final2d)
        # k_means_labels = k_means.labels_
        # #print(k_means_labels)
        #
        # dict_obstacles = {}
        # for id in range(len(k_means_labels)):
        #     dict_obstacles[str(k_means_labels[id])] = list()
        # count_ = 0
        # for p in obstacle_pts_final:
        #     dict_obstacles[str(k_means_labels[count_])].append(count_)
        #     count_ += 1

        # # Manual clustering
        # kd = scipy.spatial.KDTree(obstacle_pts_final)
        # tops_id = set()
        # tops = []
        # stack = deque()
        # stacked_points_id = set()
        #
        # pid = 0
        # for p in obstacle_pts_final:
        #     stacked_points_id.add(pid)
        #     stack.append(pid)
        #     top = pid
        #     while (len(stack) > 0):
        #         assert (len(stack) == 1)
        #         current_id = stack[-1]
        #         stack.pop()
        #
        #         # We get the higher point in the radius search
        #         higher_id = current_id
        #         subset_id = kd.query_ball_point(obstacle_pts_final[current_id], r=4)
        #
        #         for id in subset_id:
        #             if obstacle_pts_final[id][2] > obstacle_pts_final[higher_id][2]:
        #                 higher_id = id
        #
        #         if higher_id not in stacked_points_id:
        #             stack.append(higher_id)
        #             stacked_points_id.add(higher_id)
        #         else:
        #             top = higher_id
        #     pid += 1
        #     # We add the top to the top of the obstacles
        #     if top not in tops_id:
        #         tops_id.add(top)
        #         tops.append(obstacle_pts_final[top])
        #         tops_total.append(obstacle_pts_final[top])
        #     else:
        #         continue
        #
        # print("Number of obstacle detected for building ", building_nb, ": ", len(tops))
        #
        # # KD tree for the tops
        # kd_tops = scipy.spatial.KDTree(tops)
        #
        # # search for each point its closest top-point
        # dict_obstacles = {}
        # for id in range(len(tops)):
        #     dict_obstacles[str(id)] = list()
        #     id_total = int(id + len(dict_obstacles_total))
        #     dict_obstacles_total[str(id_total)] = list()
        # id_ = 0
        # for p in obstacle_pts_final:
        #     # Nearest neighbour search
        #     dist, id = kd_tops.query(p, k=1)
        #     dict_obstacles[str(id)].append(id_)
        #     id_ += 1
        #
        # # write_txt_cluster(dict_obstacles, obstacle_pts_final, "./fileout/cluster.txt")
        #
        # # Create np-array of each cluster
        # clusters_arr = []
        # for key in dict_obstacles:
        #     array_point3d = []
        #     for val in dict_obstacles[key]:
        #         point_arr = [obstacle_pts_final[val][0], obstacle_pts_final[val][1], obstacle_pts_final[val][2]]
        #         array_point3d.append(point_arr)
        #     if len(array_point3d) > 3:  # add condition of number of points in the cluster
        #         clusters_arr.append(array_point3d)
        #
        # # Obstacle points convex-hull
        # for cluster_ in clusters_arr:
        #     point_set = []
        #     cluster = np.array(cluster_)
        #     for p in cluster:
        #         if tuple(p[:2]) not in point_set:
        #             point_set.append(tuple(p[:2]))
        #
        #     # fig, ax = plt.subplots()
        #     # ax.scatter(*zip(*point_set))
        #     # plt.show()
        #
        #     try:
        #         alpha = alphashape.alphashape(point_set, 1.0)
        #         # print("done")
        #         hull_arr = []
        #         coords = list(alpha.exterior.coords)
        #         # print(len(coords))
        #         for vertex in coords:
        #             # print(vertex)
        #             hull_arr.append(vertex)
        #             hulls.append(hull_arr)
        #     except:
        #         continue

        # # Area calculation of hulls
        # obstacle_area = 0
        # for hull in hulls:
        #     a = area_polygon_3d(hull)
        #     obstacle_area += a

        # # Solar potential area computation
        # # For now we do an ugly cross proportional cross product!
        # obst_3d = (obstacle_area * area_3d) / projected_area_2d
        # new_attribute_area3d = area_3d - obst_3d
        # new_attribute_percent = (new_attribute_area3d / area_3d) *100
        # #print("Area free for solar panels (m**2) ! ", new_attribute_area3d, " (over ", area_3d, " in total)")
        # #print("In percentage, this is ", new_attribute_percent, " % ", "of the roof surface")
        #
        # dict_buildings[str(building_nb)] = new_attribute_percent /100
        # # print("Projected roof area in 2D: ", projected_area_2d)
        # # print("Roof area in 3D: ", area_3d)

        # Write files to geoJSON
        for hexs in hexagons:
            polygon = []
            for vertex in hexs:
                v = Point([vertex[0], vertex[1]])
                polygon.append(v)
            v_last = Point([hexs[0][0], hexs[0][1]])
            polygon.append(v_last)
            p = Polygon([polygon])
            building_id = get_buildingID(input_json, building_nb - 1)
            features.append(Feature(geometry=p, properties={"CityObject": str(building_id),
                                                            "Relative height": str(rel_height),
                                                            "Max obstacle height": str(max_height),
                                                            "Slope": str(get_slope(min_point[-1], max_point[-1]))
                                                            }))

        building_nb += 1

    crs = {
        "type": "name",
        "properties": {
            "name": "EPSG:28992"
        }
    }
    feature_collection = FeatureCollection(features, crs=crs)
    with open(str('./fileout/output_extract' + str(extract_nb) + '.geojson'), 'w') as geojson:

        dump(feature_collection, geojson)

    # # Visualise convex-hulls -> to obj file
    # hulls_vertices = []
    # hulls_faces = []
    # id_p = 0
    # for hull in hulls:
    #     face_ = []
    #     for vertex in hull:
    #         v = [vertex[0], vertex[1],10]
    #         hulls_vertices.append(v)
    #         face_.append(id_p)
    #         id_p += 1
    #     hulls_faces.append(face_)
    #
    # # write file (hulls obj)
    # with open("./fileout/hulls.obj", "w") as file:
    #     for v in hulls_vertices:
    #         file.write("v ")
    #         file.write(str(v[0]))
    #         file.write(" ")
    #         file.write(str(v[1]))
    #         file.write(" ")
    #         file.write(str(v[2]))
    #         file.write("\n")
    #     file.write("Oo\n")
    #     for f in hulls_faces:
    #         i = 0
    #         file.write("f ")
    #         while i < len(f):
    #             file.write(str(f[i]+1))
    #             file.write(" ")
    #             i += 1
    #         file.write("\n")
    #     file.close()

    # Store new attribute per building
    # write_txt_cluster(dict_obstacles_total, obstacle_pts_total, "./fileout/clusters.txt")
    write_ply(obstacle_pts_total, './fileout/points_obtacle.ply')
    # write_json(input_json, "./fileout/extract_out.json", dict_buildings)
    # print (dict_buildings)

    # print("area 3D = ", area_3d)
    # print("area 2D = ", projected_area_2d)

    return


