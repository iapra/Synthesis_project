# SYNTHESIS PROJECT.2021

import math
import numpy as np
from statistics import stdev
import scipy.spatial
from plyfile import PlyData, PlyElement
from collections import defaultdict, deque
from numpy import unique, where
from sklearn.datasets import make_classification
from geojson import Point, Polygon, Feature, FeatureCollection, dump
import json
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import cascaded_union
import re

## --- Math functions

def area_2d(p1, p2, p3):
    return abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1])
                + p3[0] * (p1[1] - p2[1])) / 2.0)

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
    return numerator / denum

def get_normal(point):
    z1 = 0
    z2 = 0
    z3 = 1
    nx = point[3]
    ny = point[4]
    nz = point[5]
    # angle between two vectors in radians
    angle = np.arccos([((z1 * nx) + (z2 * ny) + (z3 * nz))
                       / (math.sqrt((z1 * z1) + (z2 * z2) + (z3 * z3)) 
                       * math.sqrt((nx * nx) + (ny * ny) + (nz * nz)))])
    return (math.degrees(angle))   

def extract_xyz(list):
    return [item[0:3] for item in list]

## --- Functions to retrieve info in input files

def get_cityJSON_ID(json_in, index):
    with open(json_in, 'r') as f:
        data = json.load(f)
        building_id = list(data["CityObjects"].keys())[index]
        return (building_id)

def get_buildingID(json_in, index):
    with open(json_in, 'r') as f:
        data = json.load(f)
        building_id = list(data["CityObjects"].keys())[index]
        identificatie = data["CityObjects"][building_id]["attributes"]["identificatie"]
        # We remove the no-digit part
        id = re.sub('\D', '', identificatie)
        return (id)


## --- Functions to write various files

def write_json(in_file, outfile, dict):
    inp_file = open(in_file, "r")
    json_obj = json.load(inp_file)
    inp_file.close()

    for key in dict:
        json_obj["CityObjects"][key]["attributes"]["available solar potential area (m^2)"] = dict[key]

    inp_file = open(outfile, "w")
    json.dump(json_obj, inp_file)
    inp_file.close()

def write_obj(vertices, faces, fileout):
    vertices_out = []
    faces_out = []
    countf = 1
    for building in faces:
        for f in building:
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
        point = tuple(new_p)
        tuples.append(point)
    a = np.array(tuples, dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
    print("Number of points in PLY file: ", len(obstacle_pts))
    # write PLY
    el = PlyElement.describe(a, 'vertex')
    with open(fileout, mode='wb') as f:
        PlyData([el], text=True).write(f)

def write_ply_final(point_cloud, dict_points, fileout):
    tuples = []
    pid = 0
    for p in point_cloud:
        new_p = [p[0], p[1], p[2], dict_points[pid]]
        point = tuple(new_p)
        tuples.append(point)
        pid += 1
    a = np.array(tuples, dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('distance_to_plane', 'f8')])
    print("Number of points in the final point cloud: ", len(point_cloud))
    # write PLY
    el = PlyElement.describe(a, 'vertex')
    with open(fileout, mode='wb') as f:
        PlyData([el], text=True).write(f)


# --- This can be deleted for final code
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

# --- This can be deleted for final code
def write_obstacles_to_obj(hulls):
    hulls_vertices = []
    hulls_faces = []
    id_p = 0
    for hull in hulls:
        face_ = []
        for vertex in hull:
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

def detect_obstacles(point_cloud, vertices, faces, output_files, input_json):

    print("Number of vertices: ", len(vertices))
    print("Number of buildings: ", len(faces))

    # 1 -- ROOF EXPORT IN OBJ FOR VISUALISATION: Check that faces are only roofs and LOD2
    #write_obj(vertices, faces, './fileout/roofs_out.obj')

    # 2 -- OBSTACLE POINTS EXTRACTION
    dict_buildings = {}
    dict_points = defaultdict(lambda:0)
    obstacle_pts_total = []
    hulls = []
    features = []
    set_point = set()
    set_point2 = set()
    max_heights = []
    building_nb = 1

    for building in faces:
        roof_area = 0.00
        hulls_polygons = []
        stack_first = deque()
        obstacle_pts = []
        point_toFace_dict = {}
        for triangle in building:
            assert (len(triangle) == 3)
            p1 = vertices[triangle[0]]
            p2 = vertices[triangle[1]]
            p3 = vertices[triangle[2]]
            roof_area += area_polygon_3d([p1, p2, p3])
        
            id_point = 0
            for point in point_cloud:
                xy = (point[0], point[1])
                point_toFace_dict[xy] = triangle
                if isInside(p1, p2, p3, point) and isAbove(p1, p2, p3, point) and get_normal(point) < 50 and id_point not in set_point:
                    set_point.add(id_point)
                    dict_points[id_point] = shortest_distance(point, plane_equation(p1, p2, p3))
                id_point += 1
                
        # Distance points to surface: discard points closer than threshold to define
        threshold = 0.4
        for pid, p in enumerate(point_cloud):
            if dict_points[pid] > threshold and pid not in set_point2:
                set_point2.add(pid)
                obstacle_pts.append(p)
                stack_first.append(p)
            else:
                continue
        
        # We add neighbours having similar normal's orientation
        kd_total = scipy.spatial.KDTree(point_cloud[:, 0:3])

        while (len(stack_first) > 0):
            current_p = stack_first[-1]
            stack_first.pop()
            n1 = get_normal(current_p)
            _subset_id = kd_total.query_ball_point(current_p[0:3], r=1)
            for subset_point_id in _subset_id:
                n2 = get_normal(point_cloud[subset_point_id])
                if n2 >= 0.99 * n1 and n2 <= 1.01 * n1 and subset_point_id not in set_point:
                    set_point.add(subset_point_id)
                    obstacle_pts.append(point_cloud[subset_point_id])
                    stack_first.append(point_cloud[subset_point_id])
                else: continue

        # 3 -- CLEAN OBSTACLE POINTS
        obstacle_pts_ = []
        obstacle_pts_final = []
        obstacle_pts_final2d = []
        for _point_ in obstacle_pts:
            # Radius search
            query1 = kd_total.query_ball_point(_point_[0:3], r=1)
            count_higher = 0 
            for id_p in query1:
                if point_cloud[id_p][2] > (_point_[2] + 0.1):
                    count_higher += 1
            if count_higher < 2:
                obstacle_pts_.append(_point_)
        
        # We get rid of isolated points
        if len(obstacle_pts_) < 2: continue
        else:
            kd_first = scipy.spatial.KDTree(extract_xyz(obstacle_pts_))
            for _point_2 in obstacle_pts_:
                query2 = kd_first.query_ball_point(_point_2[0:3], r=1)
                if(len(query2)) > 2:
                    obstacle_pts_final.append(_point_2)
                    pt = (_point_2[0], _point_2[1])
                    obstacle_pts_final2d.append(_point_2[0:2])
                    obstacle_pts_total.append(_point_2)
        
        # 4 -- OBSTACLE POINTS ARE OFFSET (AS HEXAGONS) AND MERGED IF OVERLAPPING
        hexagons = []
        for obs in obstacle_pts_final2d:
            x = obs[0]
            y = obs[1]
            param = 0.23
            one_hexagon = Polygon([(x+2*param, y), (x+param, y+2*param), (x-param, y+2*param), 
                                    (x-2*param, y), (x-param, y-2*param), (x+param, y-2*param)])
            hexagons.append(one_hexagon)
        # We merge the hegaxons
        hull = cascaded_union(hexagons)

        # 5 -- OFFSET-BACK OF THE MERGED HEXAGONS
        if hull.type == "MultiPolygon":  
            for poly in hull:
                buffered_back = poly.exterior.parallel_offset(param/1.5)
                try:    
                    polygon_new = Polygon(buffered_back)
                    coords = polygon_new.exterior.coords
                    hulls.append(coords)
                    hulls_polygons.append(polygon_new)
                except:
                    hulls.append(poly.exterior.coords)
                    hulls_polygons.append(poly)

        if hull.type == "Polygon": 
            buffered_back = hull.exterior.parallel_offset(param/1.5)
            try:
                polygon_new = Polygon(buffered_back)
                coords = polygon_new.exterior.coords
                hulls.append(coords)
                hulls_polygons.append(polygon_new)
            except:
                hulls.append(hull.exterior.coords)
                hulls_polygons.append(hull)

        # Retrieve points_relative height to the 3d model
        for h in hulls_polygons:
            id_point2d = 0
            max_height = 0.00
            for point2d in obstacle_pts_final2d:
                p2d = Point(point2d[0], point2d[1])
                if h.contains(p2d):
                    tr = point_toFace_dict[(point2d[0], point2d[1])]
                    v1, v2, v3 = vertices[tr[0]], vertices[tr[1]], vertices[tr[2]]
                    distance2 = shortest_distance(obstacle_pts_final[id_point2d], plane_equation(v1, v2, v3))
                    if distance2 > max_height:
                        max_height = distance2     
                id_point2d += 1
            max_heights.append(max_height)
        
        # Dictionnary used for final CityJSON
        
        identificatie = get_buildingID(input_json, building_nb - 1)
        cityJSON_id = get_cityJSON_ID(input_json, building_nb - 1)
        value = (cityJSON_id, roof_area)
        dict_buildings[identificatie] = value

        # 5 -- OBSTACLE AREA COMPUTATION 2D
        obstacle_area = 0
        for polygon in hulls_polygons:
            obstacle_area += polygon.area
    
        # 6 -- WRITE POLYGONS TO GEOJSON
        for p in hulls_polygons:
            features.append(Feature(geometry=p, properties={"CityObject": get_cityJSON_ID(input_json, building_nb - 1),
                                                            "Identificatie": get_buildingID(input_json, building_nb - 1),
                                                            "Max obstacle height": max_heights[-1]
                                                            }))
        building_nb += 1

    crs = {
        "type": "name",
        "properties": {
            "name": "EPSG:28992"
        }
    }
    feature_collection = FeatureCollection(features, crs=crs)
    with open(output_files + '.geojson', 'w') as geojson:
        dump(feature_collection, geojson)
    
    # 7 -- STORE NEW POINT CLOUD (PLY) with distance-to-plane attribute
    #write_ply(obstacle_pts_final, './fileout/points_obtacle.ply')
    write_ply_final(point_cloud, dict_points, output_files + '.ply')
    print('New point cloud stored in ' + output_files + '.ply')

    print(dict_buildings)
    return ((output_files + '.geojson'), dict_buildings)