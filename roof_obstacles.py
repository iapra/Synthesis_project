# SYNTHESIS PROJECT.2021

import math
import numpy as np
import scipy.spatial

#-- to speed up the nearest neighbour us a kd-tree
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
# kd = scipy.spatial.KDTree(list_pts)
# d, i = kd.query(p, k=1)

def detect_obstacles(point_cloud, vertices, faces, output_file):
    print(len(vertices))
    print(len(faces))
    # Loop through triangles and select points above it (in a local subset)
    for triangle in faces:
        assert (len(triangle) == 3)
    return
    # Distance points to surface: discard points closer than .. threshold to define

    # Obstacle points convex-hull

    # Projection on mesh

    # Solar potential area computation

    # Store new attribute per triangle




