import geojson
import numpy as np


def get_bounding_box(geometry):
    """Get bounding box of GEOJson like geometry."""
    coords = np.array(list(geojson.utils.coords(geometry)))
    return coords[:,0].min(), coords[:,1].min(), coords[:,0].max(), coords[:,1].max()

def resize_bounding_box(bbox, size):
    """Resize bounding box while keeping its center."""
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    min_x = center_x - size[0] / 2
    min_y = center_y - size[1] / 2
    max_x = center_x + size[0] / 2
    max_y = center_y + size[1] / 2
    return (min_x, min_y, max_x, max_y)

def get_fixed_box(geometry, size):
    """Get box of fixed size centered on geometry."""
    bbox = get_bounding_box(geometry)
    return resize_bounding_box(bbox, size)