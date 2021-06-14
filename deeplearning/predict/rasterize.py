import rasterio.features
import rasterio.transform


def shape_as_mask(feature, bbox, resolution):
    transform = rasterio.transform.from_bounds(*bbox, *resolution)
    return rasterio.features.geometry_mask([feature['geometry']], resolution, transform)
