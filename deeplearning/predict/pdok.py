from urllib.parse import urlencode
from urllib.request import urlopen
import skimage.io
import json


def get_bag_pand(pand_id):
    """
    Retrieve a building feature from PDOK BAG.

    Args:
        pand_id: Bag.pand ID of the desired building.

    Returns:
        GEOJSON pand feature as a dict.
    """
    endpoint = 'https://geodata.nationaalgeoregister.nl/bag/wfs/v1_1?'
    query = {
        'service': 'WFS',
        'version': '2.0.0',
        'request': 'GetFeature',
        'typeName': 'bag:pand',
        'outputFormat': 'application/json'
    }
    query['featureID'] = 'pand.bag:' + str(pand_id)

    url = endpoint + urlencode(query)

    with urlopen(url) as response:
        return json.load(response)['features'][0]


def get_luchtfoto_rgb_array(bbox, resolution):
    """
    Retrieve image as array from PDOK luchtfoto.

    Args:
        bbox: EPSG:28992 bbox
        resolution: (width, height) resolution tuple

    Returns:
        numpy array witht the RGB image
    """
    endpoint = 'https://geodata.nationaalgeoregister.nl/luchtfoto/rgb/wms?'
    query = {
        'service': 'WMS',
        'version': '1.1.1',
        'request': 'GetMap',
        'format': 'image/png',
        'layers': 'Actueel_ortho25',
        'srs': 'EPSG:28992'
    }
    query['bbox'] = ','.join([str(x) for x in bbox])
    query['width'] = resolution[0]
    query['height'] = resolution[1]
    
    url = endpoint + urlencode(query)
    return skimage.io.imread(url)
