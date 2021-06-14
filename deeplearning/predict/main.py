from geometry import get_fixed_box
from pdok import get_bag_pand, get_luchtfoto_rgb_array
from rasterize import shape_as_mask
from process import prep_input, filter_predict, mask_predict, map_predict

# Resolution, size and sigma depend on the chosen model
from models import rgb as model
_resolution = (256,256)
_size = (50,50)
_sigma = 5


def solar_panel_map(pand_id):
    """
    Show map of infered solar panels on the building aerial photo.

    Args:
        pand_id: Bag.pand ID of the desired building.

    Returns:
        Image array with the inferred solar panels mapped as magenta overlay.
    """
    luchtfoto, filtered = _solar_panel_predict(pand_id)
    return map_predict(luchtfoto, filtered)


def solar_panel_test(pand_id):
    """
    Infers whether building has solar panels on its roof.

    Args:
        pand_id: Bag.pand ID of the desired building.

    Returns:
        Boolean, True if its probable that theres and solar panel.
    """
    _, filtered = _solar_panel_predict(pand_id)
    return filtered.sum() > 150


def _solar_panel_predict(pand_id):

    building = get_bag_pand(pand_id)
    bbox = get_fixed_box(building, _size)
    luchtfoto = get_luchtfoto_rgb_array(bbox, _resolution)
    mask = shape_as_mask(building, bbox, _resolution)

    input = prep_input(luchtfoto)
    predict = model.predict(input)[0,:,:,0]
    predict = mask_predict(predict, mask)
    filtered = filter_predict(predict, _sigma)

    return luchtfoto, filtered


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    while True:
        pand_id = input('\nInput valid IDENTIFICATIE number for a dutch BAG:pand:  ')
        try:
            hasPanels = [solar_panel_test(pand_id)]
            print("Found some solar panels." if hasPanels else "No solar panels here.")
            consent = input('Want to see a picture? y/n:  ')
            if not consent.startswith('y'): 
                continue
            plt.imshow(solar_panel_map(pand_id))
            plt.show()
        except:
            print("Something went wrong.")
