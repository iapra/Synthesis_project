import skimage.filters
import numpy as np
from math import ceil, floor


def filter_predict(predict, sigma=5):
    """Filter out blobs of high probability areas out of prediction."""
    mask = skimage.filters.gaussian(predict, sigma=sigma)
    mask = mask > 0.3
    mask = skimage.filters.gaussian(mask, sigma=sigma)
    mask = mask > 0.02
    result = (predict * mask) > 0.3
    result = skimage.filters.gaussian(result, sigma=sigma/3)
    result = result > 0.45
    return result

def prep_input(input_img):
    """Prepare image array so it has right format for Keras prediction."""
    inp = np.expand_dims(input_img, 0)
    if inp.max() > 1.0:
        inp = inp / 255.0
    return inp

def mask_predict(predict, mask):
    """Mask out areas outside of building from prediction."""
    x1, y1 = predict.shape
    x2, y2 = mask.shape
    x, y = (x2 - x1), (y2 - y1)
    predict = np.pad(
        predict, 
        pad_width=[(floor(x/2), ceil(x/2)),(floor(y/2), ceil(y/2))], 
        mode='constant'
    )
    return predict * np.invert(mask).astype(int)

def map_predict(photo, predict):
    """Map areas with high prediction as magenta overlay on the image."""
    result_img = np.copy(photo)
    result_img[:,:,1] *= np.invert(predict)
    return result_img