import cv2
import numpy as np

from skimage import filters

def int_tuple(values):
    return tuple(map(int, values))

def get_maxvalue(im: np.ndarray):
    max_value = { dt().dtype.name: value for dt, value in
        [(np.uint8, 255),
        (np.float32, 1.0),
        (np.float64, 1.0)]
    }.get(im.dtype.name)
    assert max_value is not None, f"Unsupported {im.dtype=}"
    return max_value


def _mask_border(im: np.ndarray, area_thresh: float = 0.50) -> None:

    contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ### Find smallest contour with an area above the threshold
    imsize = im.shape[0] * im.shape[1]
    selected = None
    for cont in contours:
        area = cv2.contourArea(cont) / imsize
        if area >= area_thresh and (selected is None or selected[0] > area):
            selected = (area, cont)

    ### we found a counter enclosing the white screen, so mask it accordingly
    if selected is not None:
        area, cont = selected

        hull = cv2.convexHull(cont, False)
        mask = np.zeros(im.shape, dtype=np.uint8)
        # here we have a mask with 1's inside the contour
        cv2.drawContours(mask, [hull], 0, color=1, thickness=-1)

        im[:] = im * mask

def _gaussian(im: np.ndarray, sigma: float = 5.0):
    return filters.gaussian(im, sigma=sigma, preserve_range=True).astype(im.dtype)

def _high_pass(im: np.ndarray, sigma: float = 5.0, *, return_low_pass: bool = False):
    max_value = get_maxvalue(im)
    dtype = im.dtype

    im = im.astype(np.float32) / max_value

    gauss = _gaussian(im, sigma=sigma)
    high_pass = abs((im - gauss) * max_value).astype(dtype)

    return (high_pass, gauss) if return_low_pass else high_pass
