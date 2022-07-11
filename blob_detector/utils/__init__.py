import cv2
import numpy as np

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

        return im * mask

    return im


from blob_detector.utils.filters import _correlate
from blob_detector.utils.filters import _gaussian
from blob_detector.utils.filters import _high_pass
