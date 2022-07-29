import cv2
import numpy as np

from blob_detector.utils.base import show_intermediate
from blob_detector.utils.base import get_maxvalue
from blob_detector.utils.base import int_tuple

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
