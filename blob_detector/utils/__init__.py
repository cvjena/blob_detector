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


def _im_mean_std(integral, integral_sq, bbox=None):
    if bbox is None:
        arr_sum = integral[-1, -1]
        arr_sum_sq = integral_sq[-1, -1]
        N = (integral.shape[0] - 1) * (integral.shape[1] - 1)
    else:
        (x0, y0), (x1, y1) = bbox
        A, B, C, D = (y0,x0), (y1,x0), (y0,x1), (y1,x1)
        arr_sum = integral[D] + integral[A] - integral[B] - integral[C]
        arr_sum_sq = integral_sq[D] + integral_sq[A] - integral_sq[B] - integral_sq[C]

        N = (x1-x0) * (y1-y0)

    arr_mean = arr_sum / N
    arr_std  = np.sqrt((arr_sum_sq - (arr_sum**2) / N) / N)

    return arr_mean, arr_std, N

def _check_ratio(bbox, threshold=0.25):
    (x0, y0), (x1, y1) = bbox
    h, w = y1-y0, x1-x0

    ratio = min(h, w) / max(h, w)
    return ratio >= threshold

def _check_area(bbox, imshape, minarea=4e-4, maxarea=1/9):

    (x0, y0), (x1, y1) = bbox
    h, w = y1-y0, x1-x0
    H, W = imshape
    area_ratio = (h*w) / (H*W)

    return minarea <= area_ratio <= maxarea

def _rescale(im: np.ndarray, min_size: int,
             min_scale: float = -1,
             interpolation = cv2.INTER_LINEAR
            ):
    H, W = im.shape
    _scale = min_size / min(H, W)
    scale = max(min_scale, min(1, _scale))
    size = int(W * scale), int(H * scale)

    return cv2.resize(im, dsize=size, interpolation=interpolation)

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

def _enlarge(bboxes, enlarge: int):
    if enlarge <= 0:
        return bboxes

    enlarged = []
    for bbox in bboxes:
        (x0, y0), (x1, y1) = bbox

        x0, y0 = max(x0 - enlarge, 0), max(y0 - enlarge, 0)
        x1, y1 = x1 + enlarge, y1 + enlarge

        enlarged.append([(x0, y0), (x1, y1)])

    return enlarged

def _gaussian(im: np.ndarray, sigma: float = 5.0):
    return filters.gaussian(im, sigma=sigma, preserve_range=True).astype(im.dtype)

def _high_pass(im: np.ndarray, sigma: float = 5.0, *, return_low_pass: bool = False):
    max_value = get_maxvalue(im)
    dtype = im.dtype

    im = im.astype(np.float32) / max_value

    gauss = _gaussian(im, sigma=sigma)
    high_pass = abs((im - gauss) * max_value).astype(dtype)

    return (high_pass, gauss) if return_low_pass else high_pass
