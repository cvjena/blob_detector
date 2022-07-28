import cv2
import numpy as np

from blob_detector.utils.base import get_maxvalue


def _gaussian(im: np.ndarray, sigma: float = 5.0):
    return cv2.GaussianBlur(im, (0,0), sigmaX=sigma).astype(im.dtype)
    from skimage import filters
    return filters.gaussian(im, sigma=sigma, preserve_range=True).astype(im.dtype)

def _high_pass(im: np.ndarray, sigma: float = 5.0, *, return_low_pass: bool = False):
    max_value = get_maxvalue(im)
    dtype = im.dtype

    im = im.astype(np.float32) / max_value

    gauss = _gaussian(im, sigma=sigma)
    high_pass = abs((im - gauss) * max_value).astype(dtype)

    return (high_pass, gauss) if return_low_pass else high_pass


def _correlate(im1, im2, normalize=True):

    # 0..255 -> 0..1
    im1 = im1.astype(np.float32) / 255
    im2 = im2.astype(np.float32) / 255

    mean = (im1.mean() + im2.mean()) / 2

    im1 = im1 - mean
    im2 = im2 - mean

    ##### OpenCV implementation
    k_h, k_w, *_ = im2.shape

    _im = np.pad(im1, [(k_h//2, k_h//2-1), (k_w//2, k_w//2-1)], mode="reflect")
    corr = cv2.matchTemplate(_im, im2, cv2.TM_CCORR)

    if not normalize:
        return corr

    corr -= corr.min()
    return corr / corr.max()

