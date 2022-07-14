import cv2
import numpy as np

from blob_detector import utils
from blob_detector import core


class BorderRemoval:

    def __init__(self, *, area_thresh: float = 0.5):
        super().__init__()
        self.area_thresh = area_thresh

    def __call__(self, X: core.ImageWrapper) -> core.ImageWrapper:

        res = X.im * X.mask.astype(X.im.dtype)
        #utils._mask_border(X.im, area_thresh=self.area_thresh)
        return core.ImageWrapper(res, parent=X)

class BorderFinder:

    def __init__(self, *, threshold: float = 50.0, pad: int = 10):
        self.threshold = threshold
        self.pad = pad

    def __call__(self, X: core.ImageWrapper) -> core.ImageWrapper:

        im = X.im
        bin_im = np.full_like(im, 255, dtype=np.uint8)

        bin_im[im <= self.threshold] = 0.0

        if self.pad >= 1:
            bin_im = np.pad(bin_im, self.pad)
            im = np.pad(im, self.pad)


        contours, _hierarchy = cv2.findContours(
            image=bin_im,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE)

        border = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        border = cv2.approxPolyDP(border, 100, True)

        mask = np.zeros_like(im, dtype=np.float32)
        cv2.drawContours(mask, [border], -1, 1.0, -1)

        if self.pad >= 1:
            im = im[self.pad:-self.pad, self.pad:-self.pad]
            mask = mask[self.pad:-self.pad, self.pad:-self.pad]

        res = X.copy()
        res.mask = mask
        return res
