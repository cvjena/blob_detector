import cv2
import numpy as np

from blob_detector import utils
from blob_detector import core


class BorderRemoval:

    def __call__(self, X: core.ImageWrapper) -> core.ImageWrapper:

        res = X.im * X.mask.astype(X.im.dtype)
        return core.ImageWrapper(res, parent=X)

class BorderFinder:

    def __init__(self, *, threshold: float = 50.0, pad: int = 10):
        self.threshold = threshold
        self.pad = pad

    def __call__(self, X: core.ImageWrapper) -> core.ImageWrapper:

        im = X.im

        if self.pad >= 1:
            im = np.pad(im, self.pad)

        _, bin_im = cv2.threshold(im, self.threshold,
            utils.get_maxvalue(im), cv2.THRESH_BINARY)

        contours, _hierarchy = cv2.findContours(
            image=bin_im,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE)

        border = sorted(contours, key=cv2.contourArea, reverse=True)

        if border:
            mask = np.zeros_like(im, dtype=np.float32)

            border = cv2.approxPolyDP(border[0], 100, True)
            cv2.drawContours(mask, [border], -1, 1.0, -1)
        else:
            mask = np.ones_like(im, dtype=np.float32)


        if self.pad >= 1:
            im = im[self.pad:-self.pad, self.pad:-self.pad]
            mask = mask[self.pad:-self.pad, self.pad:-self.pad]

        res = X.copy()
        res.mask = mask
        return res
