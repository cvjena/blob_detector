import cv2
import numpy as np

from blob_detector import core


class Rescaler:

    def __init__(self, *, min_size: int, min_scale: float, interpolation = cv2.INTER_LINEAR):
        super().__init__()
        self.min_size = min_size
        self.min_scale = min_scale
        self.interpolation = interpolation

    def __call__(self, X: core.ImageWrapper) -> core.ImageWrapper:

        im = X.im

        H, W = im.shape
        _scale = self.min_size / min(H, W)
        scale = max(self.min_scale, min(1, _scale))
        size = int(W * scale), int(H * scale)

        im = cv2.resize(im, dsize=size, interpolation=self.interpolation)

        return core.ImageWrapper(im, parent=X)
