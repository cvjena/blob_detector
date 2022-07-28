import cv2
import numpy as np

from blob_detector import utils
from blob_detector.core import ImageWrapper


class Preprocessor:

    def __init__(self, *, equalize: bool = False, sigma: float = 5.0):
        super().__init__()

        self._equalizer = None
        self.sigma = sigma

        if equalize:
            self._equalizer = cv2.createCLAHE(
                clipLimit=2.0, tileGridSize=(10,10))

    def __call__(self, X: ImageWrapper):

        im = X.im

        if self._equalizer is not None:
            im = self._equalizer.apply(im)

        if self.sigma >= 1:
            im = utils._gaussian(im, self.sigma)

        return ImageWrapper(im, parent=X)
