import cv2
import numpy as np
import typing as T

from scipy import stats

from blob_detector.core.bbox import BBox

class ImageSetter:
    def __init__(self):
        super().__init__()
        self._im = None

    def set_image(self, im: np.ndarray) -> None:
        """ sets the original image, so that we can crop properly """
        self._im = im
        return im

    def _check_image(self):
        assert self._im is not None, "set_image was not called!"
