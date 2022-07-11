import cv2
import numpy as np

from blob_detector import utils


class BorderRemoval:

    def __init__(self, *, area_thresh: float = 0.5):
        super().__init__()
        self.area_thresh = area_thresh

    def __call__(self, im: np.ndarray):

        res = utils._mask_border(im, area_thresh=self.area_thresh)
        return res
