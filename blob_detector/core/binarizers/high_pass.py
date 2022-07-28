import cv2
import numpy as np

from blob_detector import utils
from blob_detector import core
from blob_detector.core.binarizers import base


class HighPassTresholder(base.BaseLocalThresholder):

    def __init__(self, *, sigma: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self._sigma = sigma

    def threshold(self, im: np.ndarray) -> base.ThreshReturn:

        edges = utils._high_pass(im, sigma=(self._window_size - 1)/4)

        if self._sigma >= 1:
            edges = utils._gaussian(edges, sigma=self._sigma)


        thresh = edges.mean()

        bin_im = (edges <= thresh) * 255
        bin_im = bin_im.astype(np.uint8)

        return base.ThreshReturn(thresh=thresh, bin_im=bin_im)
