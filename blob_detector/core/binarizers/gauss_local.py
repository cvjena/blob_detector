import cv2
import logging
import numpy as np

from skimage import filters

from blob_detector import core
from blob_detector import utils
from blob_detector.core.binarizers import base

class GaussLocalTresholder(base.BaseLocalThresholder):

    def threshold(self, X: core.ImageWrapper) -> base.ThreshReturn:
        im = X.im

        logging.debug(f"using blocksize {self._window_size}")

        thresh, bin_im = 0, None

        mask = X.mask if self._use_masked else 1

        if self._use_masked:
            im = im * X.mask.astype(np.uint8)

        if self._use_cv2:
            bin_im = cv2.adaptiveThreshold(im,
                maxValue=utils.get_maxvalue(im),
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                thresholdType=cv2.THRESH_BINARY,
                blockSize=self._window_size,
                C=self._offset)

        else:
            thresh = filters.threshold_local(im,
                block_size=self._window_size,
                method="gaussian",
                mode="constant",
                offset=self._offset)

        return base.ThreshReturn(thresh=thresh, bin_im=bin_im)
