import cv2
import logging
import numpy as np

try:
    from skimage import filters
except ImportError:
    SKIMAGE_AVAILABLE = False
else:
    SKIMAGE_AVAILABLE = True

from blob_detector import core
from blob_detector import utils
from blob_detector.core.binarizers import base

class GaussLocalTresholder(base.BaseLocalThresholder):

    def threshold(self, im: np.ndarray) -> base.ThreshReturn:

        logging.debug(f"using blocksize {self._window_size}")

        thresh, bin_im = 0, None

        if self._use_cv2:
            bin_im = cv2.adaptiveThreshold(im,
                maxValue=utils.get_maxvalue(im),
                adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,\
                thresholdType=cv2.THRESH_BINARY,
                blockSize=self._window_size,
                C=self._offset)

        else:
            assert SKIMAGE_AVAILABLE, "scikit-image is not installed!"
            thresh = filters.threshold_local(im,
                block_size=self._window_size,
                method="gaussian",
                mode="constant",
                offset=self._offset)

        return base.ThreshReturn(thresh=thresh, bin_im=bin_im)
