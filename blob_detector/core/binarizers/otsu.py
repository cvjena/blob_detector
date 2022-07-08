import cv2
import numpy as np

from skimage import filters

from blob_detector.core.binarizers import base

class OtsuTresholder(base.BaseThresholder):

    def __init__(self, *, use_cv2: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._use_cv2 = use_cv2

    def threshold(self, im: np.ndarray) -> base.ThreshReturn:
        if self._use_cv2:
            thresh, bin_im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
        	thresh = filters.threshold_otsu(im)
        	bin_im = None

        return base.ThreshReturn(thresh=thresh, bin_im=bin_im)
