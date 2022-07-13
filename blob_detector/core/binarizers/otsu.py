import cv2
import numpy as np

from skimage import filters
from skimage.filters import rank
from skimage.filters import threshold_otsu
from skimage.morphology import disk

from blob_detector import core
from blob_detector.core.binarizers import base

class OtsuTresholder(base.BaseThresholder):

    def __init__(self, *, use_cv2: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._use_cv2 = use_cv2

    def threshold(self, X: core.ImageWrapper) -> base.ThreshReturn:
        im = X.im
        if self._use_cv2:
            thresh, bin_im = cv2.threshold(im, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        else:
        	thresh = filters.threshold_otsu(im)
        	bin_im = None

        return base.ThreshReturn(thresh=thresh, bin_im=bin_im)

class LocalOtsuTresholder(base.BaseThresholder):

    def __init__(self, *, window_size: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size

    def threshold(self, X: core.ImageWrapper) -> base.ThreshReturn:
        mask = X.mask if self._use_masked else None
        im = X.im
        footprint = disk(self.window_size)
        thresh = filters.rank.otsu(im, footprint, mask=mask)

        bin_im = im >= thresh
        bin_im = (bin_im * 255).astype(np.uint8)

        return base.ThreshReturn(thresh=0, bin_im=bin_im)
