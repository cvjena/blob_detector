import cv2
import numpy as np

try:
    from skimage import filters
    from skimage.morphology import disk
except ImportError:
    SKIMAGE_AVAILABLE = False
else:
    SKIMAGE_AVAILABLE = True

from blob_detector import core
from blob_detector.core.binarizers import base

class OtsuTresholder(base.BaseThresholder):

    def threshold(self, im: np.ndarray) -> base.ThreshReturn:
        if self._use_cv2:
            thresh, bin_im = cv2.threshold(im, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        else:
            assert SKIMAGE_AVAILABLE, "scikit-image is not installed!"
            thresh = filters.threshold_otsu(im)
            bin_im = None

        return base.ThreshReturn(thresh=thresh, bin_im=bin_im)

class LocalOtsuTresholder(base.BaseLocalThresholder):

    def __init__(self, **kwargs):
        from .high_pass import HighPassTresholder
        super().__init__(**kwargs)

        self._high_pass_thresh = None #HighPassTresholder(**kwargs)

    def threshold(self, im: np.ndarray) -> base.ThreshReturn:

        if False: #self._use_cv2:
            import pdb; pdb.set_trace()
        else:
            assert SKIMAGE_AVAILABLE, "scikit-image is not installed!"
            footprint = disk(self._window_size)
            thresh = filters.rank.otsu(im, footprint)

            bin_im = im >= thresh

        if self._high_pass_thresh is not None:
            hp_thesh, hp_bin_im = self._high_pass_thresh.threshold(im)
            bin_im = np.logical_or(bin_im, hp_bin_im == 255)

        bin_im = (bin_im * 255).astype(np.uint8)
        return base.ThreshReturn(thresh=0, bin_im=bin_im)
