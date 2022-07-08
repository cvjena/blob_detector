import cv2
import logging
import numpy as np

from skimage import filters

from blob_detector.core.binarizers import base

class GaussLocalTresholder(base.BaseThresholder):

    def __init__(self, *, block_size_scale: float = 0.1, do_padding: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._do_padding = do_padding
        self._block_size_scale = block_size_scale

    def threshold(self, im: np.ndarray) -> base.ThreshReturn:
        # make block size an odd number
        block_size = min(im.shape) * self._block_size_scale // 2 * 2 + 1

        if self._do_padding:
            pad = int(block_size * 0.1)
            im = np.pad(im, [(pad, pad), (pad, pad)])

        logging.debug(f"using blocksize {block_size}")
        thresh = filters.threshold_local(im,
                                         block_size=block_size,
                                         mode="constant",
                                        )
        return base.ThreshReturn(thresh=thresh)
