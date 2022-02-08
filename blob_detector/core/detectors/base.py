import cv2
import numpy as np

from scipy import stats

from blob_detector.core import binarizers
from blob_detector.core.pipeline import Pipeline
from blob_detector import utils

class Detector:

    def __init__(self, *,
                 scale_min_size: int,
                 scale_min_scale: float,

                 pre_equalize: bool,
                 pre_sigma: float,

                 kernel_size: int,
                 dilate_iterations: int,

                 post_enlarge: float,

                 thresh_type: binarizers.BinarizerType,
                 **thresh_kwargs):
        super().__init__()

        self._pipeline = Pipeline()\
            .rescale(min_size=scale_min_size, min_scale=scale_min_scale)\
            .preprocess(equalize=pre_equalize, sigma=pre_sigma)\
            .binarize(type=thresh_type, **thresh_kwargs)\
            .open_close(kernel_size=kernel_size, iterations=dilate_iterations)\
            .detect()\
            .bbox_filter(enlarge=post_enlarge)

    def __call__(self, im: np.ndarray):

        imgs, final_res = self._pipeline(im)

        im0, im1, bin_im, post_bin_im, *_ = imgs

        return (im1, bin_im, post_bin_im), final_res
