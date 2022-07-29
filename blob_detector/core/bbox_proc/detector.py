import cv2
import numpy as np
import typing as T

from blob_detector import core
from blob_detector.core.bbox import BBox


class Detector:

    def __init__(self, use_masked: bool = False):
        super().__init__()
        self._use_masked = use_masked

    def __call__(self, X: core.ImageWrapper) -> core.DetectionWrapper:

        im = X.im_masked if self._use_masked else X.im

        contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        h, w, *c = im.shape
        wh = np.array([w, h]).reshape(1,1,2)
        contours = [c / wh for c in contours]

        bboxes = [BBox(
            *cont.min(axis=0)[0],
            *cont.max(axis=0)[0]
        ) for cont in contours]

        bboxes = [bbox for bbox in bboxes if 0 not in bbox.size]

        return core.DetectionWrapper(im=X, bboxes=bboxes, creator="Detector")
