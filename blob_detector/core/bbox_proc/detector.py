import cv2
import numpy as np
import typing as T

from blob_detector.core.bbox import BBox
from blob_detector.core.bbox_proc.base import Result


class Detector:

    def __call__(self, im: np.ndarray) -> Result:

        contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        h, w, *c = im.shape
        wh = np.array([w, h]).reshape(1,1,2)
        contours = [c / wh for c in contours]

        bboxes = [BBox(
            *cont.min(axis=0)[0],
            *cont.max(axis=0)[0]
        ) for cont in contours]

        return Result(im, bboxes)