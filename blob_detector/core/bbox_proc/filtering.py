import cv2
import numpy as np
import typing as T

from blob_detector.core.bbox import BBox
from blob_detector.core.bbox_proc.base import FilteredResult
from blob_detector.core.bbox_proc.base import ImageSetter
from blob_detector.core.bbox_proc.base import IntegralImage


class BBoxFilter(ImageSetter):

    def __init__(self, *,
                 score_treshold: float = 0.5,
                 nms_threshold: float = 0.3,

                 enlarge: float = 0.01,
                ) -> None:
        super().__init__()

        self.score_treshold = score_treshold
        self.nms_threshold = nms_threshold
        self.enlarge = enlarge

    def __call__(self, im: np.ndarray, bboxes: T.List[BBox]):
        self._check_image()

        _im = self._im.astype(np.float64) / 255.

        integral_im = IntegralImage(_im)
        bbox_stats = [integral_im.stats(bbox) for bbox in bboxes]

        _bboxes = [[bbox.x0, bbox.y0, bbox.w, bbox.h] for bbox in bboxes]

        inds = cv2.dnn.NMSBoxes(_bboxes,
                            scores=np.ones(len(_bboxes), dtype=np.float32),
                            score_threshold=self.score_treshold,
                            nms_threshold=self.nms_threshold,
                           )


        inds2 = []
        for i in inds.squeeze():
            bbox = bboxes[i]

            if not bbox.is_valid:
                continue

            inds2.append(i)

        bboxes = [bbox.enlarge(self.enlarge) for bbox in bboxes]
        return FilteredResult(im, bboxes, inds2, bbox_stats)
