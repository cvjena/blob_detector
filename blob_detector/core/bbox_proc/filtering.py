import cv2
import numpy as np
import typing as T

from blob_detector import core
from blob_detector.core.bbox import BBox
from blob_detector.core.bbox_proc.base import ImageSetter


class BBoxFilter(ImageSetter):

    def __init__(self, *,
                 score_threshold: float = 0.5,
                 nms_threshold: float = 0.3,

                 enlarge: float = 0.01,
                ) -> None:
        super().__init__()

        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.enlarge = enlarge

    def __call__(self, detection: core.DetectionWrapper) -> core.DetectionWrapper:
        self._check_image()

        im = detection.im
        bboxes = detection.bboxes
        _im = self._im.astype(np.float64) / 255.


        # integral_im = IntegralImage(_im)
        # bbox_stats = [integral_im.stats(bbox) for bbox in bboxes]
        # bbox_stats = [None for bbox in bboxes]

        _bboxes = [[bbox.x0, bbox.y0, bbox.w, bbox.h] for bbox in bboxes]

        inds = cv2.dnn.NMSBoxes(_bboxes,
            scores=np.ones(len(_bboxes), dtype=np.float32),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
        )

        if len(inds) != 1:
            inds = inds.squeeze()


        nms_det = detection.copy(creator="NMSBoxes", indices=inds)
        inds2 = []
        for i in inds:
            bbox = bboxes[i]

            if not bbox.is_valid:
                continue

            inds2.append(i)

        valid_det = nms_det.copy(creator="Validation", indices=inds2)

        final_det = valid_det.copy(creator="Enlargement", indices=inds2)
        final_det.bboxes = [bbox.enlarge(self.enlarge) for bbox in bboxes]

        return final_det
