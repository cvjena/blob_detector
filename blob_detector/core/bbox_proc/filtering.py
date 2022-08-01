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

        nms_det = detection.copy(creator="NMSBoxes")

        _bboxes = [bbox.as_rectangle(self._im) for bbox in nms_det.bboxes]
        _scores = [bbox.score for bbox in nms_det.bboxes]
        inds = cv2.dnn.NMSBoxes(_bboxes,
            scores=np.array(_scores),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
        )

        if len(inds) != 1 and isinstance(inds, np.ndarray):
            inds = inds.squeeze()

        nms_det.select(inds)

        valid_det = nms_det.copy(creator="Validation")

        for bbox in valid_det.bboxes:
            bbox.active = bbox.active and bbox.is_valid

        final_det = valid_det.copy(creator="Enlargement")
        final_det.bboxes = [bbox.enlarge(self.enlarge) if bbox.active else bbox
            for bbox in final_det.bboxes]

        return final_det
