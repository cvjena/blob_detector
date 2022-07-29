import numpy as np
import typing as T

from blob_detector import core
from blob_detector import utils
from blob_detector.core.bbox import BBox
from blob_detector.core.bbox_proc.base import ImageSetter
from blob_detector.core.binarizers import BinarizerType

class Splitter(ImageSetter):

    def __init__(self, preproc, detector, iou_thresh: float = 0.4) -> None:
        super().__init__()

        self.iou_thresh = iou_thresh

        self.preproc = preproc
        # self.preproc.rescale(min_size=300, min_scale=-1)
        self.preproc.preprocess(sigma=5.0, equalize=True)
        self.preproc.binarize(type=BinarizerType.otsu)

        self.preproc.open_close(kernel_size=5, iterations=2)

        self.detector = detector
        self.detector.detect()


    def __call__(self, detection: core.DetectionWrapper) -> core.DetectionWrapper:
        self._check_image()

        split_det = detection.copy(creator="Splitter")

        bboxes = list(split_det.bboxes)

        for i in range(len(split_det)):
            bbox = split_det.bboxes[i]

            if not bbox.splittable(self._im):
                continue

            orig_crop = bbox.crop(self._im, enlarge = False)
            im0 = self.preproc(orig_crop)

            crop_detection: core.DetectionWrapper = self.detector(im0)
            for new_bbox in crop_detection.bboxes:
                # rescale to the relative coordinates of the original image
                new_bbox = new_bbox * bbox.size + bbox.origin

                if bbox.iou(new_bbox) < self.iou_thresh:
                    split_det.bboxes.append(new_bbox)

        # reset the image attribute
        self._im = None

        return split_det
