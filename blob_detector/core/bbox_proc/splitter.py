import numpy as np
import typing as T

from blob_detector import core
from blob_detector import utils
from blob_detector.core.bbox import BBox
from blob_detector.core.bbox_proc.base import ImageSetter
from blob_detector.core.binarizers import BinarizerType

class Splitter(ImageSetter):

    def __init__(self, preproc, detector) -> None:
        super().__init__()

        self.preproc = preproc
        # self.preproc.rescale(min_size=300, min_scale=-1)
        self.preproc.preprocess(sigma=2.0, equalize=True)
        self.preproc.binarize(type=BinarizerType.gauss_local,
                              window_size=15, offset=2.0)

        self.preproc.open_close(kernel_size=5, iterations=2)

        self.detector = detector
        self.detector.detect()


    def __call__(self, detection: core.DetectionWrapper) -> core.DetectionWrapper:
        self._check_image()

        new_bboxes = []

        for i, bbox in enumerate(detection.bboxes):
            # always add the box itself
            new_bboxes.append(bbox)

            if not bbox.splittable(self._im):
                continue

            orig_crop = bbox.crop(self._im, enlarge = False)
            im0 = self.preproc(orig_crop)

            crop_detection: core.DetectionWrapper = self.detector(im0)
            for new_bbox in crop_detection.bboxes:
                # rescale to the relative coordinates of the original image
                new_bboxes.append(new_bbox * bbox.size + bbox.origin)

        # reset the image attribute
        self._im = None

        final_det = detection.copy(creator="Splitter", bboxes=new_bboxes)

        return final_det
