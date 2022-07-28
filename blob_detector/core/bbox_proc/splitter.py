import numpy as np
import typing as T

from blob_detector.core.bbox import BBox
from blob_detector.core.bbox_proc.base import ImageSetter
from blob_detector.core.bbox_proc.base import Result
from blob_detector.core.binarizers import BinarizerType

# from matplotlib import pyplot as plt

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


    def __call__(self, im: np.ndarray, bboxes: T.List[BBox]) -> Result:
        self._check_image()

        # fig0, ax0 = plt.subplots()
        # ax0.imshow(self._im, cmap=plt.cm.gray)

        # n_cols = int(np.ceil(np.sqrt(len(bboxes))))
        # n_rows = int(np.ceil(len(bboxes) / n_cols))
        # fig, axs = plt.subplots(n_rows, n_cols, squeeze=False)

        result = []

        for i, bbox in enumerate(bboxes):
            # bbox.plot(self._im, ax0, edgecolor="blue")
            # always add the box itself
            result.append(bbox)

            # ax = axs[np.unravel_index(i, axs.shape)]

            if not bbox.splittable(self._im):
                continue

            orig_crop = bbox.crop(self._im, enlarge = False)
            # ax.imshow(orig_crop, cmap=plt.cm.gray)

            im0 = self.preproc(orig_crop)
            _im0, new_bboxes = self.detector(im0)
            for new_bbox in new_bboxes:
                # new_bbox.plot(orig_crop, ax)
                # rescale to the relative coordinates of the original image
                result.append(new_bbox * bbox.size + bbox.origin)

        # for bbox in result:
        #     bbox.plot(self._im, ax0)

        # plt.show()
        # plt.close()

        # reset the image attribute
        self._im = None
        return Result(im, result)
