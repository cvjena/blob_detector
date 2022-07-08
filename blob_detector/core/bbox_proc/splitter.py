import numpy as np
import typing as T

from blob_detector.core.bbox import BBox
from blob_detector.core.bbox_proc.base import ImageSetter
from blob_detector.core.bbox_proc.base import Result
from blob_detector.core.binarizers import BinarizerType

class Splitter(ImageSetter):

    def __init__(self, preproc, detector) -> None:
        super().__init__()

        self.preproc = preproc
        self.preproc.rescale(min_size=300, min_scale=-1)
        self.preproc.preprocess(equalize=True, sigma=3)
        self.preproc.binarize(type=BinarizerType.high_pass,
                              window_size=30, sigma=5)

        self.preproc.open_close(kernel_size=5, iterations=3)

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

            # ax = axs[np.unravel_index(i, axs.shape)]
            orig_crop = bbox.crop(self._im, enlarge = False)
            # ax.imshow(orig_crop, cmap=plt.cm.gray)

            if bbox.area >= 0.5:
                # we dont want to split too big boxes
                result.append(bbox)
                continue

            c_h, c_w, *_ = orig_crop.shape
            crop_diag = np.sqrt(c_w**2 + c_h**2)
            if crop_diag <= 50:
                # we dont want to split too small crops
                result.append(bbox)
                continue


            im0 = self.preproc(orig_crop)
            _im0, new_bboxes = self.detector(im0)
            for new_bbox in new_bboxes:
                # new_bbox.plot(orig_crop, ax)
                # rescale to the relative coordinates of the original image
                new_bbox = new_bbox * bbox.size + bbox.origin
                result.append(new_bbox)

            else:
                # if there were no new boxes
                result.append(bbox)


        # for bbox in result:
        #     bbox.plot(self._im, ax0)

        # plt.show()
        # plt.close()

        # reset the image attribute
        self._im = None
        return Result(im, result)
