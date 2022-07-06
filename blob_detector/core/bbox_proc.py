import cv2
import numpy as np
import typing as T

from scipy import stats

from blob_detector import utils
from blob_detector.core.binarizers import BinarizerType
from blob_detector.core.bbox import BBox

import matplotlib.pyplot as plt

class Result(T.NamedTuple):
    im: np.ndarray
    bboxes: T.List[BBox]

class FilteredResult(T.NamedTuple):
    im: np.ndarray
    bboxes: T.List[BBox]
    inds: T.List[int]
    stats: T.Tuple[float, float, T.Any]


class IntegralImage:

    def __init__(self, im: np.ndarray) -> None:
        self._im = im
        self.integral, self.integral_sq = cv2.integral2(im)

        self.mean, self.std, self.N = self.stats()

    def stats(self, bbox: T.Optional[BBox] = None) -> T.Tuple[float, float, int]:
        integral, integral_sq = self.integral, self.integral_sq

        if bbox is None:
            arr_sum = integral[-1, -1]
            arr_sum_sq = integral_sq[-1, -1]
            N = (integral.shape[0] - 1) * (integral.shape[1] - 1)

        else:
            x0, y0, x1, y1 = bbox(self._im)
            A, B, C, D = (y0,x0), (y1,x0), (y0,x1), (y1,x1)
            arr_sum = integral[D] + integral[A] - integral[B] - integral[C]
            arr_sum_sq = integral_sq[D] + integral_sq[A] - integral_sq[B] - integral_sq[C]

            N = (x1-x0) * (y1-y0)

        arr_mean = arr_sum / N
        arr_std  = np.sqrt((arr_sum_sq - (arr_sum**2) / N) / N)

        return arr_mean, arr_std, N

    def ttest(self, bbox: BBox):
        mean, std, N = self.stats(bbox)
        ttest_res = stats.ttest_ind_from_stats(self.mean, self.std, self.N,
            mean, std, N)

        return (mean, self.mean), (std, self.std), (N, self.N), ttest_res

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

class ImageSetter:
    def __init__(self):
        super().__init__()
        self._im = None

    def set_image(self, im: np.ndarray) -> None:
        """ sets the original image, so that we can crop properly """
        self._im = im
        return im

    def _check_image(self):
        assert self._im is not None, "set_image was not called!"

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
        # ax0.imshow(self._im, cmap="gray")

        # n_cols = int(np.ceil(np.sqrt(len(bboxes))))
        # n_rows = int(np.ceil(len(bboxes) / n_cols))
        # fig, axs = plt.subplots(n_rows, n_cols, squeeze=False)

        result = []

        for i, bbox in enumerate(bboxes):
            # bbox.plot(self._im, ax0, edgecolor="blue")

            # ax = axs[np.unravel_index(i, axs.shape)]
            orig_crop = bbox.crop(self._im, enlarge = False)
            # ax.imshow(orig_crop, cmap="gray")

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

class BBoxFilter(ImageSetter):

    def __init__(self, *,
                 score_treshold: float = 0.99,
                 nms_threshold: float = 0.1,

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
        bbox_stats = [integral_im.ttest(bbox) for bbox in bboxes]

        _bboxes = [[bbox.x0, bbox.y0, bbox.w, bbox.h] for bbox in bboxes]

        inds = cv2.dnn.NMSBoxes(_bboxes,
                            scores=np.ones(len(_bboxes), dtype=np.float32),
                            score_threshold=self.score_treshold,
                            nms_threshold=self.nms_threshold,
                           )


        inds2 = []
        for i in inds.squeeze():
            bbox = bboxes[i]

            if not (bbox.ratio >= 0.1 and 4e-4 <= bbox.area <= 1/9):
                continue

            inds2.append(i)

        bboxes = [bbox.enlarge(self.enlarge) for bbox in bboxes]
        return FilteredResult(im, bboxes, inds2, bbox_stats)

class ScoreEstimator(ImageSetter):

    def __call__(self, im: np.ndarray, bboxes: T.List[BBox], idxs, bbox_stats):
        self._check_image()

        # fig0, ax0 = plt.subplots()
        # ax0.imshow(self._im, cmap="gray")

        # n_cols = int(np.ceil(np.sqrt(len(bboxes))))
        # n_rows = int(np.ceil(len(bboxes) / n_cols))
        # fig, axs = plt.subplots(n_rows, n_cols, squeeze=False)

        # for idx, bbox in enumerate(bboxes):
        #     if idx in idxs:
        #         continue
        #     bbox.plot(self._im, ax=ax0, edgecolor="blue")

        # for c, idx in enumerate(idxs):
        #     bbox = bboxes[idx]
        #     (mean, std, N, ttest_res) = bbox_stats[idx]
        #     crop = bbox.crop(self._im, enlarge=False)
        #     bbox.plot(self._im, ax=ax0)

        #     ax = axs[np.unravel_index(c, axs.shape)]
        #     print(mean, std, N)
        #     ax.imshow(crop, cmap="gray")
        #     ax.set_title(f"stat={ttest_res.statistic:.2f} \n p={ttest_res.pvalue:.2f}")


        # plt.show()
        # plt.close()

        return [bboxes[i] for i in idxs], None, None
        # import pdb; pdb.set_trace()

