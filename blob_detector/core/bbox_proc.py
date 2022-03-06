import cv2
import numpy as np

from scipy import stats

from blob_detector import utils
from blob_detector.core.binarizers import BinarizerType


class Detector:

    def __call__(self, im: np.ndarray):

        contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        bboxes = [(
            utils.int_tuple(cont.min(axis=0)[0]),
            utils.int_tuple(cont.max(axis=0)[0])
        ) for cont in contours]

        return im, bboxes

class BBoxFilter:

    def __init__(self, *,
                 score_treshold: float = 0.99,
                 nms_threshold: float = 0.1,

                 enlarge: float = 0.01,
                 ):
        super().__init__()

        self.score_treshold = score_treshold
        self.nms_threshold = nms_threshold
        self.enlarge = enlarge

    def __call__(self, im: np.ndarray, bboxes: list):

        _im = im.astype(np.float64) / 255.

        integral, integral_sq = cv2.integral2(_im)

        im_mean, im_std, im_n = _im_mean_std(integral, integral_sq)
        _bboxes = [[x0, y0, x1-x0, y1-y0] for (x0, y0), (x1, y1) in bboxes]
        inds = cv2.dnn.NMSBoxes(_bboxes,
                            np.ones(len(bboxes), dtype=np.float32),
                            score_threshold=self.score_treshold,
                            nms_threshold=self.nms_threshold,
                           )
        inds2 = []
        means_stds = []

        for i in inds.squeeze():
            bbox = bboxes[i]
            box_mean, box_std, box_n = _im_mean_std(integral, integral_sq, bbox)
            ttest_res = stats.ttest_ind_from_stats(im_mean, im_std, im_n, box_mean, box_std, box_n)
            means_stds.append((box_mean, box_std, ttest_res))

            # if ttest_res.statistic < 0:
            #     continue

            #if box_std < 5e-2:
            #    continue

            if not (_check_ratio(bbox) and _check_area(bbox, im.shape)):
                continue

            inds2.append(i)

        factor = int(self.enlarge * max(im.shape))
        bboxes = _enlarge(bboxes, factor)

        return bboxes, inds2, means_stds

class Splitter:

    def __init__(self, preproc, detector):
        super().__init__()
        self._im = None

        self.preproc = preproc
        self.preproc.rescale(min_size=150, min_scale=-1)
        self.preproc.preprocess(equalize=True, sigma=3)
        self.preproc.binarize(type=BinarizerType.high_pass,
                              window_size=30, sigma=5)

        self.preproc.open_close(kernel_size=5, iterations=3)

        self.detector = detector.detect()


    def set_image(self, im: np.ndarray):
        self._im = im
        return im

    def split(self, im: np.ndarray, bboxes):

        n_cols = int(np.ceil(np.sqrt(len(bboxes))))
        n_rows = int(np.ceil(len(bboxes) / n_cols))
        result = []

        for i, bbox in enumerate(bboxes):
            (X0, Y0), (X1, Y1) = bbox

            orig_crop = self.crop(im, bbox)
            # print(orig_crop.shape, self._im.shape)
            im0 = self.preproc(orig_crop)

            _, new_bboxes = self.detector(im0)

            H, W = Y1 - Y0, X1 - X0
            h, w = im0.shape

            area = (W / im.shape[1]) * (H / im.shape[0])
            if area >= 0.5:
                # we dont want to split too big boxes
                result.append(bbox)
                continue

            for (x0, y0), (x1, y1) in new_bboxes:
                rel_x0, rel_x1 = x0 / w, x1 / w
                rel_y0, rel_y1 = y0 / h, y1 / h

                xy0 = utils.int_tuple([W * rel_x0 + X0, H * rel_y0 + Y0])
                xy1 = utils.int_tuple([W * rel_x1 + X0, H * rel_y1 + Y0])

                result.append((xy0, xy1))

            else: # if there were no new boxes
                result.append(bbox)

        # reset the image attribute
        self._im = None
        return im, result

    def crop(self, im, bbox):
        assert self._im is not None
        (x0,y0), (x1,y1) = bbox

        _h, _w = im.shape
        rel_bbox = (x0 / _w, y0 / _h), (x1 / _w, y1 / _h)
        (x0,y0), (x1,y1) = rel_bbox
        h, w = self._im.shape
        return self._im[int(y0*h):int(y1*h), int(x0*w):int(x1*w)]


#### bbox operations

def _enlarge(bboxes, enlarge: int):
    if enlarge <= 0:
        return bboxes

    enlarged = []
    for bbox in bboxes:
        (x0, y0), (x1, y1) = bbox

        x0, y0 = max(x0 - enlarge, 0), max(y0 - enlarge, 0)
        x1, y1 = x1 + enlarge, y1 + enlarge

        enlarged.append([(x0, y0), (x1, y1)])

    return enlarged

def _check_ratio(bbox, threshold: float = 0.25):
    (x0, y0), (x1, y1) = bbox
    h, w = y1-y0, x1-x0

    ratio = min(h, w) / max(h, w)
    return ratio >= threshold

def _check_area(bbox, imshape, minarea: float = 4e-4, maxarea:float = 1/9):

    (x0, y0), (x1, y1) = bbox
    h, w = y1-y0, x1-x0
    H, W = imshape
    area_ratio = (h*w) / (H*W)

    return minarea <= area_ratio <= maxarea

def _im_mean_std(integral, integral_sq, bbox=None):
    if bbox is None:
        arr_sum = integral[-1, -1]
        arr_sum_sq = integral_sq[-1, -1]
        N = (integral.shape[0] - 1) * (integral.shape[1] - 1)
    else:
        (x0, y0), (x1, y1) = bbox
        A, B, C, D = (y0,x0), (y1,x0), (y0,x1), (y1,x1)
        arr_sum = integral[D] + integral[A] - integral[B] - integral[C]
        arr_sum_sq = integral_sq[D] + integral_sq[A] - integral_sq[B] - integral_sq[C]

        N = (x1-x0) * (y1-y0)

    arr_mean = arr_sum / N
    arr_std  = np.sqrt((arr_sum_sq - (arr_sum**2) / N) / N)

    return arr_mean, arr_std, N

