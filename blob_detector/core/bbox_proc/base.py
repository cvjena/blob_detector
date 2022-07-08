import cv2
import numpy as np
import typing as T

from scipy import stats

from blob_detector.core.bbox import BBox

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
