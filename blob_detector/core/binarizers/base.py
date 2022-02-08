import abc
import numpy as np
import typing as T

from blob_detector import utils

ThreshReturn = T.Tuple[float, T.Optional[np.ndarray]]

class BaseThresholder(abc.ABC):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, im: np.ndarray) -> np.ndarray:
        assert im.ndim == 2, "Should be an image with one channel!"

        max_value = utils.get_maxvalue(im)

        thresh, bin_im = self.threshold(im)

        if bin_im is None:
            bin_im = ((im > thresh) * max_value).astype(im.dtype)

        return max_value - bin_im

    @abc.abstractmethod
    def threshold(self, im: np.ndarray) -> ThreshReturn:
        pass
