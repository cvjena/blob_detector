import abc
import numpy as np
import typing as T

from blob_detector import utils
from blob_detector import core

class ThreshReturn(T.NamedTuple):
    thresh: float
    bin_im: T.Optional[np.ndarray] = None

class BaseThresholder(abc.ABC):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, X: core.ImageWrapper) -> core.ImageWrapper:
        assert X.im.ndim == 2, "Should be an image with one channel!"

        im = X.im
        max_value = utils.get_maxvalue(im)

        thresh, bin_im = self.threshold(X)

        if bin_im is None:
            bin_im = ((im > thresh) * max_value).astype(im.dtype)

        return core.ImageWrapper(max_value - bin_im, parent=X)

    @abc.abstractmethod
    def threshold(self, X: core.ImageWrapper) -> ThreshReturn:
        return ThreshReturn(0.0)
