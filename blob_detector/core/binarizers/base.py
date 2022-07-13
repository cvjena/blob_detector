import abc
import numpy as np
import typing as T

from blob_detector import utils
from blob_detector import core

class ThreshReturn(T.NamedTuple):
    thresh: float
    bin_im: T.Optional[np.ndarray] = None

class BaseThresholder(abc.ABC):

    def __init__(self, *args,
                 use_masked: bool = True,
                 use_cv2: bool = False,
                 **kwargs):
        super().__init__()
        self._use_masked = use_masked
        self._use_cv2 = use_cv2

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

class BaseLocalThresholder(BaseThresholder):

    def __init__(self, *args,
                 window_size: int = 31,
                 offset: float = 0,
                 **kwargs):
        super().__init__(**kwargs)
        self._window_size = window_size
        self._offset = offset
