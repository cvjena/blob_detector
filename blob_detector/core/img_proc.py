import cv2
import numpy as np

from blob_detector import utils

class Rescaler:

    def __init__(self, *, min_size: int, min_scale: float, interpolation = cv2.INTER_LINEAR):
        super().__init__()
        self.min_size = min_size
        self.min_scale = min_scale
        self.interpolation = interpolation

    def __call__(self, im: np.ndarray):

        H, W = im.shape
        _scale = self.min_size / min(H, W)
        scale = max(self.min_scale, min(1, _scale))
        size = int(W * scale), int(H * scale)

        return cv2.resize(im, dsize=size, interpolation=self.interpolation)

class Preprocessor:

    def __init__(self, *, equalize: bool = False, sigma: bool = 5.0):
        super().__init__()

        self._equalizer = None
        self.sigma = sigma

        if equalize:
            self._equalizer = cv2.createCLAHE(
                clipLimit=2.0, tileGridSize=(10,10))

    def __call__(self, im: np.ndarray):

        if self._equalizer is not None:
            im = self._equalizer.apply(im)

        if self.sigma >= 1:
            im = utils._gaussian(im, self.sigma)

        return im

class MorphologicalOps:

    def __init__(self, *, kernel_size: int, iterations: int):
        super().__init__()

        self.kernel = None
        self.iterations = iterations

        if kernel_size >= 3:
            self.kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)


    def __call__(self, im: np.ndarray):

        if self.kernel is not None:
            kernel = self.kernel.astype(im.dtype)
            im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
            im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)

        if self.iterations >= 1:
            im = cv2.erode(im, kernel, iterations=self.iterations)
            im = cv2.dilate(im, kernel, iterations=self.iterations)

        return im
