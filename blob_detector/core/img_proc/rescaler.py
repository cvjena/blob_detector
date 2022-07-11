import cv2
import numpy as np

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
