import cv2
import numpy as np

from blob_detector.core import ImageWrapper


class MorphologicalOps:

    def __init__(self, *, kernel_size: int, iterations: int):
        super().__init__()

        self.kernel = None
        self.iterations = iterations

        if kernel_size >= 3:
            self.kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)


    def __call__(self, X: ImageWrapper):

        im = X.im
        if self.kernel is not None:
            kernel = self.kernel.astype(im.dtype)
            im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
            im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)

        if self.iterations >= 1:
            im = cv2.erode(im, kernel, iterations=self.iterations)
            im = cv2.dilate(im, kernel, iterations=self.iterations)

        return ImageWrapper(im, parent=X)
