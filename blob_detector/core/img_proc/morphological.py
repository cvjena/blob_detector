import cv2
import numpy as np

from blob_detector.core import ImageWrapper


class MorphologicalOps:

    def __init__(self, *, kernel_size: int, iterations: int):
        super().__init__()

        self.kernel_size = kernel_size
        self.iterations = iterations


    def __call__(self, X: ImageWrapper):

        im = X.im
        ksize = self.kernel_size
        if ksize is not None and ksize >= 3:
            kernel = np.ones((ksize, ksize), dtype=im.dtype)
            im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
            im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)

        if self.iterations >= 1:
            im = cv2.erode(im, kernel, iterations=self.iterations)
            im = cv2.dilate(im, kernel, iterations=self.iterations)

        return ImageWrapper(im, parent=X)
