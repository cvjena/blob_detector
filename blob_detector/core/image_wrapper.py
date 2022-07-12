from __future__ import annotations

import numpy as np
import typing as T

class ImageWrapper:

    def __init__(self, im: np.ndarray, parent: T.Optional[ImageWrapper] = None):
        self.im = im

        assert isinstance(im, np.ndarray), f"Wrong im type: {type(im)}"

        self.parent = parent
        if parent is None:
            self.mask = np.ones_like(im, dtype=np.float32)
        else:
            self.mask = parent.mask.copy()

