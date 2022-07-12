from __future__ import annotations

import numpy as np
import typing as T

from dataclasses import dataclass
from matplotlib import pyplot as plt

@dataclass
class ImageWrapper:
    im: np.ndarray
    parent: T.Optional[ImageWrapper] = None
    __mask: T.Optional[np.ndarray] = None

    def __post_init__(self):

        assert isinstance(self.im, np.ndarray), f"Wrong im type: {type(self.im)}"

        # trigger default mask handling
        self.mask = self.__mask

    def copy(self):
        return self.__class__(im=self.im, parent=self)

    @property
    def shape(self):
        return self.im.shape

    @property
    def dtype(self):
        return self.im.dtype

    @property
    def size(self):
        return self.im.size

    @property
    def mask(self):
        return self.__mask

    def __set_mask(self, mask):
        self.__mask = mask
        assert mask.shape == self.im.shape

    @mask.setter
    def mask(self, mask):
        if mask is None:
            if self.parent is not None:
                mask = self.parent.mask.copy()
            else:
                mask = np.ones_like(self.im, dtype=np.float32)

        return self.__set_mask(mask)


    def show(self, ax: T.Optional[plt.Axes] = None, masked: bool = False):
        ax = ax or plt.gca()


        if masked:
            ax.imshow(self.im, cmap=plt.cm.gray, alpha=0.5)
            ax.imshow(self.mask, cmap=plt.cm.jet, alpha=0.5)

        else:
            ax.imshow(self.im, cmap=plt.cm.gray)
