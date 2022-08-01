from __future__ import annotations

import cv2
import numpy as np
import typing as T

from dataclasses import dataclass
from matplotlib import pyplot as plt

from blob_detector.core.wrapper.base import BaseWrapper

@dataclass
class _ImAttrs:
    im: np.ndarray

@dataclass
class ImageWrapper(BaseWrapper, _ImAttrs):
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

    @property
    def im_masked(self):
        im = self.im.copy()
        im[self.mask == 0] = 0
        return im


    @mask.setter
    def mask(self, mask):
        if mask is None:
            if self.parent is None:
                mask = np.ones_like(self.im, dtype=np.float32)
            else:
                mask = self.parent.mask.copy()

        return self.__set_mask(mask)

    def resize(self, size, *, interpolation: int = cv2.INTER_LINEAR) -> ImageWrapper:
        self.im = cv2.resize(self.im, dsize=size, interpolation=interpolation)
        self.mask = cv2.resize(self.mask, dsize=size, interpolation=interpolation)
        return self

    def show(self, ax: T.Optional[plt.Axes] = None, masked: bool = False) -> plt.Axes:
        ax = ax or plt.gca()

        if masked:
            ax.imshow(self.im, cmap=plt.cm.gray, alpha=0.5)
            ax.imshow(self.mask, cmap=plt.cm.jet, alpha=0.5, vmin=0.0, vmax=1.0)

        else:
            ax.imshow(self.im, cmap=plt.cm.gray)

        ax.set_title(self.creator)
        return ax
