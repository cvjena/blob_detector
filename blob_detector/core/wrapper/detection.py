from __future__ import annotations

import copy
import numpy as np
import typing as T

from dataclasses import dataclass

from blob_detector.core import ImageWrapper
from blob_detector.core.bbox import BBox
from blob_detector.core.wrapper.base import BaseWrapper

@dataclass
class _DetAttr:
    im: ImageWrapper
    bboxes: T.List[BBox]

@dataclass
class DetectionWrapper(BaseWrapper, _DetAttr):

    def copy(self, *, im = None, bboxes = None, parent = None, creator = None):
        kwargs = dict(
            im=im or copy.copy(self.im),
            bboxes=bboxes or copy.deepcopy(self.bboxes),
            parent=parent or self,
            creator=creator,
        )

        return self.__class__(**kwargs)

    def __len__(self):
        return len(self.bboxes)

    def select(self, indices):
        for i, bbox in enumerate(self.bboxes):
            bbox.active = i in indices

    def show(self, ax: T.Optional[plt.Axes] = None, masked: bool = False) -> plt.Axes:
        ax = ax or plt.gca()

        self.im.show(ax, masked=masked)
        ax.set_title(self.creator)

        for bbox in self.bboxes:
            bbox.plot(self.im, ax=ax)

        return ax
