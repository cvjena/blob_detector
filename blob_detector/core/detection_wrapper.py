from __future__ import annotations

import numpy as np
import typing as T

from dataclasses import dataclass
from blob_detector.core import ImageWrapper
from blob_detector.core.bbox import BBox

@dataclass
class DetectionWrapper:
    im: ImageWrapper
    bboxes: T.List[BBox]

    labels: T.Optional[T.List[int]] = None
    indices: T.Optional[T.List[int]] = None
    scores: T.Optional[T.List[float]] = None

    parent: T.Optional[DetectionWrapper] = None
    creator: T.Optional[str] = None

    def __post_init__(self):
        if self.indices is None:
            self.indices = np.arange(len(self.bboxes))

        if self.scores is None or len(self.scores) != len(self.bboxes):
            self.scores = np.zeros(len(self.bboxes), dtype=np.float32)


    def copy(self, **kwargs):
        for attr in ["im", "bboxes", "indices", "score", "labels"]:
            if attr not in kwargs and getattr(self, attr, None) is not None:
                kwargs[attr] = getattr(self, attr)

        return self.__class__(parent=self, **kwargs)

    def show(self, ax: T.Optional[plt.Axes] = None, masked: bool = False) -> plt.Axes:
        ax = ax or plt.gca()

        self.im.show(ax, masked=masked)
        ax.set_title(self.creator)

        for i, (bbox, score) in enumerate(zip(self.bboxes, self.scores)):
            if i in self.indices:
                color = "blue"
            elif self.parent is not None and i in self.parent.indices:
                color = "red"
            else:
                continue

            bbox.plot(self.im, score=score, ax=ax, edgecolor=color)


        return ax
