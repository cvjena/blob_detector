import numpy as np

from blob_detector.core import img_proc
from blob_detector.core import bbox_proc
from blob_detector.core import binarizers


class Pipeline(object):
    def __init__(self):
        super(Pipeline, self).__init__()
        self._operations = []

    def reset(self):
        self._operations.clear()

    def __call__(self, im: np.ndarray, return_all: bool = False):

        results = []

        res = im
        for op in self._operations:
            if isinstance(res, (tuple, list)):
                res = op(*res)

            elif isinstance(im, (dict)):
                res = op(**res)

            else:
                res = op(res)

            results.append(res)

        if return_all:
            return results

        else:
            return results[-1]


    def rescale(self, **kwargs):
        self._operations.append(img_proc.Rescaler(**kwargs))
        return self

    def preprocess(self, **kwargs):
        self._operations.append(img_proc.Preprocessor(**kwargs))
        return self

    def binarize(self, **kwargs):
        self._operations.append(binarizers.new(**kwargs))
        return self

    def open_close(self, **kwargs):
        self._operations.append(img_proc.MorphologicalOps(**kwargs))
        return self

    def detect(self, **kwargs):
        self._operations.append(bbox_proc.Detector(**kwargs))
        return self

    def bbox_filter(self, **kwargs):
        self._operations.append(bbox_proc.BBoxFilter(**kwargs))
        return self


