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

            elif isinstance(res, (dict)):
                res = op(**res)

            else:
                res = op(res)

            results.append(res)

        if return_all:
            return results

        else:
            return results[-1]


    def rescale(self, **kwargs):
        op = img_proc.Rescaler(**kwargs)
        return self.add_operation(op)

    def preprocess(self, **kwargs):
        op = img_proc.Preprocessor(**kwargs)
        return self.add_operation(op)

    def binarize(self, **kwargs):
        op = binarizers.new(**kwargs)
        return self.add_operation(op)

    def open_close(self, **kwargs):
        op = img_proc.MorphologicalOps(**kwargs)
        return self.add_operation(op)

    def remove_border(self, **kwargs):
        op = img_proc.BorderRemoval(**kwargs)
        return self.add_operation(op)

    def detect(self, **kwargs):
        op = bbox_proc.Detector(**kwargs)
        return self.add_operation(op)

    def bbox_filter(self, **kwargs):
        op = bbox_proc.BBoxFilter(**kwargs)
        return self.add_operation(op)

    def split_bboxes(self, **kwargs):
        op = bbox_proc.Splitter(**kwargs)
        return self.add_operation(op)

    def score(self, **kwargs):
        op = bbox_proc.ScoreEstimator(**kwargs)
        return self.add_operation(op)

    def add_operation(self, op):
        assert callable(op), f"{op} was not callable!"
        self._operations.append(op)
        return self, op



