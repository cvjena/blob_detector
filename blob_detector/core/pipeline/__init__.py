import numpy as np
import logging
import traceback as tb
import typing as T

from blob_detector import core
from blob_detector.core import bbox_proc
from blob_detector.core import binarizers
from blob_detector.core import img_proc


class Pipeline(object):
    def __init__(self, require_input: T.Optional[T.List[T.Callable]] = None,
                 verbose: bool = False):
        super(Pipeline, self).__init__()
        self._operations = []
        self._require_input = require_input or []
        self._verbose = verbose

    def reset(self):
        self._operations.clear()
        self._require_input.clear()

    def __call__(self, im: np.ndarray):

        for op in self._require_input:
            op(im)

        res = im
        if not isinstance(res, core.ImageWrapper):
            res = core.ImageWrapper(im, creator="Input")

        for op in self._operations:
            name = self.get_op_name(op)

            try:
                res = self.call_op(op, res)
            except Exception as e:
                logging.warning(f"{name} failed: {str(e)}")
                if self._verbose:
                    tb.print_exc()
                res = res.copy()
                name = f"{name}\n(failed: {str(e)})"

            finally:
                if isinstance(res, core.wrapper.base.BaseWrapper):
                    res.creator = name

        return res

    def get_op_name(self, op):

        if hasattr(op, "__name__"):
            return op.__name__

        elif hasattr(op.__class__, "__name__"):
            return op.__class__.__name__

        return "Unnamed"

    def call_op(self, op, res):
        if isinstance(res, (tuple, list)):
            return op(*res)

        elif isinstance(res, (dict)):
            return op(**res)

        else:
            return op(res)


    def add_operation(self, op: T.Callable):
        assert callable(op), f"{op} is not callable!"
        self._operations.append(op)
        return self, op

    def requires_input(self, op: T.Callable):
        assert callable(op), f"{op} is not callable!"
        self._require_input.append(op)
        return self, op


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

    def find_border(self, **kwargs):
        op = img_proc.BorderFinder(**kwargs)
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



