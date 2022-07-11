import numpy as np


def int_tuple(values):
    return tuple(map(int, values))

def get_maxvalue(im: np.ndarray):
    max_value = { dt().dtype.name: value for dt, value in
        [(np.uint8, 255),
        (np.float32, 1.0),
        (np.float64, 1.0)]
    }.get(im.dtype.name)
    assert max_value is not None, f"Unsupported {im.dtype=}"
    return max_value
