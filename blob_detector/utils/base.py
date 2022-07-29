import numpy as np

from matplotlib import pyplot as plt


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

def show_intermediate(result,
                      masked: bool = False,
                      separate: bool = False):
    ims = []
    while result is not None:
        ims.append(result)
        result = result.parent

    if separate:
        for im in reversed(ims):
            fig, ax = plt.subplots()
            im.show(ax, masked=masked)

    else:
        rows = int(np.ceil(np.sqrt(len(ims))))
        cols = int(np.ceil(len(ims) / rows))

        fig, axs = plt.subplots(rows, cols, squeeze=False)

        for i, im in enumerate(reversed(ims)):
            ax = axs[np.unravel_index(i, axs.shape)]
            im.show(ax, masked=masked)

        for _ in range(i, rows*cols):
            axs[np.unravel_index(_, axs.shape)].axis("off")

    plt.tight_layout()
