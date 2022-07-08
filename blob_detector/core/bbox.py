from __future__ import annotations

import numpy as np
import typing as T

try:
    from matplotlib import pyplot as plt
    HAS_PYPLOT, PYPLOT_ERROR = True, None
except ImportError as e:
    HAS_PYPLOT, PYPLOT_ERROR = False, e


class BBox(T.NamedTuple):
    VALID_RATIO = 0.1
    MIN_AREA = 4e-4
    MAX_AREA = 1/9

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self):
        return abs(self.x1 - self.x0)

    @property
    def h(self):
        return abs(self.y1 - self.y0)

    @property
    def size(self):
        return (self.w, self.h)

    @property
    def origin(self):
        return (self.x0, self.y0)

    @property
    def area(self):
        return self.h * self.w

    @property
    def ratio(self):
        return min(self.size) / max(self.size)

    def _new(self, x0, y0, x1, y1) -> BBox:
        return self._replace(
            x0=x0, y0=y0,
            x1=x1, y1=y1,
        )

    def __check_xy(self, xy) -> T.Tuple[float, float]:
        xy: T.Union[T.Tuple, T.List, float, int]
        if isinstance(xy, (tuple, list)):
            x, y = xy

        elif isinstance(xy, (int, float)):
            x = y = xy

        else:
            raise TypeError(f"Unsupported argument type: {type(xy)=}")

        return x, y

    def __sub__(self, xy: T.Union[T.Tuple, T.List, float, int]) -> BBox:
        """ translate the bounding box by x and y """
        x, y = self.__check_xy(xy)

        return self._new(
            x0=self.x0 - x, y0=self.y0 - y,
            x1=self.x1 - x, y1=self.y1 - y,
        )

    __rsub__ = __sub__

    def __add__(self, xy: T.Union[T.Tuple, T.List, float, int]) -> BBox:
        """ translate the bounding box by x and y """
        x, y = self.__check_xy(xy)

        return self._new(
            x0=self.x0 + x, y0=self.y0 + y,
            x1=self.x1 + x, y1=self.y1 + y,
        )

    __radd__ = __add__

    def __mul__(self, xy: T.Union[T.Tuple, T.List, float, int]) -> BBox:
        """ scale the bounding box by x and y """
        x, y = self.__check_xy(xy)

        return self._new(
            x0=self.x0 * x, y0=self.y0 * y,
            x1=self.x1 * x, y1=self.y1 * y,
        )

    __rmul__ = __mul__

    def __truediv__(self, xy: T.Union[T.Tuple, T.List, float, int]) -> BBox:
        """ scale the bounding box by x and y """
        x, y = self.__check_xy(xy)

        return self._new(
            x0=self.x0 / x, y0=self.y0 / y,
            x1=self.x1 / x, y1=self.y1 / y,
        )

    __rtruediv__ = __truediv__

    def enlarge(self, xy: T.Union[T.Tuple, T.List, float, int]) -> BBox:
        """ enlarge the bounding box by x and y """
        x, y = self.__check_xy(xy)
        if x <= 0 and y <= 0:
            return self

        return self._new(
            x0=self.x0 - x, y0=self.y0 - y,
            x1=self.x1 + x, y1=self.y1 + y,
        )

    def __call__(self, im: np.ndarray):
        """
            translates the relative coordinates to pixel
            coordinates for the given image
        """

        x0, y0, x1, y1 = self
        H, W, *_ = im.shape

        # translate from relative coordinates to pixel
        # coordinates for the given image

        x0, x1 = max(int(x0 * W), 0), min(int(x1 * W), W)
        y0, y1 = max(int(y0 * H), 0), min(int(y1 * H), H)

        return x0, y0, x1, y1

    @property
    def is_valid(self):
        return self.ratio >= BBox.VALID_RATIO and \
            BBox.MIN_AREA <= self.area <= BBox.MAX_AREA

    def crop(self, im: np.ndarray, enlarge: bool = True):

        if im.ndim not in [2, 3]:
            ValueError(f"Unsupported ndims: {im.ndims=}")
        x0, y0, x1, y1 = self(im)
        H, W, *_ = im.shape

        # enlarge to a square extent
        if enlarge:
            h, w = int(self.h * H), int(self.w * W)
            size = max(h, w)
            dw, dh = (size - w) / 2, (size - h) / 2
            x0, y0 = max(int(x0 - dw), 0), max(int(y0 - dh), 0)
            x1, y1 = int(x0 + size), int(y0 + size)

        return im[y0:y1, x0:x1]


    def plot(self, im: np.ndarray, ax: T.Optional[plt.Axis] = None, **kwargs) -> plt.Axis:
        global HAS_PYPLOT, PYPLOT_ERROR
        assert HAS_PYPLOT, f"Could not import pyplot: {PYPLOT_ERROR}"

        h, w, *_ = im.shape
        box = self * (w, h)
        rect = plt.Rectangle(box.origin, *box.size, fill=False, **kwargs)

        ax = ax or plt.gca()
        ax.add_patch(rect)
        return ax

    def tiles(self, nx: int, ny: int) -> T.List[BBox]:

        tile_w = self.w / nx
        tile_h = self.h / ny

        tiles = []
        for x0 in np.arange(self.x0, self.x1, tile_w):
            for y0 in np.arange(self.y0, self.y1, tile_h):
                x1, y1 = x0 + tile_w, y0 + tile_h
                tiles.append(BBox(x0, y0, x1, y1))

        return tiles

