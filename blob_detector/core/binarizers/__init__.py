import enum

from blob_detector.core.binarizers.gauss_local import GaussLocalTresholder
from blob_detector.core.binarizers.high_pass import HighPassTresholder
from blob_detector.core.binarizers.otsu import OtsuTresholder

class BinarizerType(enum.Enum):

    gauss_local = GaussLocalTresholder
    high_pass = HighPassTresholder
    otsu = OtsuTresholder


def new(type: BinarizerType, *args, **kwargs):
    assert isinstance(type, BinarizerType), \
        f"unknown binarizer type: {type}"

    return type.value(*args, **kwargs)
