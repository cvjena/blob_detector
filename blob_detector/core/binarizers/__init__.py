import enum

from blob_detector.core.binarizers.gauss_local import GaussLocalTresholder
from blob_detector.core.binarizers.high_pass import HighPassTresholder
from blob_detector.core.binarizers.otsu import OtsuTresholder

class BinarizerType(enum.Enum):

    gauss_local = GaussLocalTresholder
    high_pass = HighPassTresholder
    otsu = OtsuTresholder


def new_binarizer(bin_type: BinarizerType, *args, **kwargs):
    assert isinstance(bin_type, BinarizerType), \
        f"unknown binarizer type: {bin_type}"

    return bin_type.value(*args, **kwargs)
