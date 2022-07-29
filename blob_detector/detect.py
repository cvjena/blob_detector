#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

try:
    from cvargparse import Arg
    from cvargparse import BaseParser
except ImportError as e:
    print("Please install cvargparse to use this script:")
    print("pip install cvargparse~=0.5")
    exit()

try:
    import skimage
except ImportError as e:
    print("Please install scikit-image to use this script:")
    print("pip install scikit-image")
    exit()


import cv2
import numpy as np

from matplotlib import pyplot as plt

from blob_detector import utils
from blob_detector.core.bbox import BBox
from blob_detector.core.bbox_proc import Splitter
from blob_detector.core.binarizers import BinarizerType
from blob_detector.core.pipeline import Pipeline


def main(args):
    img_proc = Pipeline()
    img_proc.find_border()
    if args.min_size > 0:
        img_proc.rescale(min_size=args.min_size, min_scale=0.1)

    img_proc.preprocess(equalize=False, sigma=args.sigma)
    img_proc.binarize(
        type=BinarizerType.gauss_local,
        use_masked=True,
        use_cv2=True,
        window_size=args.window_size,
        offset=args.C,
        )

    img_proc.remove_border()
    img_proc.open_close(
        kernel_size=args.morph_kernel,
        iterations=args.morph_iters)

    bbox_proc = Pipeline()
    bbox_proc.detect(use_masked=True)

    _, splitter = bbox_proc.split_bboxes(
        preproc=Pipeline(), detector=Pipeline())

    _, bbox_filter = bbox_proc.bbox_filter(
        score_threshold=0.5,
        nms_threshold=0.3,
        enlarge=args.enlarge,
    )
    _, scorer = bbox_proc.score()

    img_proc.requires_input(splitter.set_image)
    img_proc.requires_input(bbox_filter.set_image)
    img_proc.requires_input(scorer.set_image)

    im = cv2.imread(args.file_path, cv2.IMREAD_COLOR)
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    res = img_proc(gray_im)

    if args.show_intermediate:
        utils.show_intermediate(res, masked=args.show_masked, separate=args.show_separate)

    detections = bbox_proc(res)

    if args.show_intermediate:
        utils.show_intermediate(detections, masked=args.show_masked, separate=args.show_separate)

    fig, ax0 = plt.subplots()
    ax0.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    for bbox in detections.bboxes:
        if not bbox.active:
            continue
        bbox.plot(im, ax=ax0, edgecolor="blue")

    plt.show()
    plt.close()



parser = BaseParser([
    Arg("file_path"),

    Arg.int("--min_size", "-size", default=1080),
    Arg.int("--C", "-C", default=2),
    Arg.int("--window_size", "-ws", default=31),
    Arg.float("--sigma", "-sig", default=5.0),
    Arg.int("--morph_kernel", "-mk", default=5),
    Arg.int("--morph_iters", "-mi", default=2),
    Arg.float("--enlarge", default=0.01),

    Arg.flag("--show_intermediate", "-intermediate"),
    Arg.flag("--show_masked", "-masked"),
    Arg.flag("--show_separate", "-separate"),

])

main(parser.parse_args())



