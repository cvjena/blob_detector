import cv2
import matplotlib.pyplot as plt
import numpy as np
import typing as T

from blob_detector import utils
from blob_detector.core.bbox import BBox
from blob_detector.core.bbox_proc.base import ImageSetter


VIS = False
class ScoreEstimator(ImageSetter):


    def __call__(self, im: np.ndarray, bboxes: T.List[BBox], idxs=None, bbox_stats=None):
        global VIS
        self._check_image()

        if idxs is None:
            idxs = np.arange(len(bboxes))

        # return [bboxes[i] for i in idxs], None, None

        if VIS:
            fig0, ax0 = plt.subplots()
            ax0.imshow(self._im, cmap=plt.cm.gray)

            n_cols = int(np.ceil(np.sqrt(len(idxs))))
            n_rows = int(np.ceil(len(idxs) / n_cols))
            fig, axs = plt.subplots(n_rows, n_cols*2, squeeze=False)

            for idx, bbox in enumerate(bboxes):
                if idx in idxs:
                    continue
                bbox.plot(self._im, ax=ax0, edgecolor="blue")

        scores = []
        labels = None
        for c, idx in enumerate(idxs):
            bbox = bboxes[idx]
            crop = bbox.crop(self._im, enlarge=False)

            if VIS:
                bbox.plot(self._im, ax=ax0)

                ax = axs[np.unravel_index(c*2, axs.shape)]
                ax.imshow(crop, cmap=plt.cm.gray, alpha=0.7)
                # ax.imshow(self_corr, cmap=plt.cm.jet, alpha=0.3)

            h, w, *_ = crop.shape
            H, W, *_ = self._im.shape

            # enlarge in each direction by the bbox size, resulting the tripple extent
            offset_bbox = bbox.enlarge((w / W, h / H))
            offset_crop = offset_bbox.crop(self._im, enlarge=False).copy()
            corr = _correlate(offset_crop, crop, normalize=True)
            score = _score(corr)
            scores.append(score)

            if VIS:
                offset_bbox.plot(self._im, ax=ax0, edgecolor="gray")

                ax1 = axs[np.unravel_index(c*2 +1, axs.shape)]
                ax1.imshow(offset_crop, alpha=0.5, cmap=plt.cm.gray)
                ax1.imshow(corr, alpha=0.5, cmap=plt.cm.jet)
                ax1.set_title(f"Score: {score:.3f}")



        if VIS:
            plt.show()
            plt.close()

        return [bboxes[i] for i in idxs], labels, scores


def _correlate(im1, im2, normalize=True, boundary="symm", mode="same", **kwargs):

    # 0..255 -> 0..1
    im1 = im1.astype(np.float32) / 255
    im2 = im2.astype(np.float32) / 255

    mean = (im1.mean() + im2.mean()) / 2

    im1 = im1 - mean
    im2 = im2 - mean

    ##### OpenCV implementation
    k_h, k_w, *_ = im2.shape

    _im = np.pad(im1, [(k_h//2, k_h//2-1), (k_w//2, k_w//2-1)], mode="reflect")
    corr = cv2.matchTemplate(_im, im2, cv2.TM_CCORR)

    if not normalize:
        return corr

    corr -= corr.min()
    return corr / corr.max()


def _score(patch):

    middle_mean, adjacent_mean = 0, 0

    for i, tile in enumerate(BBox(0, 0, 1, 1).tiles(3,3)):
        coords = np.unravel_index(i, (3,3))
        if coords == (1, 1):
            middle_mean = tile.crop(patch, enlarge=False).mean()
        else:
            adjacent_mean += tile.crop(patch, enlarge=False).mean() / 8

    return 1 - adjacent_mean / middle_mean
