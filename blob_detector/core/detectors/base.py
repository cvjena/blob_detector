import cv2
import numpy as np

from scipy import stats

from blob_detector.core import binarizers
from blob_detector import utils

class Detector:

    def __init__(self, *,
                 scale_min_size: int,
                 scale_min_scale: float,

                 pre_equalize: bool,
                 pre_sigma: float,

                 kernel_size: int,
                 dilate_iterations: int,
                 open_close: bool,

                 post_enlarge: float,

                 thresh_type: binarizers.BinarizerType,
                 **thresh_kwargs):
        super().__init__()

        self.scale_min_size = scale_min_size
        self.scale_min_scale = scale_min_scale

        self.sigma = pre_sigma
        self.equalize = pre_equalize

        self.open_close = open_close
        self.dilate_iterations = dilate_iterations
        self.kernel_size = kernel_size

        self.enlarge = post_enlarge

        self.thresholder = binarizers.new_binarizer(thresh_type, **thresh_kwargs)

    def __call__(self, im: np.ndarray):

        im0 = utils._rescale(im, self.scale_min_size, self.scale_min_scale)
        im1 = self.preprocess(im0)
        bin_im = self.thresholder(im1)
        post_bin_im = self.postprocess(bin_im)

        bboxes = self.detect(post_bin_im)

        return (im1, bin_im, post_bin_im), self.postprocess_boxes(im0, bboxes, post_bin_im)

    def detect(self, im: np.ndarray):


        contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # im = np.repeat(np.expand_dims(im, axis=2), 3, axis=2)
        # for cont in contours:
        #     c = random_color()
        #     cv2.drawContours(im, [cont], 0, c, 3)
        #     center = cont.mean(axis=(0,1)).astype(np.int32)
        #     cv2.circle(im, center, radius=10, color=c, thickness=-1)

        # fig, ax = plt.subplots(figsize=(16,9))
        # ax.imshow(im)
        # plt.show()
        # plt.close()


        bboxes = [(
            utils.int_tuple(cont.min(axis=0)[0]),
            utils.int_tuple(cont.max(axis=0)[0])
        ) for cont in contours]
        return bboxes

    def preprocess(self, im: np.ndarray):

        if self.equalize:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
            im = clahe.apply(im)

        if self.sigma < 1:
            return im

        return utils._gaussian(im, sigma=self.sigma)

    def postprocess(self, im: np.ndarray):
        utils._mask_border(im)

        kernel = np.ones((self.kernel_size, self.kernel_size), dtype=im.dtype)

        if self.open_close:
            im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
            im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)

        if self.dilate_iterations >= 1:
            im = cv2.erode(im, kernel, iterations=self.dilate_iterations)
            im = cv2.dilate(im, kernel, iterations=self.dilate_iterations)

        return im

    def postprocess_boxes(self, im: np.ndarray, bboxes: list, bin_im: np.ndarray = None):


        _im = im.astype(np.float64) / 255.

        integral, integral_sq = cv2.integral2(_im)

        im_mean, im_std, im_n = utils._im_mean_std(integral, integral_sq)

        inds = cv2.dnn.NMSBoxes([[x0, y0, x1-x0, y1-y0] for (x0, y0), (x1, y1) in bboxes],
                            np.ones(len(bboxes), dtype=np.float32),
                            score_threshold=0.99,
                            nms_threshold=0.1,
                           )

        inds2 = []
        means_stds = []
        try:
            for i in inds.squeeze():
                bbox = bboxes[i]
                box_mean, box_std, box_n = utils._im_mean_std(integral, integral_sq, bbox)
                ttest_res = stats.ttest_ind_from_stats(im_mean, im_std, im_n, box_mean, box_std, box_n)
                means_stds.append((box_mean, box_std, ttest_res))

                # if ttest_res.statistic < 0:
                #     continue

                #if box_std < 5e-2:
                #    continue

                if not (utils._check_ratio(bbox) and utils._check_area(bbox, im.shape)):
                    continue

                inds2.append(i)
        except:
            pass

        bboxes = utils._enlarge(bboxes, int(self.enlarge * max(im.shape)))
        return bboxes, inds2, means_stds
