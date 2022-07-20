#include <opencv2/opencv.hpp>

#include "blob_detector/core.h"

using namespace std;

void detect(InputImage image, InputImage mask, BBoxes &boxes)
{
    cv::Mat masked = cv::Mat::zeros( image.size(), image.type() );
    image.copyTo(masked, mask);
    detect(masked, boxes);
}

void detect(InputImage image, BBoxes &boxes)
{
    vector<vector<cv::Point> > contours;

    cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    sort(contours.begin(), contours.end(), compareContours);
    boxes.resize(contours.size());

    for (int i = 0; i < contours.size(); ++i)
        boxes[i] = BBox(contours[i], image.size());

}

void splitBoxes(InputImage image, const BBoxes &boxes, BBoxes &outputBoxes)
{

    cv::Mat crop, crop_processed, bin_crop;
    vector<BBox> newBoxes;
    for (BBox bbox : boxes){
        outputBoxes.push_back(bbox);
        if (!bbox.splittable(image.size()))
            continue;


        bbox.crop(image, crop);

        preprocess(crop, crop_processed, 2.0, true);
        binarize(crop_processed, bin_crop, 15);
        open_close(bin_crop, 5.0, 2);

        newBoxes.clear();

        detect(bin_crop, newBoxes);


        for (auto _box: newBoxes)
            outputBoxes.push_back(_box.rescale(bbox));
    }

}

void nmsBoxes( BBoxes &boxes,
               vector<int> &indices,
               const float score_threshold,
               const float nms_threshold)
{

    vector<cv::Rect2d> _boxes(boxes.size());
    vector<float> scores(boxes.size(), 1.0);

    int i = 0;
    for (BBox box : boxes)
        _boxes[i++] = box.asRect();

    cv::dnn::NMSBoxes(
        _boxes,
        scores,
        score_threshold,
        nms_threshold,
        indices);
}
