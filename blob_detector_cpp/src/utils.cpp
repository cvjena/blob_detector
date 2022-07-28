
#include <opencv2/opencv.hpp>

#include "blob_detector/core.h"

using namespace std;

namespace blobDet {

void gaussian( InputImage image,
               OutputImage output,
               double sigma )
{
    cv::GaussianBlur(image, output, cv::Size(0, 0), sigma);
}

int scaledThicknes( int thickness,
                    InputImage im,
                    int min_size )
{

    double thickness_scaler = min(im.rows, im.cols) / min_size;
    return thickness * max(1.0, thickness_scaler);
}

void correlate( InputImage image,
                InputImage kernel,
                OutputImage corr,
                bool normalize )
{
    cv::Mat _image, _image_padded, _kernel;

    image.convertTo(_image, CV_32F);
    kernel.convertTo(_kernel, CV_32F);

    // 0..255 -> 0..1
    _image /= 255.0;
    _kernel /= 255.0;

    // subtract average mean
    auto mean = (cv::mean(_image) + cv::mean(_kernel)) / 2;
    _image -= mean;
    _kernel -= mean;

    // add proper padding
    int k_w = _kernel.cols,
        k_h = _kernel.rows;

    _image_padded = cv::Mat(
        _image.rows + k_h,
        _image.cols + k_w,
        _image.depth());

    cv::copyMakeBorder(_image, _image_padded,
                       k_h/2, k_h/2, k_w/2, k_w/2,
                       cv::BORDER_REFLECT,
                       cv::Scalar(0));


    // perform correlation operation
    cv::matchTemplate(_image_padded, _kernel, corr, cv::TM_CCORR_NORMED);

    // normalize if needed
    if (normalize)
    {
        double px_min, px_max;
        cv::minMaxLoc( corr, &px_min, 0);
        corr -= px_min;
        cv::minMaxLoc( corr, 0, &px_max);
        corr /= px_max;
    }

}


void putText( OutputImage image,
              const string &text,
              const cv::Point2d &pos,
              const cv::Scalar &color,
              int thickness,
              int lineType )
{

    const int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
    const float fontScale = max(1.0, min(image.cols, image.rows) / (2 * 720.0) );
    const int textThickness = thickness / 3 * 2;
    const int offset = 5 + textThickness;
    int baseline = 0;

    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, textThickness, &baseline);
    baseline += textThickness;

    const int marginSize = offset + baseline;

    cv::Point textOrigin(image.cols * pos.x, image.rows * pos.y);
    cv::Point upperLeft = textOrigin + cv::Point(
        textSize.width + 2*marginSize,
        -textSize.height - 2*marginSize);


    cv::rectangle(image, textOrigin, upperLeft, color, thickness, lineType);

    // cv::circle(image, textOrigin, 10, cv::Scalar(0, 0, 255), -1, lineType);
    // cv::circle(image, upperLeft, 10, cv::Scalar(0, 0, 255), -1, lineType);

    cv::putText(image,
                text,
                textOrigin + cv::Point(marginSize, -marginSize),
                fontFace,
                fontScale,
                color,
                textThickness,
                lineType);

}


bool compareContours( vector<cv::Point> c1, vector<cv::Point> c2 )
{
    return fabs(cv::contourArea(cv::Mat(c1))) > fabs(cv::contourArea(cv::Mat(c2)));
}

} // blobDet
