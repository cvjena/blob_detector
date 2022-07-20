
#include <opencv2/opencv.hpp>

#include "blob_detector/core.h"

using namespace std;

void gaussian( InputImage image,
               OutputImage output,
               double sigma )
{
    cv::GaussianBlur(image, output, cv::Size(0, 0), sigma);
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
    int k_w = _kernel.size().width,
        k_h = _kernel.size().height;

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


void imshow( const string &name, InputImage im ){
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    cv::imshow(name, im);
}

void putText( OutputImage image,
              const string &text,
              const cv::Point2d &pos,
              const cv::Scalar &color,
              int thickness,
              int lineType )
{

    const int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
    const float fontScale = 0.6;
    const int offset = 5;
    int baseline = 0;

    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;


    cv::Point textOrigin(
        (image.cols * pos.x) + offset,
        (image.rows * pos.y) - baseline - offset);

    cv::rectangle(image,
                  textOrigin + cv::Point(-offset, baseline+offset),
                  textOrigin + cv::Point(textSize.width+offset, -textSize.height-offset),
                  color, thickness, lineType);

    cv::putText(image, text, textOrigin, fontFace, fontScale, color, thickness, lineType);

}

int waitKey( float timer )
{

    for (;;)
    {
        char key = (char) cv::waitKey(timer);
        if ( key == 'q' || key == 'Q' || key == 27)
            break;
    }

    cv::destroyAllWindows();
    return 0;
}

void showBoxes( const string &name,
                OutputImage im,
                const BBoxes &boxes,
                const cv::Scalar& color,
                int thickness,
                int lineType )
{
    cv::namedWindow(name, cv::WINDOW_NORMAL);

    for (BBox box: boxes)
        box.draw(im, color, thickness, lineType);

    cv::imshow(name, im);

}

void showBoxes( const string &name,
                OutputImage im,
                const BBoxes &boxes,
                const vector<int> &indices,
                const cv::Scalar& color,
                int thickness,
                int lineType )
{
    cv::namedWindow(name, cv::WINDOW_NORMAL);

    for (int i: indices)
        boxes[i].draw(im, color, thickness, lineType);

    cv::imshow(name, im);

}

bool compareContours( vector<cv::Point> c1, vector<cv::Point> c2 )
{
    return fabs(cv::contourArea(cv::Mat(c1))) > fabs(cv::contourArea(cv::Mat(c2)));
}

