#include <opencv2/opencv.hpp>

#include "blob_detector/core.h"

using namespace std;

void findBorder( InputImage image, OutputImage border_mask, double threshold, int pad )
{
    cv::Mat padded, bin_im;
    vector<vector<cv::Point> > contours;

    if ( pad >= 1 )
    {
        cv::copyMakeBorder(image, padded,
                           pad, pad, pad, pad,
                           cv::BORDER_CONSTANT,
                           cv::Scalar(0));
    } else
        padded = image.clone();

    cv::threshold(padded, bin_im, threshold, 255.0, cv::THRESH_BINARY);


    cv::findContours(bin_im, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    border_mask = cv::Mat::zeros(padded.size(), padded.type());

    sort(contours.begin(), contours.end(), compareContours);

    // cv::drawContours(border_mask, contours, 0, cv::Scalar(190), 2, cv::LINE_AA);
    cv::approxPolyDP(cv::Mat(contours[0]), contours[0], 100, true);

    // cv::drawContours(border_mask, contours, largestIdx, cv::Scalar(127), -1, cv::LINE_AA);
    cv::drawContours(border_mask, contours, 0, cv::Scalar(255), -1, cv::LINE_AA);

    if ( pad >= 1 ){
        auto size = border_mask.size();
        border_mask = border_mask(
            cv::Range(pad, size.height - pad),
            cv::Range(pad, size.width - pad));
    }

}

void preprocess( InputImage image, OutputImage output, double sigma, bool equalize){

    if ( equalize )
        cv::createCLAHE(2.0, cv::Size(10, 10))->apply(image, output);
    else
        output = image.clone();

    gaussian(image, output, sigma);

}

void binarize( InputImage image, OutputImage output, int window_size, float C)
{
    auto maxValue = 255;
    cv::adaptiveThreshold(image, output,
        maxValue,
        cv::ADAPTIVE_THRESH_MEAN_C,
        cv::THRESH_BINARY_INV,
        window_size, C);
}
void binarize( InputImage image, OutputImage output, InputImage mask, int window_size, float C)
{
    cv::Mat masked = cv::Mat::zeros( image.size(), image.type() );
    image.copyTo(masked, mask);
    binarize(masked, output, window_size, C);
    output.copyTo(output, mask);
}

void open_close( OutputImage img, int kernel_size, int iterations)
{
    cv::Mat kernel = cv::Mat::ones(kernel_size, kernel_size, img.type());

    if (kernel_size >= 1){
        cv::morphologyEx(img, img, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(img, img, cv::MORPH_CLOSE, kernel);
    }

    if (iterations >= 1){
        cv::erode(img, img, kernel, cv::Point(-1, -1), iterations);
        cv::dilate(img, img, kernel, cv::Point(-1, -1), iterations);
    }

}
