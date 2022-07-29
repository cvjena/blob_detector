#include <opencv2/opencv.hpp>

#include "blob_detector/core.h"

using namespace std;

namespace blobDet {

void rescale( OutputImage image, const int min_size, const double min_scale )
{
    const double min_im_size = std::min(image.cols, image.rows);
    double scale = min_size / min_im_size;
    scale = std::max(min_scale, std::min(1.0, scale));

    cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_LINEAR);
}

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
        border_mask = border_mask(
            cv::Range(pad, border_mask.rows - pad),
            cv::Range(pad, border_mask.cols - pad));
    }

}

void preprocess( InputImage image, OutputImage output, double sigma, bool equalize )
{

    if ( equalize )
        cv::createCLAHE(2.0, cv::Size(10, 10))->apply(image, output);
    else
        output = image.clone();

    gaussian(output, output, sigma);

}

void binarize( InputImage image, OutputImage output, int windowSize, float C)
{
    auto maxValue = 255;

    if ( windowSize < 1 )
        cv::threshold(image, output,
            0, maxValue,
            cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    else
        cv::adaptiveThreshold(image, output,
            maxValue,
            cv::ADAPTIVE_THRESH_MEAN_C,
            cv::THRESH_BINARY_INV,
            windowSize, C);
}
void binarize( InputImage image, OutputImage output, InputImage mask, int windowSize, float C)
{
    cv::Mat masked = cv::Mat::zeros( image.size(), image.type() );
    image.copyTo(masked, mask);
    binarize(masked, output, windowSize, C);
    output.copyTo(output, mask);
}

void openClose( OutputImage img, int kernelSize, int iterations)
{
    cv::Mat kernel = cv::Mat::ones(kernelSize, kernelSize, img.type());

    if (kernelSize >= 1){
        cv::morphologyEx(img, img, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(img, img, cv::MORPH_CLOSE, kernel);
    }

    if (iterations >= 1){
        cv::erode(img, img, kernel, cv::Point(-1, -1), iterations);
        cv::dilate(img, img, kernel, cv::Point(-1, -1), iterations);
    }

}

} // blobDet
