#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>

#include <opencv2/opencv.hpp>

#include "blob_detector/core.h"

namespace blob = blobDet;


void imshow( const std::string &name,
             blob::InputImage im );

void imshow( const std::string &name,
             blob::InputImage im ,
             blob::InputImage border,
             const double alpha = 0.5 );

void showBoxes( const std::string &name,
                blob::OutputImage im,
                const blob::BBoxes &boxes,
                const cv::Scalar &color,
                int thickness = 1,
                int lineType = cv::LINE_AA );

void showBoxes( const std::string &name,
                blob::OutputImage im,
                const blob::BBoxes &boxes,
                const std::vector<int> &indices,
                const cv::Scalar &color,
                int thickness = 1,
                int lineType = cv::LINE_AA );

int waitKey( float timer = 0.1 );


void imshow( const std::string &name, blob::InputImage im ){
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    cv::imshow(name, im);
}

void imshow( const std::string &name, blob::InputImage im, blob::InputImage border, const double alpha ){
    cv::Mat _im = alpha * im + (1-alpha) * border;
    imshow(name, _im);
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

void showBoxes( const std::string &name,
                blob::OutputImage im,
                const blob::BBoxes &boxes,
                const cv::Scalar& color,
                int thickness,
                int lineType )
{
    cv::namedWindow(name, cv::WINDOW_NORMAL);

    for (blob::BBox box: boxes)
        box.draw(im, color, thickness, lineType);

    cv::imshow(name, im);

}

void showBoxes( const std::string &name,
                blob::OutputImage im,
                const blob::BBoxes &boxes,
                const std::vector<int> &indices,
                const cv::Scalar& color,
                int thickness,
                int lineType )
{
    cv::namedWindow(name, cv::WINDOW_NORMAL);

    for (int i: indices)
        boxes[i].draw(im, color, thickness, lineType);

    cv::imshow(name, im);

}



int main(int argc, char** argv)
{
    if ( argc != 2 )
    {
        printf("usage: %s <Image_Path>\n", argv[0]);
        return -1;
    }
    cv::Mat image, foo;
    image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    cv::Mat gray, gray_scaled, gray_processed, bin_im;

    // 1. BGR -> Gray
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    // imshow("BW original", gray);

    cv::Mat border;
    // 2. create border mask
    blob::findBorder(gray, border);
    foo = 0.5 * gray + 0.5 * border;
    // imshow("BW with Border", foo);

    gray_scaled = gray.clone();

    // 2a rescale both
    const int minSize = 1080;
    blob::rescale(gray_scaled, minSize);
    blob::rescale(border, minSize);


    // 3. gaussian blur and optional histogram equalization
    const double sigma = 5.0;
    const bool equalize = false;
    blob::preprocess(gray_scaled, gray_processed, sigma, equalize );
    // imshow("Processed with Border", gray_processed);

    // 4. binarization (including border information)
    blob::binarize(gray_processed, bin_im, border);
    // imshow("binarized with border", bin_im);

    // 4.5 [optional] morphological operations
    // (set parameters to -1 to disable this operation)
    const int kernelSize = 5, iterations = 2;
    blob::openClose(bin_im, kernelSize, iterations);
    // imshow("after morph", bin_im);

    std::vector<blob::BBox> boxes;
    // 5. Detect bounding boxes (including border information)
    blob::detect(bin_im, border, boxes);
    foo = bin_im.clone();
    // showBoxes("Detection", foo, boxes, cv::Scalar(127, 0, 0), 1);

    std::vector<blob::BBox> boxes2;
    // 6. Try to split bounding boxes
    blob::splitBoxes(gray, boxes, boxes2);
    foo = bin_im.clone();
    // showBoxes("After Split", foo, boxes2, cv::Scalar(127, 0, 0), 1);

    // 7.1. Filter bounding boxes: NMS
    std::vector<int> indices(boxes2.size());
    const double score_threshold = 0.5, nms_threshold = 0.3;
    blob::nmsBoxes(boxes2, indices, score_threshold, nms_threshold);
    foo = bin_im.clone();
    // showBoxes("After NMS", foo, boxes2, indices, cv::Scalar(127, 0, 0), 1);

    // 7.2. Filter bounding boxes: Validation checks
    indices.erase(remove_if(
        indices.begin(), indices.end(),
        [&boxes2](int i) { return !boxes2[i].isValid(); }
    ), indices.end());
    foo = bin_im.clone();
    // showBoxes("After Validation", foo, boxes2, indices, cv::Scalar(127, 0, 0), 1);

    // 8. Enlarge boxes
    const double enlarge = 0.01;
    for (int i : indices)
        boxes2[i].enlarge(enlarge);

    // 9. Score boxes
    for (int i : indices)
        boxes2[i].setScore(gray);


    showBoxes("Final", image, boxes2, indices, cv::Scalar(127, 0, 0), 1);

    return waitKey();

}
