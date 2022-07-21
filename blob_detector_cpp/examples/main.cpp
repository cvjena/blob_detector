#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>

#include <opencv2/opencv.hpp>

#include "blob_detector/core.h"

using namespace std;


void imshow( const std::string &name,
             InputImage im );

void showBoxes( const std::string &name,
                OutputImage im,
                const BBoxes &boxes,
                const cv::Scalar &color,
                int thickness = 1,
                int lineType = cv::LINE_AA );

void showBoxes( const std::string &name,
                OutputImage im,
                const BBoxes &boxes,
                const std::vector<int> &indices,
                const cv::Scalar &color,
                int thickness = 1,
                int lineType = cv::LINE_AA );

int waitKey( float timer = 0.1 );


void imshow( const string &name, InputImage im ){
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    cv::imshow(name, im);
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



int main(int argc, char** argv)
{
    if ( argc != 2 )
    {
        printf("usage: %s <Image_Path>\n", argv[0]);
        return -1;
    }
    cv::Mat image;
    image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    cv::Mat gray, gray_processed, bin_im;

    // 1. BGR -> Gray
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat border;
    // 2. create border mask
    findBorder(gray, border);

    // 3. gaussian blur and optional histogram equalization
    preprocess(gray, gray_processed /* sigma, equalize*/ );

    // 4. binarization (including border information)
    binarize(gray_processed, bin_im, border);

    // 4.5 [optional] morphological operations
    // (set parameters to -1 to disable this operation)
    open_close(bin_im, 5, 2);

    vector<BBox> boxes;
    // 5. Detect bounding boxes (including border information)
    detect(bin_im, border, boxes);

    vector<BBox> boxes2;
    // 6. Try to split bounding boxes
    splitBoxes(gray, boxes, boxes2);

    // 7.1. Filter bounding boxes: NMS
    vector<int> indices(boxes2.size());
    nmsBoxes(boxes2, indices, 0.5, 0.3);

    // 7.2. Filter bounding boxes: Validation checks
    indices.erase(remove_if(
        indices.begin(), indices.end(),
        [&boxes2](int i) { return !boxes2[i].isValid(); }
    ), indices.end());

    // 8. Enlarge boxes
    for (int i : indices)
        boxes2[i].enlarge(0.01);

    // 9. Score boxes
    for (int i : indices)
        boxes2[i].setScore(gray);


    showBoxes("Final", image, boxes2, indices, cv::Scalar(255, 0, 0), 2);

    return waitKey();
    //return 0;
}
