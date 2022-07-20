#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>

#include <opencv2/opencv.hpp>

#include "blob_detector/core.h"

using namespace std;



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

    return 0;
}
