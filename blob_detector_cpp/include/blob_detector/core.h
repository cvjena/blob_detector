#pragma once

#include <opencv2/opencv.hpp>

#define MIN_AREA 4e-4
#define MAX_AREA 1/9.0
#define VALID_RATIO 0.1

typedef const cv::Mat& InputImage;
typedef cv::Mat& OutputImage;


#include "blob_detector/bbox.h"
#include "blob_detector/utils.h"
#include "blob_detector/operations.h"
