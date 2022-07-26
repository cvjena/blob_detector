#pragma once
#include <opencv2/opencv.hpp>

#include "blob_detector/core.h"

namespace blobDet {

bool compareContours( std::vector<cv::Point> c1,
                      std::vector<cv::Point> c2 );

void gaussian( InputImage image,
               OutputImage output,
               double sigma );

void correlate( InputImage im1,
                InputImage im2,
                OutputImage corr,
                bool normalize = true );

void putText( OutputImage image,
              const std::string &text,
              const cv::Point2d &pos,
              const cv::Scalar &color = cv::Scalar(0),
              int thickness = 1,
              int lineType = cv::LINE_AA,
              const double boxAlpha = 0.3 );

int scaledThicknes( int thickness,
                    InputImage im,
                    int min_size = 720 );
} // blobDet
