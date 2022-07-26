#pragma once
#include <opencv2/opencv.hpp>

#include "blob_detector/core.h"

namespace blobDet {

void findBorder( InputImage image,
                 OutputImage border_mask,
                 const double threshold = 50.0,
                 const int pad = 10 );

void rescale( OutputImage image,
              const int min_size = 1080,
              const double min_scale = 0.1 );

void preprocess( InputImage image,
                 OutputImage output,
                 const double sigma = 5.0,
                 const bool equalize = false );

void binarize( InputImage image,
               OutputImage output,
               const int window_size = 31,
               const float C = 2.0 );

void binarize( InputImage image,
               OutputImage output,
               InputImage mask,
               const int window_size = 31,
               const float C = 2.0 );

void removeBorder( InputImage image,
                   InputImage border_mask,
                   OutputImage output );

void openClose( OutputImage image,
                int kernel_size = 3,
                int iterations = 2 );

void detect( InputImage image,
             BBoxes &boxes );

void detect( InputImage image,
             InputImage mask,
             BBoxes &boxes );

void splitBoxes( InputImage image,
                 const BBoxes &boxes,
                 BBoxes &newBoxes );

void filterBoxes( BBoxes &boxes,
                  std::vector<int> &indices,
                  const float score_threshold = 0.5,
                  const float nms_threshold = 0.3 );

void nmsBoxes( BBoxes &boxes,
               std::vector<int> &indices,
               const float score_threshold,
               const float nms_threshold);

} // blobDet
