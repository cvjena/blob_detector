#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include "blob_detector/core.h"

namespace blob = blobDet;

using json = nlohmann::json; // for convenience

namespace det2json {

    struct Detection {
        double x;
        double y;
        double w;
        double h;
        double score;
    };

    struct DetectionEntry {
        std::string filePath;
        std::vector<Detection> detections;
    };

    void to_json( json& j, const Detection& det ){
        j = json {
            {"x", det.x},
            {"y", det.y},
            {"width", det.w},
            {"height", det.h},
            {"score", det.score}
        };
    }

    void to_json( json& j, const DetectionEntry& detEnt ){
        j = json {
            {"filePath", detEnt.filePath},
            {"detections", detEnt.detections},
        };

    }

} // det2json



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
    blob::findBorder(gray, border);

    // 3. gaussian blur and optional histogram equalization
    blob::preprocess(gray, gray_processed /* sigma, equalize*/ );

    // 4. binarization (including border information)
    blob::binarize(gray_processed, bin_im, border);

    // 4.5 [optional] morphological operations
    // (set parameters to -1 to disable this operation)
    blob::openClose(bin_im, 5, 2);

    std::vector<blob::BBox> boxes;
    // 5. Detect bounding boxes (including border information)
    blob::detect(bin_im, border, boxes);

    std::vector<blob::BBox> boxes2;
    // 6. Try to split bounding boxes
    blob::splitBoxes(gray, boxes, boxes2);

    // 7.1. Filter bounding boxes: NMS
    std::vector<int> indices(boxes2.size());
    blob::nmsBoxes(boxes2, indices, 0.5, 0.3);

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

    std::vector<det2json::Detection> detections;
    detections.resize(indices.size());

    // 10. construct struct entries
    int j = 0;
    for (int i: indices)
    {
        auto box = boxes2[i];
        auto score = box.getScore();
        auto origin = box.origin();
        detections[j++] = {
            origin.x, origin.y,
            box.width(), box.height(),
            std::isnan(score) ? 0 : score
        };
    }

    det2json::DetectionEntry entry {
        std::string(argv[1]),
        detections
    };

    json json_entry = entry;

    std::cout << json_entry << std::endl;

    return 0;
}
