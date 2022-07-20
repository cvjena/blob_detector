#pragma once

#include <opencv2/opencv.hpp>

#include "blob_detector/core.h"

using namespace std;

class BBox
{

private:
    cv::Point2d upper_left = cv::Point2d(0.0, 0.0);
    cv::Point2d lower_right = cv::Point2d(1.0, 1.0);

    double score = NAN;

public:
    BBox() = default;
    BBox(vector<cv::Point>, cv::Size);
    BBox(const BBox& other): upper_left(other.upper_left), lower_right(other.lower_right), score(other.score){}
    BBox(double x0, double y0, double x1, double y1): upper_left(x0, y0), lower_right(x1, y1) {}
    BBox(double x0, double y0, double x1, double y1, double score): upper_left(x0, y0), lower_right(x1, y1), score(score) {}
    BBox(cv::Point2d ul, cv::Point2d lr): upper_left(ul), lower_right(lr) {}
    ~BBox() {};


    void draw(OutputImage image) const;
    void draw(OutputImage image,
              const cv::Scalar &color,
              int thickness = 1,
              int lineType = cv::LINE_AA,
              int shift = 0) const;
    void crop(InputImage image, OutputImage crop);

    void toAbsolute( const cv::Size &size, int&, int&, int&, int& ) const;
    void toAbsolute( InputImage image, int&, int&, int&, int& ) const;

    bool splittable(const cv::Size &size) const;
    bool isValid(const float validRatio = VALID_RATIO, const float minArea = MIN_AREA, const float maxArea = MAX_AREA) const;

    BBox rescale(const BBox &other);

    void enlarge(const float scale = 0.01);
    void enlarge(const float scaleX, const float scaleY);
    void enlarge(BBox &other, const float scale = 0.01);
    void enlarge(BBox &other, const float scaleX, const float scaleY);

    void setScore(InputImage image);
    double getScore(){ return this->score; };

    void tile(vector<BBox> &tiles, const int n_tiles);
    void tile(vector<BBox> &tiles, const int n_xtiles, const int n_ytiles);

    cv::Rect2d asRect() const
    {
        return cv::Rect2d(this->upper_left, this->lower_right);
    }

    double area() const
    {
        return this->width() * this->height();
    }

    double ratio() const
    {
        return min(this->height(), this->width()) / max(this->height(), this->width());
    }

    cv::Point2d size() const
    {
        return cv::Point2d(this->width(), this->height());
    }

    cv::Point2d origin() const
    {
        return this->upper_left;
    }


    double width() const
    {
        return this->lower_right.x - this->upper_left.x;
    }

    double height() const
    {
        return this->lower_right.y - this->upper_left.y;
    }


    friend ostream& operator<<(ostream& , const BBox&);
    BBox operator *(cv::Point2d);
    BBox operator +(cv::Point2d);

};

typedef vector<BBox> BBoxes;
