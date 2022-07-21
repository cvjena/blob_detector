#include <opencv2/opencv.hpp>

#include "blob_detector/core.h"

using namespace std;

namespace blobDet {

BBox::BBox(vector<cv::Point> contour, cv::Size size)
{
    float x0 = -1, x1 = -1, y0 = -1, y1 = -1;
    for (cv::Point pt: contour)
    {
        if(x0 == -1 || x0 > pt.x) x0 = pt.x;
        if(y0 == -1 || y0 > pt.y) y0 = pt.y;
        if(x1 == -1 || x1 < pt.x) x1 = pt.x;
        if(y1 == -1 || y1 < pt.y) y1 = pt.y;
    }
    this->upper_left.x = x0 / size.width;
    this->upper_left.y = y0 / size.height;
    this->lower_right.x = x1 / size.width;
    this->lower_right.y = y1 / size.height;
}

void BBox::toAbsolute(const cv::Size &size, int& x0, int& y0, int& x1, int& y1) const
{
    int w = size.width, h = size.height;

    x0 = max(0, int(this->upper_left.x * w));
    y0 = max(0, int(this->upper_left.y * h));
    x1 = min(w - 1, int(this->lower_right.x * w));
    y1 = min(h - 1, int(this->lower_right.y * h));
}

void BBox::toAbsolute(InputImage image, int& x0, int& y0, int& x1, int& y1) const
{
    this->toAbsolute(image.size(), x0, y0, x1, y1);
}

bool BBox::splittable(const cv::Size &size) const
{
    int x0, y0, x1, y1;
    this->toAbsolute(size, x0, y0, x1, y1);
    int w = x1-x0, h = y1-y0;
    double diag = sqrt(pow(w, 2) + pow(h, 2));
    return diag >= 50.0 and MIN_AREA < this->area() and this->area() <= MAX_AREA;
}

bool BBox::isValid(const float validRatio, const float minArea, const float maxArea) const
{

    return this->width() != 0 && this->height() != 0 &&
        this->ratio() >= validRatio &&
        minArea <= this->area() && this->area() <= maxArea;
}

void BBox::draw(OutputImage image) const
{
    cv::Scalar color;
    switch(image.channels()){
        case 1:
            color = cv::Scalar(0);
            break;
        case 3:
            color = cv::Scalar(0,0,0);
            break;
        case 4:
            color = cv::Scalar(0,0,0,1);
            break;
        default:
            cerr << "ERROR! Wrong number of channels: " << image.channels() << endl;
            exit(1);
    }

    this->draw(image, color);
}

void BBox::draw(OutputImage image, const cv::Scalar& color, int thickness, int lineType, int shift) const
{
    int x0, y0, x1, y1;
    this->toAbsolute(image.size(), x0, y0, x1, y1);

    cv::Point ul(x0, y0), lr(x1, y1);
    cv::rectangle(image, ul, lr, color, thickness, lineType, shift);

    if (isnan(this->score))
        return;

    const char *fmt = "%.3f";
    int sz = snprintf(nullptr, 0, fmt, this->score);
    vector<char> buf(sz + 1); // note +1 for null terminator
    snprintf(&buf[0], buf.size(), fmt, this->score);

    putText(image, string(&buf[0]), this->origin(), color, 1, lineType);
}

void BBox::crop(InputImage image, OutputImage crop){

    int x0, y0, x1, y1;
    this->toAbsolute(image.size(), x0, y0, x1, y1);
    crop = image(cv::Range(y0, y1), cv::Range(x0, x1)).clone();
}

BBox BBox::rescale(const BBox &other)
{
    cv::Point2d scale = other.size();
    cv::Point2d shift = other.origin();
    return (*this) * scale + shift;
}

void BBox::tile(BBoxes &tiles, const int n_tiles)
{
    this->tile(tiles, n_tiles, n_tiles);
}

void BBox::tile(BBoxes &tiles, const int n_xtiles, const int n_ytiles)
{
    tiles.resize(n_xtiles * n_ytiles);

    double tile_w = this->width() / n_xtiles,
           tile_h = this->height() / n_ytiles;

    double x0 = this->upper_left.x,
           x1 = this->lower_right.x,
           y0 = this->upper_left.y,
           y1 = this->lower_right.y;

    int i = 0;
    for (double x = x0; x < x1; x += tile_w)
        for (double y = y0; y < y1; y += tile_h)
            tiles[i++] = BBox(x, y, x + tile_w, y + tile_h);

}

void BBox::setScore(InputImage image)
{
    cv::Mat crop, offsetCrop, corr;
    BBox offset_box;

    this->crop(image, crop);

    float W = image.size().width, H = image.size().height;
    float w = crop.size().width, h = crop.size().height;

    this->enlarge(offset_box, w / W, h / H);
    offset_box.crop(image, offsetCrop);

    correlate(offsetCrop, crop, corr);

    BBoxes tiles;
    BBox().tile(tiles, 3);

    cv::Mat patch;
    double middleMean = 0.0, adjacentMean = 0.0;

    for (int i = 0; i < 3*3; i++)
    {
        tiles[i].crop(corr, patch);
        auto mean = cv::mean(patch);

        if (i == 4) // corresponds to the middle tile
            middleMean = mean[0];
        else // in total, there are 8 adjacent patches
            adjacentMean += mean[0] / 8;
    }

    this->score = 1 - (adjacentMean / middleMean);

}

void BBox::enlarge(const float scaleX, const float scaleY)
{
    if ( scaleX <= 0 || scaleX >= 1 || scaleY <= 0 || scaleY >= 1 )
        return;

    this->upper_left.x = max(0.0, this->upper_left.x - scaleX);
    this->upper_left.y = max(0.0, this->upper_left.y - scaleY);

    this->lower_right.x = min(1.0, this->lower_right.x + scaleX);
    this->lower_right.y = min(1.0, this->lower_right.y + scaleY);
}
void BBox::enlarge(const float scale)
{
    this->enlarge(scale, scale);
}

void BBox::enlarge(BBox &other, const float scaleX, const float scaleY)
{
    other = BBox(*this);
    other.enlarge(scaleX, scaleY);
}

void BBox::enlarge(BBox &other, const float scale)
{
    this->enlarge(other, scale, scale);
}


BBox BBox::operator*(cv::Point2d scale) {
    return BBox(
        this->upper_left.x * scale.x,
        this->upper_left.y * scale.y,
        this->lower_right.x * scale.x,
        this->lower_right.y * scale.y
    );
}

BBox BBox::operator+(cv::Point2d shift) {
    return BBox(
        this->upper_left.x + shift.x,
        this->upper_left.y + shift.y,
        this->lower_right.x + shift.x,
        this->lower_right.y + shift.y
    );
}

ostream& operator<<(ostream& os, const BBox& bbox) {
    const char *fmt = "[ LR(%.3f, %.3f) | UR(%.3f, %.3f) ]";
    int sz = std::snprintf(nullptr, 0, fmt,
        bbox.upper_left.x,
        bbox.upper_left.y,
        bbox.lower_right.x,
        bbox.lower_right.y);

    std::vector<char> buf(sz + 1); // note +1 for null terminator
    std::snprintf(&buf[0], buf.size(), fmt,
        bbox.upper_left.x,
        bbox.upper_left.y,
        bbox.lower_right.x,
        bbox.lower_right.y);
    os << &buf[0];
    return os;
}

} // blobDet
