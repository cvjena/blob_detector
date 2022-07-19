#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>

#include <opencv2/opencv.hpp>


using namespace std;
/* heads */

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


    void draw(cv::Mat &image) const;
    void draw(cv::Mat &image,
              const cv::Scalar &color,
              int thickness = 1,
              int lineType = cv::LINE_AA,
              int shift = 0) const;
    void crop(cv::Mat image, cv::Mat &crop);

    void toAbsolute(const cv::Size &size, int&, int&, int&, int&) const;
    void toAbsolute(const cv::Mat &image, int&, int&, int&, int&) const;

    bool splittable(const cv::Size &size) const;
    bool isValid(const float validRatio = 0.1, const float minArea = 4e-4, const float maxArea = 1/9.0) const;

    BBox rescale(const BBox &other);

    void enlarge(const float scale = 0.01);
    void enlarge(const float scaleX, const float scaleY);
    void enlarge(BBox &other, const float scale = 0.01);
    void enlarge(BBox &other, const float scaleX, const float scaleY);

    void setScore(const cv::Mat &image);
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

void detect( cv::Mat image, BBoxes &boxes );
void detect( cv::Mat image, cv::Mat mask, BBoxes &boxes );
void splitBoxes(cv::Mat image, const BBoxes &boxes, BBoxes &newBoxes );

void findBorder( cv::Mat image, cv::Mat &border_mask, const double threshold = 50.0, const int pad = 10 );
void preprocess( cv::Mat image, cv::Mat &output, const double sigma = 5.0, const bool equalize = false );
void binarize( cv::Mat image, cv::Mat &output, const int window_size = 31, const float C = 2.0 );
void binarize( cv::Mat image, cv::Mat &output, cv::Mat mask, const int window_size = 31, const float C = 2.0);
void filterBoxes( BBoxes &boxes, vector<int> &indices, const float score_threshold = 0.5, const float nms_threshold = 0.3);

void open_close( cv::Mat &image, int kernel_size = 3, int iterations = 2 );
bool compareContours(vector<cv::Point> c1, vector<cv::Point> c2 );
void gaussian( cv::Mat image, cv::Mat &output, double sigma );
void correlate( const cv::Mat &im1, const cv::Mat &im2, cv::Mat &corr, bool normalize = true);
void imshow( const string &name, cv::Mat im );

void putText( cv::Mat &image,
              const string &text,
              const cv::Point2d &pos,
              const cv::Scalar &color = cv::Scalar(0),
              int thickness = 1,
              int lineType = cv::LINE_AA);

void showBoxes( const string &name, cv::Mat &im, const BBoxes &boxes, const cv::Scalar &color,
                int thickness = 1,
                int lineType = cv::LINE_AA);

void showBoxes( const string &name, cv::Mat &im, const BBoxes &boxes, const vector<int> &indices,
                const cv::Scalar &color,
                int thickness = 1,
                int lineType = cv::LINE_AA);

int waitKey( float timer = 0.1 );

/* end heads */

/* Utils */
void gaussian( cv::Mat image, cv::Mat &output, double sigma ){
    cv::GaussianBlur(image, output, cv::Size(0, 0), sigma);
}

void correlate( const cv::Mat &image, const cv::Mat &kernel, cv::Mat &corr, bool normalize)
{
    cv::Mat _image, _image_padded, _kernel;

    image.convertTo(_image, CV_32F);
    kernel.convertTo(_kernel, CV_32F);

    // 0..255 -> 0..1
    _image /= 255.0;
    _kernel /= 255.0;

    // subtract average mean
    auto mean = (cv::mean(_image) + cv::mean(_kernel)) / 2;
    _image -= mean;
    _kernel -= mean;

    // add proper padding
    int k_w = _kernel.size().width,
        k_h = _kernel.size().height;

    _image_padded = cv::Mat(
        _image.rows + k_h,
        _image.cols + k_w,
        _image.depth());

    cv::copyMakeBorder(_image, _image_padded,
                       k_h/2, k_h/2, k_w/2, k_w/2,
                       cv::BORDER_REFLECT,
                       cv::Scalar(0));


    // perform correlation operation
    cv::matchTemplate(_image_padded, _kernel, corr, cv::TM_CCORR_NORMED);

    // normalize if needed
    if (normalize)
    {
        double px_min, px_max;
        cv::minMaxLoc( corr, &px_min, 0);
        corr -= px_min;
        cv::minMaxLoc( corr, 0, &px_max);
        corr /= px_max;
    }

}


void imshow( const string &name, cv::Mat im ){
    cv::namedWindow(name, cv::WINDOW_NORMAL);
    cv::imshow(name, im);
}

void putText(cv::Mat &image, const string &text, const cv::Point2d &pos, const cv::Scalar &color, int thickness, int lineType)
{

    const int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
    const float fontScale = 0.6;
    const int offset = 5;
    int baseline = 0;

    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;


    cv::Point textOrigin(
        (image.cols * pos.x) + offset,
        (image.rows * pos.y) - baseline - offset);

    cv::rectangle(image,
                  textOrigin + cv::Point(-offset, baseline+offset),
                  textOrigin + cv::Point(textSize.width+offset, -textSize.height-offset),
                  color, thickness, lineType);

    cv::putText(image, text, textOrigin, fontFace, fontScale, color, thickness, lineType);

}

int waitKey(float timer)
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

void showBoxes( const string &name, cv::Mat &im, const BBoxes &boxes, const cv::Scalar& color, int thickness, int lineType){
    cv::namedWindow(name, cv::WINDOW_NORMAL);

    for (BBox box: boxes)
        box.draw(im, color, thickness, lineType);

    cv::imshow(name, im);

}

void showBoxes( const string &name, cv::Mat &im, const BBoxes &boxes, const vector<int> &indices, const cv::Scalar& color, int thickness, int lineType){
    cv::namedWindow(name, cv::WINDOW_NORMAL);

    for (int i: indices)
        boxes[i].draw(im, color, thickness, lineType);

    cv::imshow(name, im);

}

bool compareContours( vector<cv::Point> c1, vector<cv::Point> c2 )
{
    return fabs(cv::contourArea(cv::Mat(c1))) > fabs(cv::contourArea(cv::Mat(c2)));
}


/* End Utils */


/* Bounding Box */
BBox::BBox(vector<cv::Point> contour, cv::Size size){
    float x0 = -1, x1 = -1, y0 = -1, y1 = -1;
    for (cv::Point pt: contour){
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

void BBox::toAbsolute(const cv::Mat &image, int& x0, int& y0, int& x1, int& y1) const
{
    this->toAbsolute(image.size(), x0, y0, x1, y1);
}

bool BBox::splittable(const cv::Size &size) const
{
    int x0, y0, x1, y1;
    this->toAbsolute(size, x0, y0, x1, y1);
    int w = x1-x0, h = y1-y0;
    double diag = sqrt(pow(w, 2) + pow(h, 2));
    return diag >= 50.0 and this->area() <= 0.5;
}
bool BBox::isValid(const float validRatio, const float minArea, const float maxArea) const
{

//     cout << validRatio << ", " << minArea << ", " << maxArea << " | ";
//     const char *fmt = "[ H: %.5f, W: %.5f, ratio: %.3f, area: %.5f ]";
//     int sz = std::snprintf(nullptr, 0, fmt,
//         this->width(),
//         this->height(),
//         this->ratio(),
//         this->area());

//     std::vector<char> buf(sz + 1); // note +1 for null terminator
//     std::snprintf(&buf[0], buf.size(), fmt,
//         this->width(),
//         this->height(),
//         this->ratio(),
//         this->area());

//     cout << &buf[0] << endl;

    return this->width() != 0 && this->height() != 0 &&
        this->ratio() >= validRatio &&
        minArea <= this->area() && this->area() <= maxArea;
}

void BBox::draw(cv::Mat &image) const
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
    }

    this->draw(image, color);
}

void BBox::draw(cv::Mat &image, const cv::Scalar& color, int thickness, int lineType, int shift) const
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

void BBox::crop(cv::Mat image, cv::Mat &crop){

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

void BBox::setScore(const cv::Mat &image)
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

    // cv::Mat foo = offsetCrop.clone();
    // showBoxes("OffsetCrop", foo, tiles, cv::Scalar(190), 2);

    // cv::Mat bar = corr.clone();
    // showBoxes("Correlation", bar, tiles, cv::Scalar(190/255.0), 2);

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
/* End Bounding Box */



void findBorder( cv::Mat image, cv::Mat &border_mask, double threshold, int pad )
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
        auto size = border_mask.size();
        border_mask = border_mask(
            cv::Range(pad, size.height - pad),
            cv::Range(pad, size.width - pad));
    }

}



void preprocess( cv::Mat image, cv::Mat &output, double sigma, bool equalize){

    if ( equalize )
        cv::createCLAHE(2.0, cv::Size(10, 10))->apply(image, output);
    else
        output = image.clone();

    gaussian(image, output, sigma);

}


void binarize( cv::Mat image, cv::Mat &output, int window_size, float C)
{
    auto maxValue = 255;
    cv::adaptiveThreshold(image, output,
        maxValue,
        cv::ADAPTIVE_THRESH_MEAN_C,
        cv::THRESH_BINARY_INV,
        window_size, C);
}
void binarize( cv::Mat image, cv::Mat &output, cv::Mat mask, int window_size, float C)
{
    cv::Mat masked = cv::Mat::zeros( image.size(), image.type() );
    image.copyTo(masked, mask);
    binarize(masked, output, window_size, C);
    output.copyTo(output, mask);
}

void open_close( cv::Mat &img, int kernel_size, int iterations)
{
    cv::Mat kernel = cv::Mat::ones(kernel_size, kernel_size, img.type());

    if (kernel_size >= 1){
        cv::morphologyEx(img, img, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(img, img, cv::MORPH_CLOSE, kernel);
    }

    if (iterations >= 1){
        cv::erode(img, img, kernel, cv::Point(-1, -1), iterations);
        cv::dilate(img, img, kernel, cv::Point(-1, -1), iterations);
    }

}

void detect(cv::Mat image, cv::Mat mask, BBoxes &boxes)
{
    cv::Mat masked = cv::Mat::zeros( image.size(), image.type() );
    image.copyTo(masked, mask);
    detect(masked, boxes);
}

void detect(cv::Mat image, BBoxes &boxes)
{
    vector<vector<cv::Point> > contours;

    cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    sort(contours.begin(), contours.end(), compareContours);
    boxes.resize(contours.size());

    for (int i = 0; i < contours.size(); ++i)
        boxes[i] = BBox(contours[i], image.size());

}

void splitBoxes(cv::Mat image, const BBoxes &boxes, BBoxes &outputBoxes)
{

    cv::Mat crop, crop_processed, bin_crop;
    vector<BBox> newBoxes;
    for (BBox bbox : boxes){
        outputBoxes.push_back(bbox);
        if (!bbox.splittable(image.size()))
            continue;


        bbox.crop(image, crop);

        preprocess(crop, crop_processed, 2.0, true);
        binarize(crop_processed, bin_crop, 15);
        open_close(bin_crop, 5.0, 2);

        newBoxes.clear();

        detect(bin_crop, newBoxes);


        for (auto _box: newBoxes)
            outputBoxes.push_back(_box.rescale(bbox));
    }

}

void nmsBoxes(BBoxes &boxes,
                 vector<int> &indices,
                 const float score_threshold,
                 const float nms_threshold)
{

    vector<cv::Rect2d> _boxes(boxes.size());
    vector<float> scores(boxes.size(), 1.0);

    int i = 0;
    for (BBox box : boxes)
        _boxes[i++] = box.asRect();

    cv::dnn::NMSBoxes(
        _boxes,
        scores,
        score_threshold,
        nms_threshold,
        indices);
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
    auto alpha = 1.0;
    gray_processed = gray_processed * alpha + bin_im * (1 - alpha);
    imshow("Image", gray_processed);

    // 4.5 [optional] morphological operations
    // (set parameters to -1 to disable this operation)
    open_close(bin_im, 5, 2);

    vector<BBox> boxes;
    // 5. Detect bounding boxes (including border information)
    detect(bin_im, border, boxes);

    cv::Mat det = bin_im.clone();
    showBoxes("Detections", det, boxes, cv::Scalar(127), 2);

    vector<BBox> boxes2;
    // 6. Try to split bounding boxes
    splitBoxes(gray, boxes, boxes2);

    showBoxes("Split Detections", det, boxes2, cv::Scalar(190), 2);

    // 7.1. Filter bounding boxes: NMS
    vector<int> indices(boxes2.size());
    nmsBoxes(boxes2, indices, 0.5, 0.3);

    det = bin_im.clone();
    showBoxes("Filtered Detections (1)", det, boxes2, indices, cv::Scalar(127), 2);

    cout << indices.size();

    // 7.2. Filter bounding boxes: Validation checks
    indices.erase(remove_if(
        indices.begin(), indices.end(),
        [&boxes2](int i) { return !boxes2[i].isValid(); }
    ), indices.end());

    cout << " > " << indices.size() << endl;

    det = bin_im.clone();
    showBoxes("Filtered Detections (2)", det, boxes2, indices, cv::Scalar(190), 2);

    // 8. Enlarge boxes
    for (int i : indices)
        boxes2[i].enlarge(0.01);

    det = bin_im.clone();
    showBoxes("Enlared", det, boxes2, indices, cv::Scalar(190), 2);

    // 9. Score boxes
    for (int i : indices)
        boxes2[i].setScore(gray);

    showBoxes("Final", image, boxes2, indices, cv::Scalar(255, 0, 0), 2);

    return waitKey();
}
