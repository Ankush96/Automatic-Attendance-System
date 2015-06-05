#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include "math.h"

using namespace cv;

using std::cout;
using std::endl;

Mat clahe(Mat img);
Mat erode(Mat const &src,int thresh=5);

Mat dilate(Mat const &src,int thresh=2);

Mat erode_dilate(Mat const &src);

float stddev(Vector<int> w);

bool isBoundary(Mat const &src,int i,int j);

bool R1(int R, int G, int B);

bool R2(float Y, float Cr, float Cb);

bool R3(float H, float S, float V);

Mat stage1(Mat const &src);

Mat stage2(Mat const &Csrc);

Mat stage3(Mat const &src,Mat const &img,int thresh=2);

Mat stage4(Mat const &img,int thresh=4);

Mat stage5(Mat const &cs1,Mat const &s4);

Mat GetSkin(Mat const &src);
