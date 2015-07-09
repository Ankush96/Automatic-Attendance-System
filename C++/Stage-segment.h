
#ifndef __STAGE_SEGMENT_H_INCLUDED__
#define __STAGE_SEGMENT_H_INCLUDED__

#include "opencv2/core/core.hpp"

cv::Mat clahe(cv::Mat);

cv::Mat erode(cv::Mat const&, int);

cv::Mat dilate(cv::Mat const&, int);

cv::Mat erode_dilate(cv::Mat const&);

cv::Mat remove_blobs(cv::Mat const&);

float stddev(std::vector<int>);

bool isBoundary(cv::Mat const&, int, int);

bool R1(int, int, int);

bool R2(float, float, float);

bool R3(float, float, float);

cv::Mat stage1(cv::Mat const&);

cv::Mat stage2(cv::Mat const&);

cv::Mat stage3(cv::Mat const&, cv::Mat const&, int);

cv::Mat stage4(cv::Mat const&, int, int);

cv::Mat stage5(cv::Mat const&, cv::Mat const&);

void cam_movement(int, cv::Mat);

cv::Mat GetSkin(cv::Mat const&,int,int,int,int);

cv::Mat getBB(cv::Mat const&);

#endif