#ifndef __UTILS_H_INCLUDED__
#define __UTILS_H_INCLUDED__


#include "opencv2/core/core.hpp"

void dir_read(std::string, int, std::vector<cv::Mat>&, std::vector<int>&, bool);

static void read_csv(const std::string&, std::vector<cv::Mat>&, std::vector<int>&, char);

int sampler_main();


#endif
