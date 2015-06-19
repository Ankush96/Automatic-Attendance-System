#include <stdio.h>
#include "Stage-segment.h"
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>


void dir_read(string,int,vector<Mat>&,vector<int>&);

string prediction_name(int);

int video_recognizer();

int image_recognizer(string);

static void read_csv(const string& , vector<Mat>& , vector<int>& , char separator);

int model_main(string);

static void segment(const string&, char separator);

int segment_samples_main();

int sampler_main();
