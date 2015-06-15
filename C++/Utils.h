#include <stdio.h>
#include "Stage-segment.h"
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace cv;
using namespace std;

string prediction_name(int prediction);

int video_recognizer();

static void read_csv(const string& , vector<Mat>& , vector<int>& , char separator);

int model_main();

static void segment(const string&, char separator);

int segment_samples_main();
