
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "Utils.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace cv;

int main()
{
    int a=image_recognizer("../Face_bu/Faces2");
    //int a=model_main();
    vector<Mat> images;
    vector<int> labels;

    //dir_read("../Face",6,images,labels);
    return 0;
}


