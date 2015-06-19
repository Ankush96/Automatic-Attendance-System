
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "Utils.h"
#include "two_d_pca.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>

using namespace cv;
using namespace Eigen;
int main()
{
    //int a=video_recognizer();
    //int a=model_main();
    vector<Mat> images;
    vector<int> labels;
    dir_read("../Face",6,images,labels);

    pca2d model;
    model.train(images,labels);
    return 0;
}


