/*
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "Stage-segment.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static void segment(const string& filename, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            Mat tmp1=imread(path);
            imshow("img",tmp1);
            waitKey(0);
            imwrite(path,GetSkin(tmp1));
        }
    }
}

int main()
{
    string fn_csv = "samples.csv";
    try {
        segment(fn_csv);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }
	return 0;
}
*/
