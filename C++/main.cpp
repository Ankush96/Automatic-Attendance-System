
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
    vector<Mat> images;
    vector<int> labels;
    dir_read("../orl_faces/Train",40,images,labels,0);

    pca2d model;
    model.train(images,labels,0.99,"2dpca.xml");
    model.load("2dpca.xml");
    double sum=0;
    images.clear();
    labels.clear();
    dir_read("../orl_faces/Test",40,images,labels,0);
    for(int i=0;i<images.size();i++)
    {
      cout<<"\t"<<model.predict(images[i])<<"\t"<<labels[i]<<endl;
      sum+=(model.predict(images[i])==labels[i]);
    }
    //model.predict(images[22]);
    sum*=100;
    cout<<" accuracy is "<<sum/labels.size()<<endl;
    return 0;
}


