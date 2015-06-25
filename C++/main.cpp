 #include <stdio.h>
 #include "opencv2/core/core.hpp"
 #include "opencv2/contrib/contrib.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"
 #include "opencv2/objdetect/objdetect.hpp"
 #include "Utils.h"
 #include "rc2dpca.h"
 #include "two_d_pca.h"

 #include <iostream>
 #include <fstream>
 #include <sstream>
 #include <vector>
 #include <Eigen/Dense>

using namespace cv;
using namespace Eigen;
using namespace std;

int main()
{
    // vector<Mat> images;
    // vector<int> labels;
    // dir_read("../orl_faces/Train",40,images,labels,0);

    // rc2dpca model;
    // //model.train(images,labels,0.7,"rc2dpca.xml");
    // model.load("rc2dpca.xml");

    // //*-*-*-*-*-*-*- FIX ALL THE RESIZES TO SIZE(n,m)*-*-*-*-*-*-*-*-//

    // double sum=0;
    // images.clear();
    // labels.clear();
    // dir_read("../orl_faces/Test",40,images,labels,0);
    // for(int i=0;i<images.size();i++)
    // {
    //   int prediction=  model.predict(images[i]);
    //   cout<<"\t"<<prediction<<"\t"<<labels[i]<<endl;
    //   sum+=(prediction==labels[i]);
    // }
    // sum*=100;
    // cout<<" accuracy is "<<sum/labels.size()<<endl;
    // cout<<"here"<<endl;

    //plot("../orl_faces/Train","../orl_faces/Test");
    string dir_train="../orl_faces/Train",dir_test="../orl_faces/Test";
    vector<Mat> images_train,images_test;
    vector<int> labels_train,labels_test;
    dir_read(dir_train,40,images_train,labels_train,0);
    dir_read(dir_test,40,images_test,labels_test,0);
    rc2dpca model1;
    pca2d model2;
    const int bins=15;
    float accuracy1[bins];
    float accuracy2[bins];
    cout<<" Percentage of information \t rc2dpca \t 2dpca "<<endl;
    for(int i=0;i<bins;i++)
    {
        model1.train(images_train,labels_train,(30+i*5)/100.0,"rc2dpca.xml");
        model2.train(images_train,labels_train,(30+i*5)/100.0,"2dpca.xml");

        double sum1=0,sum2=0;
        for(int j=0;j<images_test.size();j++)
        {
          int prediction1=  model1.predict(images_test[j]);
          sum1+=(prediction1==labels_test[j]);
          int prediction2=  model2.predict(images_test[j]);
          sum2+=(prediction2==labels_test[j]);
        }
        sum1*=100;
        sum2*=100;
        accuracy1[i]=sum1/labels_test.size();
        accuracy2[i]=sum2/labels_test.size();
        cout<<"\t\t"<<30+i*5<<"\t\t"<<accuracy1[i]<<"\t\t"<<accuracy2[i]<<endl;
    }
    return 0;
}


