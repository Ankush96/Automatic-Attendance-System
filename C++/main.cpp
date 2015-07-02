 #include <stdio.h>
 #include "opencv2/core/core.hpp"
 #include "opencv2/contrib/contrib.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"
 #include "opencv2/objdetect/objdetect.hpp"
 #include "Utils.h"
 #include "rc2dpca.h"
 #include "two_d_pca.h"
 #include "Stage-segment.h"

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
    //--------------Code to check if segmentation is working fine--------------------//
      int cr_min=134,cr_max=175,cb_min=99,cb_max=130;
    // vector<Mat> images;
    // vector<int> labels;
    // dir_read("../cvl",14,images,labels,1);
    // cvNamedWindow("src",WINDOW_NORMAL);
    // cvNamedWindow("dst",WINDOW_NORMAL);
    // createTrackbar("cr min ","dst",&cr_min,255);
    // createTrackbar("cr max ","dst",&cr_max,255);
    // createTrackbar("cb min ","dst",&cb_min,255);
    // createTrackbar("cb max ","dst",&cb_max,255);
    // for(int i=0;i<images.size();i++)
    // {
    //   Mat src=images[i];
    //   //cout<<src.size<<endl;
    //   imshow("src",src);
    //     while(1)
    //         {
    //             Mat dst=getBB(remove_blobs(GetSkin(src,cr_min,cr_max,cb_min,cb_max)));
    //             resize(dst,dst,Size(src.cols,src.rows),0,0,INTER_CUBIC);
    //             imshow("dst",dst);
    //             int key = cv::waitKey(30);
    //             if(key==27) break;
    //         }

    // }
    //----------------------------------------------------------------------------------//

    //---------------------------Putting a bounding box on the largest blob and resizing---------------//
    // Mat img=imread("binshapes.png",0);
    // cvNamedWindow("out",WINDOW_NORMAL);
    // Mat dst=getBB(remove_blobs(img));
    // imshow("out",dst);
    // cv::waitKey(0);

    //------------------Testing using cross validation-----------------------------------------------//


     // change m and n in 2dpca
     //try to crop in segmentation
     //remove black background
    vector<Mat> images;
    vector<int> labels;
    int num_dir=13;
    int color=1;
    dir_read("../cvl",num_dir,images,labels,color);
    pca2d model;
    if(color)
    {
      for(int i=0;i<images.size();i++)
        {
            Mat src=images[i];
            //cvtColor(GetSkin(src,cr_min,cr_max,cb_min,cb_max),images[i],CV_BGR2GRAY);  //GetSkin returns a color image, hence we need to convert it to grayscale
            Mat dst=getBB(remove_blobs(GetSkin(src,cr_min,cr_max,cb_min,cb_max)));
            resize(dst,dst,Size(src.cols,src.rows),0,0,INTER_CUBIC);
            images[i]=dst;
        }
    }
    std::vector<Mat> images_test,images_train;
    std::vector<int> labels_train,labels_test;
    double accuracy[7];

    //cvNamedWindow("src",WINDOW_NORMAL);
    for(int i=60;i<100;i=i+2)
    {
        for(int k=0;k<7;k++)
        {
            //cout<<" K= "<<k<<endl;
            accuracy[k]=0;
            images_train.clear();
            images_test.clear();
            labels_train.clear();
            labels_test.clear();
            for(int i=0;i<images.size();i++)
            {
                if(i%7==k)  // Put in test set
                {
                    images_test.push_back(images[i]);
                    labels_test.push_back(labels[i]);
                }
                else        // Put in training set
                {
                    images_train.push_back(images[i]);
                    labels_train.push_back(labels[i]);
                }
            }
            model.train(images_train,labels_train,i/100.0,"2dpca.xml");
            for(int j=0;j<images_test.size();j++)
            {
                int prediction=  model.predict(images_test[j]);

                //imshow("src",images_test[j]);
                //cout<<" actual -> "<<labels_test[j]<<" predicted ->"<<prediction<<endl;
                //waitKey(0);
                accuracy[k]+=(prediction==labels_test[j]);
            }

            //cout<<" accuracy for k="<<k<<" is "<<accuracy[k]<<" "<< (accuracy[k]*100)/(labels_test.size())<<endl;
        }

        for(int k=1;k<7;k++)
        {
            accuracy[k]+=accuracy[k-1];
        }

        cout<<endl<<"percentage"<<i<<" final accuracy -> "<<(accuracy[6]*100)/(7*num_dir)<<endl;
    }

    //-------------------------------------------------------------------------------------//
    //------------------------------------------------------------------//
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


    // string dir_train="../Faces/Train",dir_test="../Faces/Test";
    // vector<Mat> images_train,images_test;
    // vector<int> labels_train,labels_test;
    // dir_read(dir_train,6,images_train,labels_train,0);
    // dir_read(dir_test,6,images_test,labels_test,0);
    // cout<<"Train = "<< images_train.size()<<" test= "<<images_test.size()<<endl;
    // rc2dpca model1;
    // pca2d model2;
    // const int bins=21;
    // float accuracy1[bins];
    // float accuracy2[bins];
    // cout<<" Percentage of information \t rc2dpca \t 2dpca "<<endl;
    // for(int i=0;i<bins;i++)
    // {
    //     model1.train(images_train,labels_train,(70+i)/100.0,"rc2dpca.xml");
    //     model2.train(images_train,labels_train,(70+i)/100.0,"2dpca.xml");

    //     double sum1=0,sum2=0;
    //     for(int j=0;j<images_test.size();j=j+6)
    //     {
    //       int prediction1=  model1.predict(images_test[j]);
    //       sum1+=(prediction1==labels_test[j]);
    //       int prediction2=  model2.predict(images_test[j]);
    //       sum2+=(prediction2==labels_test[j]);
    //     }
    //     sum1*=100;
    //     sum2*=100;
    //     accuracy1[i]=sum1/labels_test.size();
    //     accuracy2[i]=sum2/labels_test.size();
    //     cout<<"\t\t"<<70+i<<"\t\t"<<accuracy1[i]*6<<"\t\t"<<accuracy2[i]*6<<endl;
    //     //waitKey(500);
    // }
    //----------------------------------------------------------------------------------//


    return 0;
}


