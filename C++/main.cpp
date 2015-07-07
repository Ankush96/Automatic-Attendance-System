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

#define n 120
#define m 120

int main()
{
    //--------------Code to check if segmentation is working fine--------------------//
      int cr_min=134,cr_max=175,cb_min=99,cb_max=130;
    vector<Mat> images;
    vector<int> labels;
    dir_read("../Face_db",9,images,labels,1);
    cvNamedWindow("src",WINDOW_NORMAL);
    cvNamedWindow("dst",WINDOW_NORMAL);
    createTrackbar("cr min ","dst",&cr_min,255);
    createTrackbar("cr max ","dst",&cr_max,255);
    createTrackbar("cb min ","dst",&cb_min,255);
    createTrackbar("cb max ","dst",&cb_max,255);
    for(int i=0;i<images.size();i++)
    {
      Mat src=images[i];
      //cout<<src.size<<endl;
      imshow("src",src);
        while(1)
            {
                Mat dst=getBB(remove_blobs(GetSkin(src,cr_min,cr_max,cb_min,cb_max)));
                resize(dst,dst,Size(src.cols,src.rows),0,0,INTER_CUBIC);
                imshow("dst",dst);
                int key = cv::waitKey(30);
                if(key==27) break;
            }

    }
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
    // vector<Mat> images;
    // vector<int> labels;
    // int num_dir=13;     //  Number of classes or unique identities
    // int examples=7;     //  Number of images per person
    // int color=0;
    // dir_read("../CVL",num_dir,images,labels,color);
    // //rc2dpca model;
    // if(color)
    // {
    //   for(int i=0;i<images.size();i++)
    //     {
    //         Mat src=images[i];
    //         //cvtColor(GetSkin(src,cr_min,cr_max,cb_min,cb_max),images[i],CV_BGR2GRAY);  //GetSkin returns a color image, hence we need to convert it to grayscale
    //         Mat dst=getBB(remove_blobs(GetSkin(src,cr_min,cr_max,cb_min,cb_max)));
    //         resize(dst,dst,Size(n,m),0,0,INTER_CUBIC);
    //         images[i]=dst;
    //     }
    // }

    // std::vector<Mat> images_test,images_train;
    // std::vector<int> labels_train,labels_test;
    // double* accuracy = new double[examples*sizeof( double )];     // A malloc implementation has to be done in the f

    // //cvNamedWindow("src",WINDOW_NORMAL);
    // double y[101];
    // fstream myfile("Plots/11.txt", ios::out);
    // if (myfile.is_open()) cout<<"file exists"<<endl;
    // for(int i=0;i<100;i++)
    // {
    //     for(int k=0;k<examples;k++)
    //     {
    //         //cout<<" K= "<<k<<endl;
    //         accuracy[k]=0;
    //         images_train.clear();
    //         images_test.clear();
    //         labels_train.clear();
    //         labels_test.clear();
    //         for(int i=0;i<images.size();i++)
    //         {
    //             if(i%examples==k)  // Put in test set
    //             {
    //                 images_test.push_back(images[i]);
    //                 labels_test.push_back(labels[i]);
    //             }
    //             else        // Put in training set
    //             {
    //                 images_train.push_back(images[i]);
    //                 labels_train.push_back(labels[i]);
    //             }
    //         }
    //         //model.train(images_train,labels_train,i/100.0,"2dpca.xml");
    //         Ptr<FaceRecognizer> model = createEigenFaceRecognizer(4*(i+1));
    //         model->train(images_train, labels_train);
    //         for(int j=0;j<images_test.size();j++)
    //         {
    //             //int prediction=  model.predict(images_test[j]);
    //             int prediction=  model->predict(images_test[j]);


    //             //imshow("src",images_test[j]);
    //             //cout<<" actual -> "<<labels_test[j]<<" predicted ->"<<prediction<<endl;
    //             //waitKey(0);
    //             accuracy[k]+=(prediction==labels_test[j]);
    //         }

    //         //cout<<" accuracy for k="<<k<<" is "<<accuracy[k]<<" "<< (accuracy[k]*100)/(labels_test.size())<<endl;
    //     }

    //     for(int k=1;k<examples;k++)
    //     {
    //         accuracy[k]+=accuracy[k-1];
    //     }

    //     y[i]=(accuracy[examples-1]*100)/(examples*num_dir);
    //     cout<<endl<<"percentage"<<i<<" final accuracy -> "<<y[i]<<endl;
    //     myfile<<y[i]<<"\n";
    // }
    // myfile.close();
    //-------------------------------------------------------------------------------------//

    //------------------Drawing Roc curves for all the classes-------------------------//

    // int roc_range=15;       //    Number of data points we want in the ROC curve
    // int trueP[num_dir][roc_range];
    // int falseP[num_dir][roc_range];

    // for(int i=0;i<num_dir;i++)
    // {
    //     for(int j=0;j<roc_range;j++)
    //     {
    //         trueP[i][j]=0;
    //         falseP[i][j]=0;
    //     }
    // }

    // fstream tp("Plots/3.txt", ios::out);
    // if (tp.is_open()) cout<<"file exists"<<endl;
    // fstream fp("Plots/4.txt", ios::out);
    // if (fp.is_open()) cout<<"file exists"<<endl;

    // for(int class_no=0;class_no<num_dir;class_no++)       //    We find the curves for every class
    // {

    //     cout<<" Class number -> "<<class_no+1<<endl;
    //     for(int k=0;k<examples;k++)
    //     {
    //         //cout<<" Example number -> "<<k<<endl;
    //         images_train.clear();
    //         images_test.clear();
    //         labels_train.clear();
    //         labels_test.clear();
    //         for(int i=0;i<images.size();i++)
    //         {
    //             if(i%examples==k)  // Put in test set
    //             {
    //                 images_test.push_back(images[i]);
    //                 labels_test.push_back(labels[i]);
    //             }
    //             else        // Put in training set
    //             {
    //                 images_train.push_back(images[i]);
    //                 labels_train.push_back(labels[i]);
    //             }
    //         }


    //         for(int t=0;t<roc_range;t++)
    //         {

    //             model.train(images_train,labels_train,(29+5*t)/100.0,"2dpca.xml");

    //             for(int j=0;j<images_test.size();j++)
    //             {
    //                 int prediction=  model.predict(images_test[j]);
    //                 //cout<<" actual -> "<< labels_test[j] <<" predicted -> " << prediction <<endl;
    //                 if(prediction==labels_test[class_no])      //     A positive detected for that particular class
    //                 {
    //                     if(prediction==labels_test[j])  trueP[class_no][t]++;
    //                     else falseP[class_no][t]++;
    //                 }
    //             }
    //             if(k==examples-1)
    //             {
    //                 cout<<" percentage -> "<<29+5*t<<endl;
    //                 cout<<" true positive -> "<< trueP[class_no][t] << endl;
    //                 cout<<" false positive -> "<< falseP[class_no][t] << endl;
    //                 tp<<trueP[class_no][t]<<";";
    //                 fp<<falseP[class_no][t]<<";";
    //             }

    //         }




    //     }
    //     tp<<endl;
    //     fp<<endl;


    // }

    //---------------------------------------------------------------------------------//


    return 0;
}


