 #include <stdio.h>
 #include "opencv2/core/core.hpp"
 #include "opencv2/contrib/contrib.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"
 #include "opencv2/objdetect/objdetect.hpp"

 #include <iostream>
 #include <fstream>
 #include <sstream>
 #include <vector>
 #include <Eigen/Dense>

 #include "Utils.h"
 #include "rc2dpca.h"
 #include "two_d_pca.h"
 #include "Stage-segment.h"
 #include "attendance.h"


using namespace cv;
using namespace Eigen;
using namespace std;

#define n 120
#define m 120




int main()
{
    int num_dir=13;                                      //  Number of classes or unique identities
    int examples=7;                                    //  Number of images per person
    bool color=1;
    string dir="../cvl";
    int cr_min=127,cr_max=175,cb_min=99,cb_max=127;     //   A change in these values should be updated everywhere

    //model_main(dir, num_dir, color, cr_min, cr_max, cb_min, cb_max);
    image_recognizer(dir, num_dir, examples, color, cr_min, cr_max, cb_min, cb_max);
    //tune_seg_params(dir, num_dir, cr_min, cr_max, cb_min, cb_max);
    //video_recognizer(cr_min, cr_max, cb_min, cb_max);

    //------------------Drawing Roc curves for all the classes-------------------------//
{
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
}

    return 0;
}


