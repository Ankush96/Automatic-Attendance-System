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

string prediction_name(int prediction)
{
    switch(prediction)
    {
        case -1:return "Unknown";
        case 1: return "Ankush";
        case 2: return "Harsh";
        case 3: return "Mayur";
        case 4: return "Jayamani";
        case 5: return "Srishty";
        case 6: return "Satish";
        case 7: return "Narpender";
        case 8: return "Acchamal";
        case 9: return "Mridul";


    }
}



int main()
{

    int cr_min=126,cr_max=175,cb_min=99,cb_max=130;

    //--------------Code to update parameters of segmentation--------------------//
{
    // vector<Mat> images;
    // vector<int> labels;
    // dir_read("../Face_db",9,images,labels,1);
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
    //
    //}
}
    //----------------------------------------------------------------------------------//


//------------------Testing using cross validation-----------------------------------------------//

    // {



    //     vector<Mat> images;
    //     vector<int> labels;
    //     int num_dir=9;     //  Number of classes or unique identities
    //     int examples=10;     //  Number of images per person
    //     int color=0;
    //     dir_read("../Face_db",num_dir,images,labels,color);
    //     pca2d model;
    //     if(color)
    //     {
    //       for(int i=0;i<images.size();i++)
    //         {
    //             Mat src=images[i];
    //             //cvtColor(GetSkin(src,cr_min,cr_max,cb_min,cb_max),images[i],CV_BGR2GRAY);  //GetSkin returns a color image, hence we need to convert it to grayscale
    //             Mat dst=getBB(remove_blobs(GetSkin(src,cr_min,cr_max,cb_min,cb_max)));
    //             resize(dst,dst,Size(n,m),0,0,INTER_CUBIC);
    //             images[i]=dst;
    //         }
    //     }

    //     std::vector<Mat> images_test,images_train;
    //     std::vector<int> labels_train,labels_test;
    //     double* accuracy = new double[examples*sizeof( double )];

    //     //cvNamedWindow("src",WINDOW_NORMAL);
    //     double y[101];
    //     fstream myfile("Plots/o3.txt", ios::out);         //  Uncomment to write the accuracy values onto a file
    //     if (myfile.is_open()) cout<<"file exists"<<endl;
    //     for(int i=0;i<36;i++)
    //     {
    //         for(int k=0;k<examples;k++)
    //         {
    //             //cout<<" K= "<<k<<endl;
    //             accuracy[k]=0;
    //             images_train.clear();
    //             images_test.clear();
    //             labels_train.clear();
    //             labels_test.clear();
    //             for(int i=0;i<images.size();i++)
    //             {
    //                 if(i%examples==k)  // Put in test set
    //                 {
    //                     images_test.push_back(images[i]);
    //                     labels_test.push_back(labels[i]);
    //                 }
    //                 else        // Put in training set
    //                 {
    //                     images_train.push_back(images[i]);
    //                     labels_train.push_back(labels[i]);
    //                 }
    //             }
    //             model.train(images_train,labels_train,(29+2*i)/100.0,"2dpca.xml");
    //             //Ptr<FaceRecognizer> model = createEigenFaceRecognizer(4*(i+1));       //  Initialise a model for Eigenfaces.If this is uncommented all corresponding code related to EF has to be uncommented
    //             //model->train(images_train, labels_train);                             //  Train the Eigenfaces model
    //             for(int j=0;j<images_test.size();j++)
    //             {
    //                 int prediction=  model.predict(images_test[j]);     //  Prediction for 2DPCA or RC2DPCA
    //                 //int prediction=  model->predict(images_test[j]);  //  Prediction for eigenfaces


    //                 //imshow("src",images_test[j]);
    //                 //cout<<" actual -> "<<labels_test[j]<<" predicted ->"<<prediction<<endl;
    //                 //waitKey(0);
    //                 accuracy[k]+=(prediction==labels_test[j]);

    //                 //------Uncomment the following to see the misclassified images-----------//
    //                 // if(prediction!=labels_test[j])
    //                 // {
    //                 //     cvNamedWindow("Incorrect",WINDOW_NORMAL);
    //                 //     imshow("Incorrect",images_test[j]);
    //                 //     waitKey(0);
    //                 // }
    //             }

    //             //cout<<" accuracy for k="<<k<<" is "<<accuracy[k]<<" "<< (accuracy[k]*100)/(labels_test.size())<<endl;
    //         }

    //         for(int k=1;k<examples;k++)
    //         {
    //             accuracy[k]+=accuracy[k-1];
    //         }

    //         y[i]=(accuracy[examples-1]*100)/(examples*num_dir);
    //         cout<<endl<<"percentage"<<(29+2*i)<<" final accuracy -> "<<y[i]<<endl;
    //         myfile<<y[i]<<endl;   //  Writing the accuracy onto the file
    //     }
    //     myfile.close();  //   Close the file for writing the accuracy values
    // }
//-----------------------------------------------------------------------------------------------//

//----------------------------------------Model trainers-------------------------------------------//
// {
//     string dir="../Face_db";
//     vector<Mat> images;
//     vector<int> labels;
//     int num_dir=9;     //  Number of classes or unique identities
//     int examples=10;     //  Number of images per person
//     int color=1;
//     dir_read(dir,num_dir,images,labels,color);
//     pca2d model;
//     if(color)
//     {
//         for(int i=0;i<images.size();i++)
//             {
//                 Mat src=images[i];
//                 //cvtColor(GetSkin(src,cr_min,cr_max,cb_min,cb_max),images[i],CV_BGR2GRAY);  //GetSkin returns a color image, hence we need to convert it to grayscale
//                 Mat dst=getBB(remove_blobs(GetSkin(src,cr_min,cr_max,cb_min,cb_max)));
//                 resize(dst,dst,Size(n,m),0,0,INTER_CUBIC);
//                 images[i]=dst;
//             }
//     }


//     //Ptr<FaceRecognizer> lbp = createLBPHFaceRecognizer();
//     //Ptr<FaceRecognizer> ff =createFisherFaceRecognizer();
//     Ptr<FaceRecognizer> ef =createEigenFaceRecognizer();
//     pca2d model2d;
//     rc2dpca modelrc;

//     // cvNamedWindow("Face",WINDOW_NORMAL);
//     // imshow("Face",images[1]);
//     // cvWaitKey(0);

//     //lbp->train(images, labels);
//     //lbp->save("lbp.xml");


//     ef->train(images, labels);
//     ef->save("ef.xml");

//     model2d.train(images,labels,0.5,"2dpca.xml");
//     modelrc.train(images,labels,0.63,"rc2dpca.xml");

//     //ff->train(images, labels);
//     //ff->save("ff.xml");
// }


//-------------------------------------------------------------------------------------------------------//

//-----------------------Video recognizer---------------------------//


{
    VideoCapture vcap;
    Mat img,gray,sgray;

    Mat black(500,500,CV_8UC3,Scalar(0,0,0));                                                       //  Mat image to display final attendance
    Mat att=black.clone();

    char key,name[20];
    int i=0,count=-1,skip=5,y;
    int num_dir=9;                                                                                  //  Number of classes or unique identities
    double* attendance = new double[num_dir*sizeof( double )];

    int frames=-1;

    const std::string videoStreamAddress = "rtsp://root:pass123@192.168.137.89:554/axis-media/media.amp";  //open the video stream and make sure it's opened
    CascadeClassifier haar_cascade;
    haar_cascade.load("../Cascades/front_alt2.xml");


    if(!vcap.open(videoStreamAddress))
        {
            std::cout << "Error opening video stream or file" << std::endl;
            return -1;
        }
    //vcap.read(gray);
   // double height = vcap.get(CV_CAP_PROP_FRAME_HEIGHT);
   // double width = vcap.get(CV_CAP_PROP_FRAME_WIDTH);
   // cv::Size frameSize(static_cast<int>(width),static_cast<int>(height));
   // cv::VideoWriter MyVid("/home/student/Documents/MyVideo1.avi",CV_FOURCC('P','I','M','1'),30,frameSize,true);
    //cvNamedWindow("Face",WINDOW_NORMAL);

    Ptr<FaceRecognizer> ef =createEigenFaceRecognizer();
    ef->load("ef.xml");
    pca2d model2d;
    rc2dpca modelrc;
    modelrc.load("rc2dpca.xml");
    model2d.load("2dpca.xml");



    while(1)
        {
            vcap.read(img);                                                                         //  Read the frame
            frames++;
            count++;                                                                                //  Maintain a count of the frames processed

            if(count%skip==0)                                                                       //  We do our processing only on every 5th frame
            {
                img=clahe(img);                                                                     //  Perform local histogram equalisation
                Mat black(img.rows,img.cols,CV_8UC3,Scalar(0,0,0));                                 //  Maintain a black Mat image to display attendance
                vector< Rect_<int> > faces;                                                         //  Initialise a vector of rectangles that store the detected faces

                //--------------Start detecting the faces in a frame------------------//
                haar_cascade.detectMultiScale(img,faces);

                for(int i=0;i<faces.size();i++)                                                     //  Loop through every  face detected in a frame
                {

                    if(faces[i].width<50||faces[i].height<50)   continue;                                   //  Ignore small rectangles. They are probably false positives


                    faces[i].x=max(faces[i].x-20,0);                                                //  Stretch the image        
                    faces[i].y=max(faces[i].y-30,0);
                    int bottom=min(faces[i].y+faces[i].height+30,img.rows-1);
                    int right=min(faces[i].x+faces[i].width+20,img.cols-1);
                    faces[i].width=right-faces[i].x;
                    faces[i].height=bottom-faces[i].y;
                    //cout<<0<<" "<<0<<" "<<img.cols-1<<" "<<img.rows-1<<endl;
                    //cout<< faces[i].x<< " "<< faces[i].y <<" "<< faces[i].x+faces[i].width<< " "<< faces[i].y+faces[i].height <<endl;

                    Mat instance=img(faces[i]);                                                     //  Crop only the face region.
                    if ( ! instance.isContinuous() )    instance = instance.clone();

                    //equalizeHist(instance,instance);


                    //copy(instance2,black,crop);
                    //imshow("segment",black);
                    //imshow("face",instance);

                    resize(instance,instance, Size(n,m),0,0, INTER_CUBIC);                          //  Resize the facial region to dimensions n*m. This is required for all models
                    instance=getBB(remove_blobs(GetSkin(instance,cr_min,cr_max,cb_min,cb_max)));
                    //cvtColor(instance,instance,CV_BGR2GRAY);
                    int pef=-1,p2d=-1,prc=-1;                                                       //  The predictions of 3 models are returned
            
                    pef=ef->predict(instance);
                    p2d=model2d.predict(instance);
                    prc=modelrc.predict(instance);

                    if(pef==prc==p2d)                                                               //  If all 3 are equal we give a vote of 1 per frame to the predicted class
                    {
                        attendance[pef-1]+=1*skip;
                    }
                    else if(pef==prc&&pef!=p2d)                                                     //  If any 2 are equal, we give the major class a vote of 2/3 per frame
                    {                                                                               //  and a vote of 1/3 per frame to the minor class
                        attendance[pef-1]+=(2.0/3)*skip;
                        attendance[p2d-1]+=(1.0/3)*skip;
                    }
                    else if(pef==p2d&&pef!=prc)
                    {
                        attendance[pef-1]+=(2.0/3)*skip;
                        attendance[prc-1]+=(1.0/3)*skip;
                    }
                    else if(p2d==prc&&p2d!=pef)
                    {
                        attendance[p2d-1]+=(2.0/3)*skip;
                        attendance[pef-1]+=(1.0/3)*skip;
                    }
                    else                                                                            //  If all 3 predictions are different we give each predicted class a vote of 1/3 per frame
                    {
                        attendance[pef-1]+=(1.0/3)*skip;
                        attendance[prc-1]+=(1.0/3)*skip;
                        attendance[p2d-1]+=(1.0/3)*skip;
                    }



                    rectangle(img,faces[i],CV_RGB(0,255,0),2);                                       //  We draw a green rectangle around the face


                    //  We write the strings that are to be displayed on top of each face //
                    char ef[50];
                    sprintf(ef," ef %s", prediction_name(pef).c_str());

                    char d2[50];
                    sprintf(d2," 2d %s", prediction_name(p2d).c_str());

                    char rc[50];
                    sprintf(rc," rc %s", prediction_name(prc).c_str());

                    //--------------------------------------------------------------------//

                    int pos_x = std::max(faces[i].tl().x - 10, 0);
                    int pos_y = std::max(faces[i].tl().y - 10, 0);

                    putText(img, ef, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);    //Put the text on the image showing the name of the model and the prediction
                    putText(img, d2, Point(pos_x, pos_y+15), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                    putText(img, rc, Point(pos_x, pos_y+30), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

                }

                cvNamedWindow("Detection and Recognition",WINDOW_NORMAL);
                cv::imshow("Detection and Recognition", img);                                       //  Show the image with detected faces and the predicted labels


                /*
                     *****************************   Attendance Marking  *******************************

                    *   For marking attendance of the students, we check the attendance array after
                    *   every 40 frames. This 40 can be increased or decreased. We loop through the
                    *   array and if we find any student having an attendance score more than a certain
                    *   threshold, we mark that student as present. We also display their names with a
                    *   Score with somewhat represents the percentage of time they were present in the
                    *   last 40 frames. This can cross 100 as there is a chance of false positives being
                    *   detected as faces during the detection process.

                */


                if(frames%40==0)
                {
                    int y=10;
                    att=black.clone();
                    for(int i=0;i<num_dir;i++)
                    {
                        if(attendance[i]>18)                                                        //  Threshold for marking present. Try changing this
                        {
                            char present[50];
                            sprintf(present,"%s Score %f",prediction_name(i+1).c_str(),attendance[i]*2.5);
                            putText(att, present, Point(10, y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                            y=y+15;
                        }
                        attendance[i]=0;
                    }
                    frames=0;                                                                       //  Reinitialize frames=0 so that the smae thing can be repeated
                }


                imshow("attendance",att);
                key = cv::waitKey(40);
                cam_movement(key,img);
            }

        }

}



//------------------------------------------------------------------//


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


