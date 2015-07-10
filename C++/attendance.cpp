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

using namespace cv;
using namespace std;

#define n 120
#define m 120



/*
    ***************************   Prediction_Name    *********************************
    *
    *   Function that returns the name of the subject identified. The cases provided
    *   under the switch case needs to be changed according to needs. The numbers 
    *   correspond to the number of associated with the folder where the images are 
    *   stored. For example images of the person named "Ankush" are stored in the 
    *   folder "s1". Hence "Ankush" is associated with the label "1". This is similar
    *   for all other classes.
    **********************************************************************************

*/
string prediction_name(int prediction){
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

/*
    *******************************    Model_Main   ************************************
    *
    *   This function trains different models based on the images provided in the 
    *   directory "dir", which has "num_dir" different classes. The bool variable "color"
    *   is set to 0 if we do not want a segmented image to be trained, and to 1 if we 
    *   want to extract the facial region using segmentation before training the images.
    *   The 4 parameters after "color" define the thresholds applied on the Cr and Cb
    *   colorspace.
    ************************************************************************************
*/
void model_main(string dir, int num_dir, bool color, int cr_min, int cr_max, int cb_min, int cb_max){
    vector<Mat> images;
    vector<int> labels;
    dir_read(dir,num_dir,images,labels,color);

    if(color)
    {
        for(int i=0;i<images.size();i++)
            {
                Mat src=images[i];
                Mat dst=getBB(remove_blobs(GetSkin(src,cr_min,cr_max,cb_min,cb_max)));
                resize(dst,dst,Size(n,m),0,0,INTER_CUBIC);
                images[i]=dst;
            }
    }


    //Ptr<FaceRecognizer> lbp = createLBPHFaceRecognizer();
    //Ptr<FaceRecognizer> ff =createFisherFaceRecognizer();
    Ptr<FaceRecognizer> ef =createEigenFaceRecognizer();
    pca2d model2d;
    rc2dpca modelrc;

    // cvNamedWindow("Face",WINDOW_NORMAL);
    // imshow("Face",images[1]);
    // cvWaitKey(0);

    //lbp->train(images, labels);
    //lbp->save("lbp.xml");

    //ff->train(images, labels);
    //ff->save("ff.xml");

    ef->train(images, labels);
    ef->save("ef.xml");

    model2d.train(images,labels,0.6,"2dpca.xml");
    modelrc.train(images,labels,0.63,"rc2dpca.xml");
}

/*
    ************************************    Image_Recognizer    ***********************************
    *
    *   This function lets the user evaluate the trained models by performing a series of
    *   leave one out cross-validation on the dataset provided in the directory "dir".
    *   In each round of cross-validation, one sample of each class is collected in the 
    *   test set, and the remaining ("examples"-1) samples are used for training a particular 
    *   model. The accuracies are tested on the the testing set. The k_th round of cross-validation
    *   uses the k-th sample from each class as part of the testing set. After "examples" number
    *   of rounds of cross-validation, the accuracies over all rounds are averaged, thus giving
    *   us a final measure of accuracy.
    ************************************************************************************************
*/
int image_recognizer(string dir, int num_dir, int examples, int color, int cr_min, int cr_max, int cb_min, int cb_max){

        vector<Mat> images;                                                                 //  All the images are loaded along with their labels
        vector<int> labels;                                                                 //  only during the start of the function. This is done
        dir_read(dir,num_dir,images,labels,color);                                          //  just once. Any further change in training and testing
                                                                                            //  datasets is done by manipulating these two vectors 

        pca2d model;                                                                        //  Declaring the models to be tested 
        if(color)                                                                          
        {                                                                                   //  If the color is 0, no segmentation is performed
          for(int i=0;i<images.size();i++)                                                  //  Else for every image in the vector "images" 
            {                                                                               //  Segmentation is carried out to extract only the faces
                Mat src=images[i];
                Mat dst=getBB(remove_blobs(GetSkin(src,cr_min,cr_max,cb_min,cb_max)));      //  The largest blob is retained and a bounding box is put around it
                resize(dst,dst,Size(n,m),0,0,INTER_CUBIC);                                  //  The bounding box is resized to the same size as the images that were trained 
                images[i]=dst;                                                              //  The vector "images" is updated to contain the segmented images
            }
        }

        std::vector<Mat> images_test,images_train;                                          //  Declaring the vector of testing and training images for each case
        std::vector<int> labels_train,labels_test;                                          //  Declaring the vector of labels corresponding to the training and testing images
        double* accuracy = new double[examples*sizeof( double )];

        //cvNamedWindow("src",WINDOW_NORMAL);
        double y[101];
        fstream myfile("Plots/o3.txt", ios::out);                                           //  The accuracies are written into a text file for plotting purposes
        if (myfile.is_open()) cout<<"file exists"<<endl;
        for(int i=0;i<36;i++)                                                               //  This loop controls the threshold of percentage information retained for 
        {                                                                                   //  training. Different ranges can be tried out here.

            for(int k=0;k<examples;k++)                                                     //  This loop controls the rounds of cross-validation as it goes
            {                                                                               //  through all samples and choses 1 per class for testing
                //cout<<" K= "<<k<<endl;
                accuracy[k]=0;                                                              //  This stores the accuracies for every round
                images_train.clear();
                images_test.clear();
                labels_train.clear();
                labels_test.clear();
                for(int i=0;i<images.size();i++)
                {
                    if(i%examples==k)                                                       // Put in testing set 
                    {
                        images_test.push_back(images[i]);
                        labels_test.push_back(labels[i]);
                    }
                    else                                                                    // Put in training set
                    {
                        images_train.push_back(images[i]);
                        labels_train.push_back(labels[i]);
                    }
                }
                model.train(images_train,labels_train,(29+2*i)/100.0,"2dpca.xml");          //  Train the 2dpca model
                //Ptr<FaceRecognizer> model = createEigenFaceRecognizer(4*(i+1));           //  Initialise a model for Eigenfaces. If this is uncommented all corresponding code related to EF has to be uncommented
                //model->train(images_train, labels_train);                                 //  Train the Eigenfaces model
                for(int j=0;j<images_test.size();j++)
                {
                    int prediction=  model.predict(images_test[j]);                         //  Prediction for 2DPCA 
                    //int prediction=  model->predict(images_test[j]);                      //  Prediction for eigenfaces


                    //imshow("src",images_test[j]);
                    //cout<<" actual -> "<<labels_test[j]<<" predicted ->"<<prediction<<endl;
                    //waitKey(0);
                    accuracy[k]+=(prediction==labels_test[j]);                              //  Accuracy is updated according to the prediction made by the model

                    //------Uncomment the following to see the misclassified images---------//
                    // if(prediction!=labels_test[j])
                    // {
                    //     cvNamedWindow("Incorrect",WINDOW_NORMAL);
                    //     imshow("Incorrect",images_test[j]);
                    //     waitKey(0);
                    // }
                }

                //cout<<" accuracy for k="<<k<<" is "<<accuracy[k]<<" "<< (accuracy[k]*100)/(labels_test.size())<<endl;
            }

            for(int k=1;k<examples;k++)
            {
                accuracy[k]+=accuracy[k-1];                                                 //  Calculating the cumulative accuracy
            }

            y[i]=(accuracy[examples-1]*100)/(examples*num_dir);                             //  Normalising the accuracy value
            cout<<endl<<"percentage"<<(29+2*i)<<" final accuracy -> "<<y[i]<<endl;
            myfile<<y[i]<<endl;                                                             //  Writing the accuracy onto the file
        }
        myfile.close();                                                                     //   Close the file after writing the accuracy values
}

int video_recognizer(){
	VideoCapture vcap;
    Mat img,gray,sgray;

    Mat black(500,500,CV_8UC3,Scalar(0,0,0));       //  Mat image to display final attendance
    Mat att=black.clone();

    char key,name[20];
    int i=0,count=-1,skip=5,y;
    double attendance[9];
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

    while(1)
        {
            vcap.read(img);
            frames++;
            count++;
            if(count%skip==0)
            {
               // cv::imshow("Output Window1", img);
                img=clahe(img);
                //Mat black(img.rows,img.cols,CV_8UC3,Scalar(0,0,0));
                Mat segment=img.clone();
                segment=GetSkin(img,128,164,115,160);  //change the thresholds
                cvtColor(img, gray, CV_BGR2GRAY);
                cvtColor(segment, sgray, CV_BGR2GRAY);
                vector< Rect_<int> > faces;

                //--------------Start detecting the faces in a frame------------------//
                haar_cascade.detectMultiScale(gray,faces);
                for(int i=0;i<faces.size();i++)
                {
                    Rect crop=faces[i];
                    Mat instance=sgray(crop);
                    equalizeHist(instance,instance);
                    //Mat instance=sgray(crop);
                    if ( ! instance.isContinuous() )
                        {
                            instance = instance.clone();
                        }
                    //copy(instance2,black,crop);
                    //imshow("segment",black);
                    //imshow("face",instance);
                    resize(instance,instance, Size(m,n), 1.0, 1.0, INTER_CUBIC);

                    int pef=-1,pff=-1,plbp=-1;
                    double conf_ef=0.0,conf_ff=0.0,conf_lbp=0.0;
                    ef->predict(instance,pef,conf_ef);

                    if(pef==pff)
                    {
                        attendance[1+pef]+=5;
                    }
                    else
                    {
                        attendance[1+pef]+=1.67;
                        attendance[1+pff]+=1.67;
                    }


                    rectangle(img,crop,CV_RGB(0,255,0),2);


                    char ef[50];
                    sprintf(ef," ef %s Conf- %f",prediction_name(pef).c_str(),conf_ef);

                    int pos_x = std::max(crop.tl().x - 10, 0);
                    int pos_y = std::max(crop.tl().y - 10, 0);
                  //  putText(img, lbp, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                    putText(img, ef, Point(pos_x, pos_y+15), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

                }
                cv::imshow("Output Window2", img);




                if(frames%30==0)
                {   int y=10;
                    att=black.clone();
                    attendance[0]=0;
                    for(int i=1;i<7;i++)
                    {
                        if(attendance[i]>15)
                        {
                            char present[50];
                            sprintf(present,"%s confidence %f",prediction_name(i-1).c_str(),attendance[i]*3.33);
                            putText(att, present, Point(10, y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                            y=y+15;
                        }
                        attendance[i]=0;
                    }
                    frames=0;

                }

                /*
                key=cv::waitKey(40);
                cam_movement(key,img);
                    y=10;
                    att=black.clone();
                    for(int i=1;i<7;i++)
                    {
                        if((attendance[i]*100)/frames>70)
                        {
                            char present[50];
                            sprintf(present,"%s recog rate %f",prediction_name(i-1).c_str(),(attendance[i]*100)/frames);
                            putText(att, present, Point(10, y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                            y=y+10;

                        }


                    }
                */
                imshow("attendance",att);
                key = cv::waitKey(40);
                cam_movement(key,img);
            }

        }
    return 0;
}

