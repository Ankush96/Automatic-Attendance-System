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
    ******************************   Tune_seg_params    ********************************
    *
    *   Function that visualises the effect of segmentation and lets the user derive
    *   the best thresholds for segmentation. The thresholds found by the user must be
    *   updated in main.cpp
    ***********************************************************************************
*/
void tune_seg_params(string dir, int num_dir, int cr_min, int cr_max, int cb_min, int cb_max)
{
    vector<Mat> images;
    vector<int> labels;
    cout<<"Usage-\n\n\t1>\tPress <space> for next sample";
    cout<<"\n\t2>\tPress <Esc> to exit"<<endl;
    dir_read(dir,num_dir,images,labels,1);

    cvNamedWindow("src",WINDOW_NORMAL);
    cvNamedWindow("dst",WINDOW_NORMAL);

    createTrackbar("cr min ","dst",&cr_min,255);
    createTrackbar("cr max ","dst",&cr_max,255);
    createTrackbar("cb min ","dst",&cb_min,255);
    createTrackbar("cb max ","dst",&cb_max,255);

    for(int i=0;i<images.size();i++)
    {
      Mat src=images[i];
      imshow("src",src);
        while(1)
            {
                Mat dst=getBB(remove_blobs(GetSkin(src,cr_min,cr_max,cb_min,cb_max)));
                resize(dst,dst,Size(src.cols,src.rows),0,0,INTER_CUBIC);
                imshow("dst",dst);
                int key = cv::waitKey(0);
                if(key==32)break;
                if(key==27)return;
            }

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
    else
    {
        for(int i=0;i<images.size();i++)
            {
                Mat src=images[i];
                resize(src,src,Size(n,m),0,0,INTER_CUBIC);
                images[i]=src;
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

    model2d.train(images,labels,10,"2dpca.xml");
    modelrc.train(images,labels,6,"rc2dpca.xml");
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
void image_recognizer(string dir, int num_dir, int examples, int color, int cr_min, int cr_max, int cb_min, int cb_max){

        vector<Mat> images;                                                                 //  All the images are loaded along with their labels
        vector<int> labels;                                                                 //  only during the start of the function. This is done
        dir_read(dir,num_dir,images,labels,color);                                          //  just once. Any further change in training and testing
                                                                                            //  datasets is done by manipulating these two vectors
        cout<<"here"<<endl;
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
        else
        {
            for(int i=0;i<images.size();i++)                                                  
            {                                                                               
                Mat dst=images[i];
                resize(dst,dst,Size(n,m),0,0,INTER_CUBIC);                                  
                images[i]=dst;                                                    
            }
        }
        cout<<"here"<<endl;
        std::vector<Mat> images_test,images_train;                                          //  Declaring the vector of testing and training images for each case
        std::vector<int> labels_train,labels_test;                                          //  Declaring the vector of labels corresponding to the training and testing images
        double* accuracy = new double[examples*sizeof( double )];

        //cvNamedWindow("src",WINDOW_NORMAL);
        double y[101];
        fstream myfile("Plots/new_cvl_2d_3.txt", ios::out);                                           //  The accuracies are written into a text file for plotting purposes
        if (myfile.is_open()) cout<<"file exists"<<endl;
        for(int i=1;i<80;i++)                                                               //  This loop controls the number of eigenvectors retained for
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
                model.train(images_train,labels_train,i,"2dpca.xml");          //  Train the 2dpca model
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
            cout<<endl<<"eigenvectors "<<i<<" final accuracy -> "<<y[i]<<endl;
            myfile<<y[i]<<endl;                                                             //  Writing the accuracy onto the file
        }
        myfile.close();                                                                     //   Close the file after writing the accuracy values
}

int video_recognizer(int cr_min, int cr_max, int cb_min, int cb_max){
    VideoCapture vcap;
    Mat img,gray,sgray;

    Mat black(200,200,CV_8UC3,Scalar(0,0,0));                                                       //  Mat image to display final attendance
    Mat att=black.clone();

    char key,name[20];
    int i=0,count=-1,skip=10,y;
    int num_dir=9;                                                                                  //  Number of classes or unique identities
    double* attendance = new double[num_dir*sizeof( double )];

    int frames=-1;

    string videoStreamAddress = "rtsp://root:pass123@192.168.137.89:554/axis-media/media.amp";      //  Open the video stream and make sure it's opened
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
                //img=clahe(img);     
                                                                                //  Perform local histogram equalisation
                Mat black(img.rows,img.cols,CV_8UC3,Scalar(0,0,0));                                 //  Maintain a black Mat image to display attendance
                vector< Rect_<int> > faces;                                                         //  Initialise a vector of rectangles that store the detected faces

                //--------------Start detecting the faces in a frame------------------//
                haar_cascade.detectMultiScale(img,faces);

                for(int i=0;i<faces.size();i++)                                                     //  Loop through every  face detected in a frame
                {

                    if(faces[i].width<50||faces[i].height<50)   continue;                           //  Ignore small rectangles. They are probably false positives


                    // faces[i].x=max(faces[i].x-20,0);                                                //  Stretch the image
                    // faces[i].y=max(faces[i].y-30,0);
                    // int bottom=min(faces[i].y+faces[i].height+30,img.rows-1);
                    // int right=min(faces[i].x+faces[i].width+20,img.cols-1);
                    // faces[i].width=right-faces[i].x;
                    // faces[i].height=bottom-faces[i].y;
                    
                    Mat instance=img(faces[i]);                                                     //  Crop only the face region.
                    if ( ! instance.isContinuous() )    instance = instance.clone();



                    //copy(instance2,black,crop);
                    //imshow("segment",black);
                    //imshow("face",instance);

                    resize(instance,instance, Size(400,400),0,0, INTER_CUBIC);                      //  Resize the facial region to 400*400 for effective segmentation. This is required for all models
                    instance=getBB(remove_blobs(GetSkin(instance,cr_min,cr_max,cb_min,cb_max)));
                    resize(instance,instance, Size(n,m),0,0, INTER_CUBIC);                          //  This is necessary for the recognition
                    
                    //equalizeHist(instance,instance);
                    cvNamedWindow("face",WINDOW_NORMAL);
                    imshow("face",instance);
                    waitKey(30);
                    //cvtColor(instance,instance,CV_BGR2GRAY);
                    int pef=-1,p2d=-1,prc=-1;                                                       //  The predictions of 3 models are returned

                    pef=ef->predict(instance);
                    p2d=model2d.predict(instance);
                    prc=modelrc.predict(instance);
                    cout<<" pef "<<pef<<" p2d "<<p2d<<" prc "<<prc<<endl;
                    attendance[pef-1]+=(1.0/3)*skip;                                                //  Update the attendance scores of all identified people
                    attendance[prc-1]+=(1.0/3)*skip;
                    attendance[p2d-1]+=(1.0/3)*skip;

                    rectangle(img,faces[i],CV_RGB(0,255,0),2);                                      //  We draw a green rectangle around the face

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
                    int y=30;
                    att=black.clone();
                    for(int i=0;i<num_dir;i++)
                    {
                        if(attendance[i]>18)                                                        //  Threshold for marking present. Try changing this
                        {
                            char present[50];
                            sprintf(present,"%s Score %f",prediction_name(i+1).c_str(),attendance[i]*2.5);
                            putText(att, present, Point(30, y), FONT_HERSHEY_PLAIN, 3.0, CV_RGB(0,255,0), 2.0);
                            y=y+45;
                        }
                        attendance[i]=0;
                    }
                    frames=0;                                                                       //  Reinitialize frames=0 so that the same thing can be repeated
                }

                cvNamedWindow("attendance",WINDOW_NORMAL);
                imshow("attendance",att);
                key = cv::waitKey(40);
                if(key==27)
                    return 0;
                cam_movement(key,img);
            }

        }
return 0;
}

