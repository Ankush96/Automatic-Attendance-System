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

int model_main(string dir)
{
    vector<Mat> images;
    vector<int> labels;
    /*
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }

    */
    dir_read(dir,6,images,labels,0);
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }



     Ptr<FaceRecognizer> lbp = createLBPHFaceRecognizer();
     Ptr<FaceRecognizer> ef =createEigenFaceRecognizer();
     Ptr<FaceRecognizer> ff =createFisherFaceRecognizer();
   //  cvNamedWindow("Face",WINDOW_NORMAL);
    // imshow("Face",images[1]);
    // cvWaitKey(0);

    lbp->train(images, labels);

    lbp->save("lbp.xml");
    ef->train(images, labels);

    ef->save("ef.xml");
    ff->train(images, labels);

    ff->save("ff.xml");
    // The following line predicts the label of a given
    // test image:


    //
    // To get the confidence of a prediction call the model with:
    //
    //      int predictedLabel = -1;
    //      double confidence = 0.0;
    //      model->predict(testSample, predictedLabel, confidence);
    //
  //  string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
  //  cout << result_message << endl;
    // Sometimes you'll need to get/set internal model data,
    // which isn't exposed by the public cv::FaceRecognizer.
    // Since each cv::FaceRecognizer is derived from a
    // cv::Algorithm, you can query the data.
    //
    // First we'll use it to set the threshold of the FaceRecognizer
    // to 0.0 without retraining the model. This can be useful if
    // you are evaluating the model:
    //
    //model->set("threshold", 0.0);
    // Now the threshold of this model is set to 0.0. A prediction
    // now returns -1, as it's impossible to have a distance below
    // it
    //predictedLabel = model->predict(testSample);
    //cout << "Predicted class = " << predictedLabel << endl;
    // Show some informations about the model, as there's no cool
    // Model data to display as in Eigenfaces/Fisherfaces.
    // Due to efficiency reasons the LBP images are not stored
    // within the model:
    //cout << "Model Information:" << endl;
    //string model_info = format("\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
     //       model->getInt("radius"),
    //        model->getInt("neighbors"),
     //       model->getInt("grid_x"),
    //        model->getInt("grid_y"),
    //        model->getDouble("threshold"));
    //cout << model_info << endl;
    //// We could get the histograms for example:
   // vector<Mat> histograms = model->getMatVector("histograms");
    // But should I really visualize it? Probably the length is interesting:
    //cout << "Size of the histograms: " << histograms[0].total() << endl;

    return 0;
}

int image_recognizer()
{
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

}

int video_recognizer()
{
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

