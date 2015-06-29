
#include <stdio.h>
#include "Stage-segment.h"
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

#define m 92
#define n 112


string prediction_name(int prediction)
{
    switch(prediction)
    {
        case -1:return "Unknown";
        case 0 : return "Srishty";
        case 1: return "Jayamani";
        case 2: return "Achammal";
        case 3: return "Manjunath";
        case 4: return "Ankush";
        case 5: return "Mridul";


    }
}

void dir_read(string root,int num,vector<Mat>& images,vector<int>& labels,bool color)
{
    DIR *dp;
    struct dirent *dirp;
    struct stat filestat;
    int i;
    string filepath;
    for(i=1;i<=num;i++)
    {
        string dir;
        char sub[3];
        sprintf(sub,"%d",i);
        dir=root + "/s" + sub + "/" ;
        dp = opendir( dir.c_str() );
        while(dirp=readdir(dp))
        {
            filepath = dir + "/" + dirp->d_name;
            if (stat( filepath.c_str(), &filestat )) continue;
            if (S_ISDIR( filestat.st_mode )) continue;
            Mat img=imread(filepath,color);
            images.push_back(img);
            labels.push_back(i);
        }
        closedir(dp);
    }
}

//------------main.cpp----------------//


int image_recognizer(string dir)
{
    Mat img,gray,sgray;
    //Mat black(500,500,CV_8UC3,Scalar(0,0,0));
    //Mat att=black.clone();
    char key,name[20];
    CascadeClassifier haar_cascade;
    haar_cascade.load("../Cascades/front_alt2.xml");
    Ptr<FaceRecognizer> lbp = createLBPHFaceRecognizer(1,8,8,8,123.0);
    Ptr<FaceRecognizer> ef =createEigenFaceRecognizer();
    Ptr<FaceRecognizer> ff =createFisherFaceRecognizer();
    lbp->load("lbp.xml");
    ef->load("ef.xml");
    ff->load("ff.xml");

    //string filename = "samples.csv";
    //char separator = ';';

    vector<Mat> images;
    vector<int> labels;


    // std::ifstream file(filename.c_str(), ifstream::in);
    // if (!file) {
    //     string error_message = "No valid input file was given, please check the given filename.";
    //     CV_Error(CV_StsBadArg, error_message);
    // }
    // string line, path, classlabel;
    // while (getline(file, line)) {
    //     stringstream liness(line);
    //     getline(liness, path, separator);
    //     getline(liness, classlabel);
    //     if(!path.empty() && !classlabel.empty()) {
    //         Mat tmp1=imread(path, 1);
    //         Mat tmp2;
    //         resize(tmp1, tmp2, Size(m, n), 1.0, 1.0, INTER_CUBIC);
    //         ///tmp2=clahe(tmp2);
    //         //equalizeHist(tmp2,tmp2);
    //         //cvNamedWindow("image",WINDOW_NORMAL);
    //         //imshow("image",tmp2);
    //         //waitKey(0);
    //         //cvtColor(tmp2,tmp2,CV_BGR2GRAY);
    //         images.push_back(tmp2);
    //         labels.push_back(atoi(classlabel.c_str()));
    //     }
    // }
    //

    int correct,tot;
    double sum_ef=0,sum_ff=0,sum_mix=0;
    dir_read(dir,6,images,labels,1);
    tot=images.size();
    for(int i=images.size()-1;i>=0;i--)
    {
        cout<<images.size()<<" ";
        img=images[i];
        images.pop_back();
        switch(labels[i])
        {
            case 1:correct=4;
            break;
            case 2:correct=1;
            break;
            case 3:correct=2;
            break;
            case 4:correct=0;
            break;
            case 5:correct=3;
            break;
            case 6:correct=5;
            break;
        }
        Mat instance=img.clone();
        cvtColor(img,instance,CV_BGR2GRAY);
        equalizeHist(instance,instance);
        if ( ! instance.isContinuous() )
           {
                instance = instance.clone();
           }
        resize(instance,instance, Size(m,n), 1.0, 1.0, INTER_CUBIC);

        int pef=-1,pff=-1,plbp=-1;
        double conf_ef=0.0,conf_ff=0.0,conf_lbp=0.0;
        ef->predict(instance,pef,conf_ef);
        ff->predict(instance,pff,conf_ff);
        lbp->predict(instance,plbp,conf_lbp);
        //----------------Error metrics calculated -----------------
        if(pef==correct&&pff==correct)
        {
            sum_ff+=1;
            sum_ef+=1;
            sum_mix+=1;
        }
        else if(pef==correct)
        {
            sum_ef+=1;
            sum_mix+=(conf_ef/(conf_ef+conf_ff));
        }
        else if(pff==correct)
        {
            sum_ff+=1;
            sum_mix+=(conf_ff/(conf_ef+conf_ff));
        }

        char lbp[50];
        sprintf(lbp," lbp %s Conf- %f",prediction_name(plbp).c_str(),conf_lbp);
        char ef[50];
        sprintf(ef," ef %s Conf- %f",prediction_name(pef).c_str(),conf_ef);
        char ff[50];
        sprintf(ff," ff %s Conf- %f",prediction_name(pff).c_str(),conf_ff);
        int pos_x = 10;
        int pos_y = 10;
        putText(img, lbp, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        putText(img, ef, Point(pos_x, pos_y+15), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        putText(img, ff, Point(pos_x, pos_y+30), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        cout<<prediction_name(pef)<<endl;
        cvNamedWindow("Output Window2",WINDOW_NORMAL);
        cv::imshow("Output Window2", img);
        cv::waitKey(0);


    }
    cout<<"\n\t Recognition rate"<<endl;
    cout<<"\t eigenfaces- "<<(sum_ef*100)/tot<<endl;
    cout<<"\t fisherfaces- "<<(sum_ff*100)/tot<<endl;
    cout<<"\t combined- "<<(sum_mix*100)/tot<<endl;
    return 0;
}

int video_recognizer()
{
	VideoCapture vcap;
    Mat img,gray,sgray;
    Mat black(500,500,CV_8UC3,Scalar(0,0,0));
    Mat att=black.clone();
    char key,name[20];
    int i=0,count=-1,skip=5,y;
    double attendance[7];
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
    Ptr<FaceRecognizer> lbp = createLBPHFaceRecognizer(1,8,8,8,123.0);
    Ptr<FaceRecognizer> ef =createEigenFaceRecognizer();
    Ptr<FaceRecognizer> ff =createFisherFaceRecognizer();
    lbp->load("lbp.xml");
    ef->load("ef.xml");
    ff->load("ff.xml");
    //img=imread("../Faces2/s2/img3.jpg");

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
                segment=GetSkin(img,128,164,115,160);
                cvtColor(img, gray, CV_BGR2GRAY);
                cvtColor(segment, sgray, CV_BGR2GRAY);
                vector< Rect_<int> > faces;
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
                    ff->predict(instance,pff,conf_ff);
                    lbp->predict(instance,plbp,conf_lbp);
                    //char final[50];
                    if(pef==pff)
                    {
                        // double hyb_conf=(conf_ff+conf_ef)/2;
                        // sprintf(final," Hybrid %s Conf- %f",prediction_name(pef).c_str(),hyb_conf);
                        attendance[1+pef]+=5;
                    }
                    else
                    {
                        attendance[1+pef]+=1.67;
                        attendance[1+pff]+=1.67;
                        // if(conf_ef>conf_ff)
                        // {
                        //     sprintf(final," Hybrid %s Conf- %f",prediction_name(pef).c_str(),conf_ef);
                        // }
                        // else
                        // {
                        //     sprintf(final," Hybrid %s Conf- %f",prediction_name(pff).c_str(),conf_ff);
                        // }
                    }


                    rectangle(img,crop,CV_RGB(0,255,0),2);

                   // char lbp[50];
                    //sprintf(lbp," lbp %s Conf- %f",prediction_name(plbp).c_str(),conf_lbp);
                    char ef[50];
                    sprintf(ef," ef %s Conf- %f",prediction_name(pef).c_str(),conf_ef);
                    char ff[50];
                    sprintf(ff," ff %s Conf- %f",prediction_name(pff).c_str(),conf_ff);
                    int pos_x = std::max(crop.tl().x - 10, 0);
                    int pos_y = std::max(crop.tl().y - 10, 0);
                  //  putText(img, lbp, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                    putText(img, ef, Point(pos_x, pos_y+15), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                    putText(img, ff, Point(pos_x, pos_y+30), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
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
//-------------------------------------//

//------------model.cpp----------------//
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
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
            Mat tmp1=imread(path, 0);
            Mat tmp2;
            resize(tmp1, tmp2, Size(m, n), 1.0, 1.0, INTER_CUBIC);
            ///tmp2=clahe(tmp2);
            equalizeHist(tmp2,tmp2);
            //cvNamedWindow("image",WINDOW_NORMAL);
            //imshow("image",tmp2);
            //waitKey(0);
            //cvtColor(tmp2,tmp2,CV_BGR2GRAY);
            images.push_back(tmp2);
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int model_main(string dir)
{

    //string fn_csv = "samples.csv";

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
//---------------------------------------//

//-------------sampler------------------//
int sampler_main() {
    VideoCapture vcap;
    Mat img,gray,sgray;
    char key,name[20];
    int i=0,count=-1,skip=10,k=0;
    const std::string videoStreamAddress = "rtsp://root:pass123@192.168.137.89:554/axis-media/media.amp";  //open the video stream and make sure it's opened
    CascadeClassifier haar_cascade;
    haar_cascade.load("../Cascades/front_alt2.xml");

    if(!vcap.open(videoStreamAddress))
        {
            std::cout << "Error opening video stream or file" << std::endl;
            return -1;
        }
    cvNamedWindow("Face",WINDOW_NORMAL);
    while(1)
        {
            vcap.read(img);
            count++;
            if(count%skip==0)
            {
                img=clahe(img);
                Mat segment=img.clone();
                segment=GetSkin(img,128,164,115,160);
                cvtColor(segment, sgray, CV_BGR2GRAY);
                cvtColor(img, gray, CV_BGR2GRAY);
                vector< Rect_<int> > faces;
                haar_cascade.detectMultiScale(gray,faces);
                for(int i=0;i<faces.size();i++)
                {
                    Rect crop=faces[i];
                    //Mat instance=img(crop);
                    Mat instance=sgray(crop);
                    Mat instance2=gray(crop);
                    if ( ! instance.isContinuous() )
                        {
                            instance = instance.clone();
                        }
                    if ( ! instance2.isContinuous() )
                        {
                            instance2 = instance2.clone();
                        }
                    char filename[100];
                    sprintf(filename,"../Faces/s1/img%d.jpg",k);
                    imwrite(filename,instance);
                    sprintf(filename,"../Faces2/s1/img%d.jpg",k);
                    imwrite(filename,instance2);
                    k++;

                    rectangle(img,crop,CV_RGB(0,255,0),2);
                    cv::imshow("Face", instance);
                }
                cv::imshow("Output Window2", img);


                key = cv::waitKey(30);
                cam_movement(key,img);

            }

        }
}
