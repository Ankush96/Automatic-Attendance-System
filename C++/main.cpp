///*

#include <stdio.h>
#include "Stage-segment.h"
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace cv;

void copy (Mat small,Mat big,Rect roi)
{
    int x,y,i,j;
    for(y=roi.y,i=0;y<roi.y+roi.height;y++,i++)
    {
        for(x=roi.x,j=0;x<roi.x+roi.width;x++,j++)
        {
            big.at<Vec3b>(y,x)=small.at<Vec3b>(i,j);
        }
    }

}

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

int main(int, char**) {
    VideoCapture vcap;
    Mat img,gray,sgray;
    char key,name[20];
    int i=0,count=-1,skip=5;
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
    //Ptr<FaceRecognizer> lbp = createLBPHFaceRecognizer(1,8,8,8,123.0);
    Ptr<FaceRecognizer> ef =createEigenFaceRecognizer();
    Ptr<FaceRecognizer> ff =createFisherFaceRecognizer();
    //lbp->load("lbp.xml");
    ef->load("ef.xml");
    ff->load("ff.xml");
    cvNamedWindow("segment",WINDOW_NORMAL);
    while(1)
        {
            vcap.read(img);
            count++;
            if(count%skip==0)
            {
               // cv::imshow("Output Window1", img);
                img=clahe(img);
                //Mat black(img.rows,img.cols,CV_8UC3,Scalar(0,0,0));
                Mat segment=img.clone();
                segment=GetSkin(img);
                cvtColor(img, gray, CV_BGR2GRAY);
                cvtColor(segment, sgray, CV_BGR2GRAY);
                vector< Rect_<int> > faces;
                haar_cascade.detectMultiScale(gray,faces);
                for(int i=0;i<faces.size();i++)
                {
                    Rect crop=faces[i];
                    Mat instance=gray(crop);
                    //Mat instance=sgray(crop);
                    if ( ! instance.isContinuous() )
                        {
                            instance = instance.clone();
                        }
                    //copy(instance2,black,crop);
                    //imshow("segment",black);
                    imshow("face",instance);
                    resize(instance,instance, Size(120,120), 1.0, 1.0, INTER_CUBIC);
                    //int plbp=lbp->predict(instance);
                    int pef=-1,pff=-1;
                    double conf_ef=0.0,conf_ff=0.0;
                    ef->predict(instance,pef,conf_ef);
                    ff->predict(instance,pff,conf_ff);
                    conf_ff*=7;
                    char final[50];
                    if(pef==pff)
                    {
                        double hyb_conf=(conf_ff+conf_ef)/2;
                        sprintf(final," Hybrid %s Conf- %f",prediction_name(pef).c_str(),hyb_conf);
                    }
                    else
                    {
                        if(conf_ef>conf_ff)
                        {
                            sprintf(final," Hybrid %s Conf- %f",prediction_name(pef).c_str(),conf_ef);
                        }
                        else
                        {
                            sprintf(final," Hybrid %s Conf- %f",prediction_name(pff).c_str(),conf_ff);
                        }
                    }


                    rectangle(img,crop,CV_RGB(0,255,0),2);
                    //string lbp ="lbp: "+ prediction_name(plbp);
                    char ef[50];
                    sprintf(ef," ef %s Conf- %f",prediction_name(pef).c_str(),conf_ef);
                    char ff[50];                    
                    sprintf(ff," ff %s Conf- %f",prediction_name(pff).c_str(),conf_ff);
                    int pos_x = std::max(crop.tl().x - 10, 0);
                    int pos_y = std::max(crop.tl().y - 10, 0);
                    putText(img, final, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                    putText(img, ef, Point(pos_x, pos_y+15), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0), 2.0);
                    putText(img, ff, Point(pos_x, pos_y+30), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0), 2.0);
                }
                cv::imshow("Output Window2", img);


                key = cv::waitKey(30);
                cam_movement(key,img);

            }

        }

return 0;
}

//*/