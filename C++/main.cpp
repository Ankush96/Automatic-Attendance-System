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
    Mat black(500,500,CV_8UC3,Scalar(0,0,0));
    Mat att=black.clone();
    char key,name[20];
    int i=0,count=-1,skip=5,y;
    double attendance[7];
    int frames=0;
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
                segment=GetSkin(img);
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
                    resize(instance,instance, Size(120,120), 1.0, 1.0, INTER_CUBIC);

                    int pef=-1,pff=-1,plbp=-1;
                    double conf_ef=0.0,conf_ff=0.0,conf_lbp=0.0;
                    ef->predict(instance,pef,conf_ef);
                    ff->predict(instance,pff,conf_ff);
                    lbp->predict(instance,plbp,conf_lbp);
                    char final[50];
                    if(pef==pff)
                    {
                        double hyb_conf=(conf_ff+conf_ef)/2;
                        sprintf(final," Hybrid %s Conf- %f",prediction_name(pef).c_str(),hyb_conf);
                        attendance[1+pef]+=5;
                    }
                    else
                    {
                        attendance[1+pef]+=1.67;
                        attendance[1+pff]+=1.67;
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

                    char lbp[50];
                    sprintf(lbp," lbp %s Conf- %f",prediction_name(plbp).c_str(),conf_lbp);
                    char ef[50];
                    sprintf(ef," ef %s Conf- %f",prediction_name(pef).c_str(),conf_ef);
                    char ff[50];
                    sprintf(ff," ff %s Conf- %f",prediction_name(pff).c_str(),conf_ff);
                    int pos_x = std::max(crop.tl().x - 10, 0);
                    int pos_y = std::max(crop.tl().y - 10, 0);
                    putText(img, lbp, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                    putText(img, ef, Point(pos_x, pos_y+15), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                    putText(img, ff, Point(pos_x, pos_y+30), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                }
                cv::imshow("Output Window2", img);

                /*
                if(frames%2000!=0)
                {
                    key = cv::waitKey(40);
                    cam_movement(key,img);
                }
                else
                {   int y=10;
                    att=black.clone();
                    attendance[0]=0;
                    for(int i=1;i<7;i++)
                    {
                        if(attendance[i]>3000)
                        {
                            char present[50];
                            sprintf(present,"%s recog rate %f",prediction_name(i-1).c_str(),attendance[i]/50);
                            putText(att, present, Point(10, y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                            y=y+15;
                        }
                        attendance[i]=0;
                    }
                    frames=0;

                }
                */
                key=cv::waitKey(40);
                cam_movement(key,img);
                    y=10;
                    att=black.clone();
                    for(int i=1;i<7;i++)
                    {
                        if((attendance[i]*100)/frames>50)
                        {
                            char present[50];
                            sprintf(present,"%s recog rate %f",prediction_name(i-1).c_str(),(attendance[i]*100)/frames);
                            putText(att, present, Point(10, y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                            y=y+10;

                        }
                            
                        
                    }
                
                imshow("attendance",att);

                
            }

        }

return 0;
}

//*/
