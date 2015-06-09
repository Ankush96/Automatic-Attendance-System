/*

//change the folder name
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



int main(int, char**) {
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
                segment=GetSkin(img);
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


*/