
///*
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

using namespace cv;

void cam_movement(int key,Mat img) //Keyboard commands to generate movements of the camera
{
        int i=0;
        char name[20];
      switch(key)
       {

       case 'a':
        system("curl http://root:pass123@192.168.137.89/axis-cgi/com/ptz.cgi?rpan=1");
        break;
       case 'd':
        system("curl http://root:pass123@192.168.137.89/axis-cgi/com/ptz.cgi?rpan=-1");
        break;
       case 's':
        system("curl http://root:pass123@192.168.137.89/axis-cgi/com/ptz.cgi?rtilt=1");
        break;
       case 'w':
        system("curl http://root:pass123@192.168.137.89/axis-cgi/com/ptz.cgi?rtilt=-1");
        break;
       case 'z':
        system("curl http://root:pass123@192.168.137.89/axis-cgi/com/ptz.cgi?rzoom=100");
        break;
       case 'x':
        system("curl http://root:pass123@192.168.137.89/axis-cgi/com/ptz.cgi?rzoom=-100");
        break;
       case 'p':
        system("curl http://root:pass123@192.168.137.89/axis-cgi/com/ptz.cgi?continuouspantiltmove=0,0");
        break;
       case 'o':
        system("curl http://root:pass123@192.168.137.89/axis-cgi/com/ptz.cgi?continuouspantiltmove=-5,0");
        break;
       case 'i':
       sprintf(name,"images/%i.png",i);
        cv::imwrite(name, img);
        i++;
        break;
       }
}

Mat clahe(Mat img) //Does a local histogram equalization to improve illumination
{
    Mat tmp;
    cvtColor(img,tmp,CV_BGR2Lab);
    std::vector<Mat>planes(3);
    split(tmp,planes);
    Ptr<CLAHE> cl=createCLAHE();
    cl->setClipLimit(4);
    Mat dst;
    cl->apply(planes[0],dst);
    dst.copyTo(planes[0]);
    merge(planes,tmp);
    cvtColor(tmp,img,CV_Lab2BGR);
    return img;

}

string prediction_name(int prediction)
{
    switch(prediction)
    {
        case 0 : return "Rahul";
        case -1:return "Unknown";
        case 4: return "Ankush";
        case 2: return "Srishty";
        case 1: return "Achammal";
        case 3: return "Jaymanyu";


    }
}

int main(int, char**) {
    VideoCapture vcap;
    Mat img,gray;
    char key,name[20];
    int i=0,count=-1,skip=4;
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

    while(1)
        {
            vcap.read(img);
            count++;
            if(count%skip==0)
            {
               // cv::imshow("Output Window1", img);
                img=clahe(img);
                Mat original =img.clone();
                cvtColor(img, gray, CV_BGR2GRAY);
                vector< Rect_<int> > faces;
                haar_cascade.detectMultiScale(gray,faces);
                for(int i=0;i<faces.size();i++)
                {
                    Rect crop=faces[i];
                    Mat instance=gray(crop);
                    if ( ! instance.isContinuous() )
                        {
                            instance = instance.clone();
                        }
                    resize(instance,instance, Size(120,120), 1.0, 1.0, INTER_CUBIC);
                    int plbp=lbp->predict(instance);
                    int pef=ef->predict(instance);
                    int pff=ff->predict(instance);


                    rectangle(img,crop,CV_RGB(0,255,0),2);
                    string lbp ="lbp: "+ prediction_name(plbp);
                    string ef = "ef: "+prediction_name(pef);
                    string ff = "ff: "+prediction_name(pff);
                    int pos_x = std::max(crop.tl().x - 10, 0);
                    int pos_y = std::max(crop.tl().y - 10, 0);
                    putText(img, lbp, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                    putText(img, ef, Point(pos_x, pos_y+15), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                    putText(img, ff, Point(pos_x, pos_y+30), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                }
                cv::imshow("Output Window2", img);


                key = cv::waitKey(30);
                cam_movement(key,img);

            }

        }

return 0;
}

//*/
