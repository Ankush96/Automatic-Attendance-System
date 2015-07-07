
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

#define m 160
#define n 160



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
            // Rect crop(120,30,400,400);
            // imwrite(filepath,img(crop));
            images.push_back(img);
            labels.push_back(i);
        }
        closedir(dp);
    }
}
//-------------------------------------//

//--------------------------//
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
