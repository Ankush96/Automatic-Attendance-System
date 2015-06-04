
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include "math.h"

using namespace cv;

using std::cout;
using std::endl;

int cr_min=128,cr_max=164,cb_min=115,cb_max=160,kernel=8;

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

Mat erode(Mat const &src,int thresh=5)
{
    int i,j,l,k,count;
    Mat dst=src.clone();
    for(i=1;i<src.rows;i++)
    {
        for(j=1;j<src.cols;j++)
        {
            if(src.at<uchar>(i,j)==255)
            {
                count=0;
                 for(l=i-1;l<=i+1;l++)
                 {
                    for(k=j-1;k<=j+1;k++)
                        {
                            if(!(i==l&&j==k)&&src.at<uchar>(l,k)==255)
                            {
                                count ++;
                            }

                        }

                 }
                 if(count<thresh) dst.at<uchar>(i,j)=0;
            }

        }
    }

    //namedWindow("erode",WINDOW_NORMAL);
    //imshow("erode",dst);
    return dst;
}


Mat dilate(Mat const &src,int thresh=2)
{
    int i,j,l,k,count;
    Mat dst=src.clone();
    for(i=1;i<src.rows;i++)
    {
        for(j=1;j<src.cols;j++)
        {
            if(src.at<uchar>(i,j)<255)
            {
                count=0;
                 for(l=i-1;l<=i+1;l++)
                 {
                    for(k=j-1;k<=j+1;k++)
                        {
                            if(!(i==l&&j==k)&&src.at<uchar>(l,k)==255)
                            {
                                count ++;
                            }

                        }

                 }
                 if(count>thresh) dst.at<uchar>(i,j)=255;
            }

        }
    }

    // namedWindow("dilate",WINDOW_NORMAL);
    // imshow("dilate",dst);
    return dst;
}


Mat erode_dilate(Mat const &src)
{
    int i,j;
    Mat dst=src.clone();
    for(i=0;i<dst.rows;i++)
    {
        dst.at<uchar>(i,0)=0;
        dst.at<uchar>(i,src.cols-1)=0;

    }
    for(i=0;i<dst.cols;i++)
    {
        dst.at<uchar>(0,i)=0;
        dst.at<uchar>(src.rows-1,i)=0;

    }
    dst=(dilate(erode(dst)));
    for(i=0;i<dst.rows;i++)
    {
        for(j=0;j<dst.cols;j++)
        {
            if(dst.at<uchar>(i,j)<255) dst.at<uchar>(i,j)=0;
        }
    }
    namedWindow("erode_dilate stage 2",WINDOW_NORMAL);
    imshow("erode_dilate stage 2",dst);
    waitKey(0);
    return dst;
}


float stddev(Vector<int> w)
{
    int i;
    float mean=0,sum=0;
    for(i=0;i<w.size();i++) sum+=w[i];
    mean=sum/w.size();
    sum=0;
    for(i=0;i<w.size();i++) sum+=(w[i]-mean)*(w[i]-mean);
    return sqrt((sum/w.size()));
}

bool isBoundary(Mat const &src,int i,int j)
{
    int k,l;
    for(k=i-1;k<=i+1;k++)
    {
        for(l=j-1;l<=j+1;l++)
        {
            if((src.at<uchar>(i,j)!=src.at<uchar>(k,l))&&(k>=0)&&(l>=0)&&(k<src.rows)&&(l<src.cols)&&(i!=k&&j!=l)) return 1;
        }
    }
    return 0;
}
bool R1(int R, int G, int B) {
    bool e1 = (R>95) && (G>40) && (B>20) && ((max(R,max(G,B)) - min(R, min(G,B)))>15) && (abs(R-G)>15) && (R>G) && (R>B);
    bool e2 = (R>220) && (G>210) && (B>170) && (abs(R-G)<=15) && (R>B) && (G>B);
    return (e1||e2);
}

bool R2(float Y, float Cr, float Cb) {
   /* bool e3 = Cr <= 1.5862*Cb+20;
    bool e4 = Cr >= 0.3448*Cb+76.2069;
    bool e5 = Cr >= -4.5652*Cb+234.5652;
    bool e6 = Cr <= -1.15*Cb+301.75;
    bool e7 = Cr <= -2.2857*Cb+432.85;
    return e3 && e4 && e5 && e6 && e7;
    */
    bool e3= Cr>=cr_min;
    bool e4= Cr<=cr_max;
    bool e5= Cb>=cb_min;
    bool e6= Cb<=cb_max;
    return e3 && e4 && e5 && e6;
}

bool R3(float H, float S, float V)
{
    return (H<25) || (H > 230);
}

Mat stage1(Mat const &src) {
    // allocate the result matrix
    Mat dst = src.clone();
    Vec3b cwhite = Vec3b::all(255);
    Vec3b cblack = Vec3b::all(0);

    Mat src_ycrcb, src_hsv;
    // OpenCV scales the YCrCb components, so that they
    // cover the whole value range of [0,255], so there's
    // no need to scale the values:
    cvtColor(src, src_ycrcb, CV_BGR2YCrCb);
    //vector<Mat> planes;
    //split( src_ycrcb, planes );
    //imshow("cr",planes[1]);
    //imshow("cb",planes[2]);
    // OpenCV scales the Hue Channel to [0,180] for
    // 8bit images, so make sure we are operating on
    // the full spectrum from [0,360] by using floating
    // point precision:
    src.convertTo(src_hsv, CV_32FC3);
    cvtColor(src_hsv, src_hsv, CV_BGR2HSV);
    //vector<Mat> planes2;
    //split( src_hsv, planes2 );
     // imshow("h",planes2[0]);
     //imshow("s",planes2[1]);
    // Now scale the values between [0,255]:
    normalize(src_hsv, src_hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);

    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {

            Vec3b pix_bgr = src.ptr<Vec3b>(i)[j];
            int B = pix_bgr.val[0];
            int G = pix_bgr.val[1];
            int R = pix_bgr.val[2];
            // apply rgb rule
            bool a = R1(R,G,B);

            Vec3b pix_ycrcb = src_ycrcb.ptr<Vec3b>(i)[j];
            int Y = pix_ycrcb.val[0];
            int Cr = pix_ycrcb.val[1];
            int Cb = pix_ycrcb.val[2];
            // apply ycrcb rule
            bool b = R2(Y,Cr,Cb);

            Vec3f pix_hsv = src_hsv.ptr<Vec3f>(i)[j];
            float H = pix_hsv.val[0];
            float S = pix_hsv.val[1];
            float V = pix_hsv.val[2];
            // apply hsv rule
            bool c = R3(H,S,V);

            if(!(b))
                dst.ptr<Vec3b>(i)[j] = cblack;
            else
                dst.ptr<Vec3b>(i)[j] = cwhite;
        }
    }
    namedWindow("Stage1",WINDOW_NORMAL);
    imshow("Stage1",dst);
    waitKey(0);
    return dst;
}

Mat stage2(Mat const &Csrc)
{
    Mat src;
    cvtColor(Csrc,src,CV_BGR2GRAY);
    int i,j,k,l,density;
    Mat dst(src.rows/kernel,src.cols/kernel,CV_8UC1,Scalar(0));
    for(i=0;i<src.rows;i=i+kernel)
    {
        for(j=0;j<src.cols;j=j+kernel)
        {
            density=0;
            for(l=0;l<min(kernel,src.rows-i+1);l++)
            {
                for(k=0;k<min(kernel,src.cols-j+1);k++)
                {
                    density+=src.at<uchar>(i+l,j+k);

                }

            }
            density=density/(kernel*kernel);
            dst.at<uchar>(i/kernel,j/kernel)=density;

            /*
            for(l=0;l<min(kernel,src.rows-i+1);l++)
            {
                for(k=0;k<min(kernel,src.cols-j+1);k++)
                {
                    dst.at<uchar>(i+l,j+k)=density;

                }

            }
            */

        }

    }

    namedWindow("density map",WINDOW_NORMAL);
    imshow("density map",dst);
    waitKey(0);
    return erode_dilate(dst);
}

Mat stage3(Mat const &src,Mat const &img,int thresh=2)
{
    int i,j,k,l;
    Mat gray;
    cvtColor(src,gray,CV_BGR2GRAY);
    Mat dst=img.clone();
    for(i=0;i<img.rows;i++)
    {
        for(j=0;j<img.cols;j++)
        {
            if(img.at<uchar>(i,j)==255)
            {
                Vector<int> w;
                for(l=0;l<min(kernel,gray.rows-kernel*i);l++)
                {
                    for(k=0;k<min(kernel,gray.cols-kernel*j);k++)
                    {
                        w.push_back(gray.at<uchar>(i*kernel+l,j*kernel+k));
                    }
                }
                if(stddev(w)<=thresh)dst.at<uchar>(i,j)=0;
            }
        }

    }
    namedWindow("stddev_stage 3",WINDOW_NORMAL);
    imshow("stddev_stage 3",dst);
    waitKey(0);
    return dst;
}


Mat stage4(Mat const &img,int thresh=4)
{
    Mat dst=img.clone();

    int i,j,start,k,l,count;

    for(i=1;i<dst.rows-1;i++)
    {
        for(j=1;j<dst.cols-1;j++)
        {
            if(dst.at<uchar>(i,j)==255)
            {
                 count=0;
                 for(l=i-1;l<=i+1;l++)
                 {
                    for(k=j-1;k<=j+1;k++)
                        {

                            if(!(i==l&&j==k)&&dst.at<uchar>(l,k)==255)
                            {
                                count ++;
                            }

                        }

                 }
                 if(count>3) dst.at<uchar>(i,j)=255;
            }

        }

    }
    for(i=1;i<dst.rows-1;i++)
    {
        for(j=1;j<dst.cols-1;j++)
        {
            if(dst.at<uchar>(i,j)==0)
            {
                 count=0;
                 for(l=i-1;l<=i+1;l++)
                 {
                    for(k=j-1;k<=j+1;k++)
                        {

                            if(!(i==l&&j==k)&&dst.at<uchar>(l,k)==255)
                            {
                                count ++;
                            }

                        }

                 }
                 if(count>5) dst.at<uchar>(i,j)=255;
            }

        }

    }


    Mat dst2=dst.clone();
    for(i=0;i<dst.rows;i++)
    {
        for(j=0;j<dst.cols;j++)
        {
            if(dst.at<uchar>(i,j)==255)
            {
                start=j;
                while(dst.at<uchar>(i,j)==255&&j<dst.cols)j++;
                if((j-start)<thresh)
                    for(k=start;k<j;k++)
                    {
                        dst2.at<uchar>(i,k)=0;
                    }

            }
        }
    }

     for(i=0;i<dst.cols;i++)
    {
        for(j=0;j<dst.rows;j++)
        {
            if(dst.at<uchar>(j,i)==255)
            {
                start=j;
                while(dst.at<uchar>(j,i)==255&&j<dst.rows)j++;
                if((j-start)<thresh)
                    for(k=start;k<j;k++)
                    {
                        dst2.at<uchar>(k,i)=0;
                    }

            }
        }
    }
    namedWindow("Stage4 geo correct",WINDOW_NORMAL);
    imshow("Stage4 geo correct",dst2);
    waitKey(0);
    //cout<<"sTAGE 4 EXIT"<<dst2.rows<<" "<<dst2.cols<<endl;
    return dst2;

}

Mat stage5(Mat const &cs1,Mat const &s4)
{
    //Mat dst(cs1.rows,cs1.cols,CV_8UC1,Scalar(0));
    //cout<<"stage1"<<s1.rows<<" "<<s1.cols<<endl;
    //cout<<"stage4"<<s4.rows<<" "<<s4.cols<<endl;
    //cout<<" channels of s1"<<s1.channels()<<endl;
    //Mat s1(cs1.rows,cs1.cols,CV_8UC1,Scalar(0));
    //Mat s4(cs4.rows,cs4.cols,CV_8UC1,Scalar(0));
    //cvtColor(cs1,s1,CV_BGR2GRAY);
    //cvtColor(cs4,s4,CV_BGR2GRAY);
    Mat dst=cs1.clone();
    Mat s1=cs1.clone();
    //Mat s4=cs4.clone();

    cvtColor(dst,dst,CV_BGR2GRAY);

    cvtColor(s1,s1,CV_BGR2GRAY);
    //cvtColor(s4,s4,CV_BGR2GRAY);
    cout<<"s1<<"<<s1.channels()<<" "<<s1.depth()<<endl;
    cout<<"s4<<"<<s4.channels()<<" "<<s4.depth()<<endl;
    int i,j,k,l;
    for(i=0;i<dst.rows;i++)
    {
        for(j=0;j<dst.cols;j++)
        {
            dst.at<uchar>(i,j)=0;
        }
    }
    //imshow("dst",dst);
    //imshow(" s1" ,s1);

    for(i=0;i<s4.rows;i++)
    {
        for(j=0;j<s4.cols;j++)
        {

            //for(k=0;k<min(kernel,s1.rows-i*kernel);k++)
          /*
                for(k=0;k<kernel;k++)
                    {
                        for(l=0;l<kernel;l++)
                        {
                            //cout<<i*kernel+k<<" "<<j*kernel+l<<" "<<s1.at<uchar>(i*kernel+k,j*kernel+l)<<endl;
                            //cout<<i*kernel+k<<" "<<j*kernel+l<<" \t";
                            dst.at<uchar>(i*kernel+k,j*kernel+l)=s1.at<uchar>(i*kernel+k,j*kernel+l);
                            //dst.at<uchar>(i*kernel+k,j*kernel+l)=(255*j*kernel+l)/(1+s4.cols*kernel+kernel);
                            //dst.at<uchar>(i*kernel+k,j*kernel+l)=255-k*l*4;
                        }
                    }*/
               if(isBoundary(s4,i,j))
                {
                    //dst.at<uchar>(i*kernel,j*kernel)=255;
                    

                    for(k=0;k<min(kernel,s1.rows-i*kernel);k++)
                    {
                        for(l=0;l<min(kernel,s1.cols-j*kernel);l++)
                        {
                            //cout<<i*kernel+k<<" "<<j*kernel+l<<" "<<s1.at<uchar>(i*kernel+k,j*kernel+l)<<endl;
                            dst.at<uchar>(i*kernel+k,j*kernel+l)=s1.at<uchar>(i*kernel+k,j*kernel+l);
                            //dst.at<uchar>(i*kernel+k,j*kernel+l)=255-k*l*4;
                        }
                    }

                }
                else
                {
                    for(k=0;k<min(kernel,s1.rows-i*kernel);k++)
                    {
                        for(l=0;l<min(kernel,s1.cols-j*kernel);l++)
                        {
                            dst.at<uchar>(i*kernel+k,j*kernel+l)=s4.at<uchar>(i,j);
                            //dst.at<uchar>(i*kernel+k,j*kernel+l)=s1.at<uchar>(i*kernel+k,j*kernel+l);
                        }
                    }
                }

        }
    }

    //cout<<s1.rows-dst.rows<<" "<<s1.cols-dst.cols;
    cout<<"dst"<<dst.rows<<" "<<dst.cols<<endl;
    //namedWindow("contour stage 5",WINDOW_AUTOSIZE);
    imshow("contour stage 5",dst);

    waitKey(0);
    //cout<<dst.rows<<" " <<dst.cols;
    return dst;
}

Mat GetSkin(Mat const &src)
{
    Mat s1=stage1(src);
    return stage5(s1,stage4(stage2(s1)));
}

int main()
 {

  ///*  // Load image & get skin proportions:
    Mat image = cv::imread("b15.jpeg");
    namedWindow("original");
    //namedWindow("skin",WINDOW_NORMAL);
    image=clahe(image);
    imshow("original", image);

    Mat skin = GetSkin(image);

    // Show the results:

    imshow("skin", skin);

    waitKey(0);
//*/

  /*
    VideoCapture vcap;
    Mat img,gray;
    char key,name[20];
    int i=0,count=-1,skip=2;
    const std::string videoStreamAddress = "rtsp://root:pass123@192.168.137.89:554/axis-media/media.amp";  //open the video stream and make sure it's opened
     if(!vcap.open(videoStreamAddress))
        {
            std::cout << "Error opening video stream or file" << std::endl;
            return -1;
        }
    namedWindow("original");
    namedWindow("skin",WINDOW_NORMAL);
    createTrackbar("cr min ","skin",&cr_min,255);
    createTrackbar("cr max ","skin",&cr_max,255);
    createTrackbar("cb min ","skin",&cb_min,255);
    createTrackbar("cb max ","skin",&cb_max,255);


    while(1)
        {
            vcap.read(img);
            count++;
            if(count%skip==0)
            {
               // cv::imshow("Output Window1", img);
                img=clahe(img);
                imshow("original", img);

                Mat skin = GetSkin(img);
                imshow("skin", skin);
                key = cv::waitKey(30);
               // cam_movement(key,img);

            }

        }
    //*/


    return 0;
}
