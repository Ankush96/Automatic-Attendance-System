
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include "math.h"
#include <Eigen/Dense>

using namespace Eigen;
using namespace cv;

using std::cout;
using std::endl;
using std::queue;

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
    //cout<<"inside erode_dilate "<<endl;
    //cout<<" "<<src.rows<<" "<<src.cols<<endl;
    Mat dst(src.rows,src.cols,CV_8UC1,Scalar(0));
    //cout<<" "<<dst.rows<<" "<<dst.cols<<endl;
    for(i=0;i<dst.rows;i++)
    {
        dst.at<uchar>(i,0)=0;
        dst.at<uchar>(i,dst.cols-1)=0;

    }
    for(i=0;i<dst.cols;i++)
    {
        dst.at<uchar>(0,i)=0;
        dst.at<uchar>(src.rows-1,i)=0;

    }
    //cout<<"erode_dilate starts"<<endl;
    dst=(dilate(erode(src)));
    for(i=0;i<dst.rows;i++)
    {
        for(j=0;j<dst.cols;j++)
        {
            if(dst.at<uchar>(i,j)<255) dst.at<uchar>(i,j)=0;
        }
    }
    //namedWindow("erode_dilate stage 2",WINDOW_NORMAL);
    //imshow("erode_dilate stage 2",dst);
    //waitKey(0);

    //imwrite("s2.jpg",dst);
    return dst;
}

int bfs(Mat image,int x,int y,int** visited,int num_blobs)  //Detects the connected components of a pixel and returns the sum of all pixels in the blob
{
    //cout<<" new blob "<< num_blobs<<endl;
    queue<Point> q;
    Point p2,p(x,y);
    q.push(p);
    visited[x][y]=0;
    int i,j;
    int sum=1; //Denotes the sum of all pixels in the blob. Initialised as 1 because it contains the point (x,y)
    while(!q.empty())
    {
        for(i=p.x-1;i<=p.x+1;i++)
        {
            for(j=p.y-1;j<=p.y+1;j++)
            {
                if(visited[i][j]==-1&&image.at<uchar>(i,j)!=0)
                {
                    if(i>0&&i<image.rows-1&&j>0&&j<image.cols-1)
                    {
                        sum++;  // new pixel in the blob. sum updated
                        p2.x=i;
                        p2.y=j;
                        q.push(p2);
                        visited[i][j]=0;
                    }
                }
            }
        }
        visited[p.x][p.y]=num_blobs;
        q.pop();
        p=q.front();
    }
    return sum;
}

Mat remove_blobs(Mat const &img)
{
    Mat src=img.clone();
    if(src.channels()>1)
        cvtColor(src,src,CV_BGR2GRAY);

    //------------------Initialising visted array----------------------//
    int **visited=new int*[(src.rows*sizeof(int*))];

    for(int i=0;i<src.rows;i++)
    {
        visited[i]=new int[src.cols*sizeof(int)];
        for(int j=0;j<src.cols;j++) visited[i][j]=-1;
    }

    //-------------------------------------------------------------------//
    vector<int> blob_sizes;
    blob_sizes.push_back(-1);   //  The vector contains the size of all the blobs detected in the image. We want the indexing to start from 1. Hence we push -1 in the 0th position
    int num_blobs=0;

     for (int i = 1; i < src.rows-1; i++)
     {
        for (int j = 1; j < src.cols-1; j++)
        {
                                            //A new blob has been found
            if((src.at<uchar>(i,j)!=0)&&(visited[i][j]==-1)) // if node is not visited and not black
             {
                //cout<<" i j = "<<i<<" "<<j<<endl;
                num_blobs++;
                blob_sizes.push_back(bfs(src,i,j,visited,num_blobs));
             }

        }
     }
     //cout<<num_blobs<<" "<<src.rows*src.cols<<endl;

     //---------Once we have the visited array we can use the information to manipulation the image contained in src-------------//

     //---------We first check for the largest blob by using the vector blob_sizes----------------//

     int max=1;
     for(int i=max;i<blob_sizes.size();i++)
     {
        if(blob_sizes[i]>blob_sizes[max])
        {
            max=i;
        }
     }

    //----------- Now keep only the largest blob. Delete all the other blobs by making them black--------------//
     for (int i = 0; i < src.rows; i++)
     {
        for (int j = 0; j < src.cols; j++)
        {
            if(i==0||j==0||i==src.rows-1||j==src.cols-1)
            {
                src.at<uchar>(i,j)=0;
            }
            else
            {
               if(visited[i][j]!=max)
                {
                    src.at<uchar>(i,j)=0;
                } 
            }
            

        }
     }

    return src;
}

//-------------Function that calculates a bounding box(having same aspect ratio as the image) on a non-black object, and returns the cropped Bounding Box-----------//
Mat getBB(Mat const& img)
{
    Mat src=img.clone();
    int left=src.cols-1,right=0,top=src.rows-1,bottom=0;
    for(int i=0;i<src.rows;i++)
    {
        for(int j=0;j<src.cols;j++)
        {
            if(src.at<uchar>(i,j)!=0)
            {
                if(i<top) top=i;
                if(i>bottom)
                {

                    bottom=i;
                    //----------The commented code is a visualization of how the points move when bottom gets updated------------//

                    // cv::namedWindow("circle",WINDOW_NORMAL);
                    // Mat src2=img.clone();
                    // circle(src2,Point(left,top), 20 , Scalar(200) );
                    // circle(src2,Point(right,top), 20 , Scalar(200) );
                    // circle(src2,Point(left,bottom),20 , Scalar(200) );
                    // circle(src2,Point(right,bottom), 20 , Scalar(200) );
                    // imshow("circle",src2);
                    // waitKey(0);
                    //----------------------------------------------------------------//
                }    
                if(j<left) left=j;
                if(j>right) right=j;
                

            }
        }
    }
    //double ratio_h2w=(src.rows*1.0)/src.cols;

    //int bb_height=bottom-top, bb_width= right-left;
    
    //-------We include some space beacuse we dont require an exact bounding box-----------//
    left=max(left-10,0);
    top=max(top-10,0);
    right=min(right+10,src.cols-1);
    bottom=min(bottom+10,src.rows-1);

    Rect roi(left,top,right-left,bottom-top);


    return src(roi);
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
    bool e3= Cr>=cr_min;
    bool e4= Cr<=cr_max;
    bool e5= Cb>=cb_min;
    bool e6= Cb<=cb_max;
    return e3 && e4 && e5 && e6;
}

bool R3(float H, float S, float V){

    return (H<25) || (H > 230);
}

Mat stage1(Mat const &src) {
    Mat dst = src.clone();
    Vec3b cwhite = Vec3b::all(255);
    Vec3b cblack = Vec3b::all(0);

    Mat src_ycrcb, src_hsv;
    // OpenCV scales the YCrCb components, so that they
    // cover the whole value range of [0,255], so there's
    // no need to scale the values:
    cvtColor(src, src_ycrcb, CV_BGR2YCrCb);
    // OpenCV scales the Hue Channel to [0,180] for
    // 8bit images, so make sure we are operating on
    // the full spectrum from [0,360] by using floating
    // point precision:
    src.convertTo(src_hsv, CV_32FC3);
    cvtColor(src_hsv, src_hsv, CV_BGR2HSV);
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
    //namedWindow("Stage1",WINDOW_NORMAL);
    //imshow("Stage1",dst);
    //imwrite("s1.jpg",dst);
    //waitKey(0);
    return dst;
}

Mat stage2(Mat const &Csrc){

    Mat src(Csrc.rows,Csrc.cols,CV_8UC1,Scalar(0));

    cvtColor(Csrc,src,CV_BGR2GRAY);
    int i,j,k,l,density;
    Mat dst(ceil(src.rows/kernel),ceil(src.cols/kernel),CV_8UC1,Scalar(0));

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
        }

    }
    //cout<<" "<<dst.rows<<" "<<dst.cols<<endl;
    //namedWindow("density map",WINDOW_NORMAL);
    //imshow("density map",dst);
    //imwrite("density.jpg",dst);
    //waitKey(0);
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
    //namedWindow("stddev_stage 3",WINDOW_NORMAL);
    //imshow("stddev_stage 3",dst);
    //imwrite("s3.jpg",dst);
    //waitKey(0);
    return dst;
}

Mat stage4(Mat const &img,int thresh=4,int remove=0)
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
    //namedWindow("Stage5 geo correct",WINDOW_NORMAL);
    //imshow("Stage5 geo correct",dst2);
    //imwrite("s5.jpg",dst);
    //waitKey(0);
    //cout<<"sTAGE 4 EXIT"<<dst2.rows<<" "<<dst2.cols<<endl;
    if(remove)
        return remove_blobs(dst2);
    else
        return dst2;
}

Mat stage5(Mat const &cs1,Mat const &s4)
{
    Mat s1=cs1.clone();
    cvtColor(s1,s1,CV_BGR2GRAY);
    Mat dst(s1.rows,s1.cols,CV_8UC1,Scalar(0));
    int i,j,k,l;
    for(i=0;i<s4.rows;i++)
    {
        for(j=0;j<s4.cols;j++)
        {
               if(isBoundary(s4,i,j))
                {
                    for(k=0;k<min(kernel,s1.rows-i*kernel);k++)
                    {
                        for(l=0;l<min(kernel,s1.cols-j*kernel);l++)
                        {
                            dst.at<uchar>(i*kernel+k,j*kernel+l)=s1.at<uchar>(i*kernel+k,j*kernel+l);
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
                        }
                    }
                }

        }
    }
    //namedWindow("contour stage 6",WINDOW_AUTOSIZE);
    //imshow("contour stage 6",dst);
    //imwrite("s6.jpg",dst);
    //waitKey(0);
    return dst;
}

Mat GetSkin(Mat const &src,int cr_min_arg=128,int cr_max_arg=164,int cb_min_arg=115,int cb_max_arg=160)
{
    ::cr_min=cr_min_arg;
    ::cr_max=cr_max_arg;
    ::cb_min=cb_min_arg;
    ::cb_max=cb_max_arg;
    Mat dst=src.clone();
    Vec3b cblack = Vec3b::all(0);
    Mat s1=stage1(src);
    s1= stage5(s1,stage4(stage2(s1)));
    int i,j;
    for(i=0;i<s1.rows;i++)
    {
        for(j=0;j<s1.cols;j++)
        {
            if(s1.at<uchar>(i,j)==0)
            {
                dst.ptr<Vec3b>(i)[j] = cblack;
            }
        }
    }
    return dst;
}

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
       //sprintf(name,"images/%i.png",i);
       // cv::imwrite(name, img);
        i++;
        break;
       }
}

