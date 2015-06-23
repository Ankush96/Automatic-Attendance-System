#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "Utils.h"
#include "two_d_pca.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>


using namespace cv;
using namespace Eigen;

#define m 120
#define n 120

void swap(VectorXf &arr, const int a, const int b,MatrixXf &evec)
{
    float tmp_val=arr(a);
    arr(a)=arr(b);
    arr(b)=tmp_val;
    VectorXf tmp_vec;
    tmp_vec=evec.col(a);
    evec.col(a)=evec.col(b);
    evec.col(b)=tmp_vec;
}

int partition(VectorXf &arr, const int left, const int right,MatrixXf &evec)
{
    const int mid = left + (right - left) / 2;
    const float pivot = arr(mid);
    // move the mid point value to the front.
    swap(arr,mid,left,evec);
    int i = left + 1;
    int j = right;
    while (i <= j) {
    while(i <= j && arr(i) >= pivot) {
    i++;
    }

    while(i <= j && arr(j) < pivot) {
    j--;
    }
    //cout<<i<<" "<<j<<endl;

    if (i < j) {
    swap(arr,i,j,evec);
    }
    }
    swap(arr,i - 1,left,evec);
    return i - 1;
}


void quicksort(VectorXf &arr, const int left, const int right, const int sz,MatrixXf &evec)
{

    if (left >= right) {
    return;
    }


    int part = partition(arr, left, right,evec);


    quicksort(arr, left, part - 1, sz,evec);
    quicksort(arr, part + 1, right, sz,evec);
}

MatrixXf pca2d::copy_cv2eigen(Mat src)
{
    MatrixXf dst(m,n);
	for(int i=0;i<src.rows;i++)
	{
		for(int j=0;j<src.cols;j++)
		{
			//cout<<"\nSource value is "<<(int)src.at<uchar>(i,j);
			dst(i,j)=(float)src.at<uchar>(i,j);
			//cout<<" and matrix value becomes "<<dst(i,j)<<endl;
		}
	}
	return dst;
}

Mat pca2d::copy_eigen2cv(MatrixXf src)
{
    Mat dst(m,n,CV_8UC1,Scalar(0));

	for(int i=0;i<m;i++)
	{
		for(int j=0;j<n;j++)
		{
			//cout<<"\nSource value is "<<(int)src.at<uchar>(i,j);
			dst.at<uchar>(i,j)=(int)src(i,j);
			//cout<<" and matrix value becomes "<<dst(i,j)<<endl;
		}
	}
	return dst;
}


void pca2d::train(vector<Mat> images,vector<int> labels,double e_val_thresh=0.8)
{
	int num_images=images.size();
	Mat input;
	MatrixXf mean(m,n);
	MatrixXf G(m,n);
	mean=MatrixXf::Zero(m,n);
	cvNamedWindow("out",WINDOW_NORMAL);

	//--------------Taking input images and calculating their mean------------------//
	for(int i=images.size()-1;i>=0;i--)
	{
		input=images[i];
		if(input.channels()==3)
			cvtColor(input,input,CV_BGR2GRAY);
		resize(input,input, Size(m,n) , 1.0, 1.0, INTER_CUBIC);
		MatrixXf A(m,n);
		A=copy_cv2eigen(input);
		mean=mean+A;
	}
	mean=mean/num_images;
	input=copy_eigen2cv(mean);
	imshow("out",input);
	waitKey(0);

	//-------------Calculating covariance matrix in G-------------------//
	G=MatrixXf::Zero(m,n);
	for(int i=images.size()-1;i>=0;i--)
	{
		input=images[i];
		if(input.channels()==3)
			cvtColor(input,input,CV_BGR2GRAY);
		resize(input,input, Size(m,n), 1.0, 1.0, INTER_CUBIC);
		MatrixXf A(m,n);
		A=copy_cv2eigen(input);
		A=A-mean;
		G=G+A.transpose()*A;
		//input=copy_eigen2cv(A);
		//imshow("out",input);
		//waitKey(0);
	}
	G=G/num_images;

	//**************Finding out the eigenvectors***************//
	EigenSolver<MatrixXf> es(G);
	//cout << "The eigenvalues of A are:\n" <<es.eigenvalues() << endl;
	VectorXcf complex_eval=es.eigenvalues();
    MatrixXf initial_evec=es.eigenvectors().real();
    //cout<<"initial evec"<<endl<<initial_evec<<endl;


    for(int i=0;i<complex_eval.size();i++)
    {
        if(std::fabs(complex_eval(i).imag())>0.001) complex_eval(i).real()=0;
    }
    VectorXf evals=complex_eval.real();
    quicksort(evals,0,evals.size()-1,evals.size(),initial_evec);
    double total_sum=evals.sum(),sum=0;
    cout<<endl<<"sum"<<total_sum;
    int num_evecs=0;
    MatrixXf X,temp;
   /* for(int i=0;i<evals.size();i++)
    {
        if(complex_eval(i).real()/sum>=e_val_thresh)
        {
            if(num_evecs==0)
            {
                X=initial_evec.col(i);
            }
            else
            {
                temp.resize(X.rows(),X.cols()+1);
                temp<<X,initial_evec.col(i);
                X=temp;
            }
            num_evecs++;
        }
    }*/
    cout<<"Building X.../n Adding 1st eigen vector. eigenvalue is "<< evals(0)<<endl<<"percentage is"<<evals(0)/total_sum;    
    X=initial_evec.col(0);
    sum+=evals(0);
    while((sum/total_sum)<e_val_thresh)
    {
    	num_evecs++;
    	temp.resize(X.rows(),X.cols()+1);
        temp<<X,initial_evec.col(num_evecs);
        X=temp;
        sum+=evals(num_evecs);
        cout<<"Adding "<<num_evecs<<"th eigenvector"<<endl<<"eigenvalue is "<<evals(num_evecs)<<endl<<"percentage is "<<sum/total_sum<<endl;
    }    

    cout<<endl<<"final x size is "<<endl<<X.rows()<<"*"<<X.cols()<<endl;
    // ********** X has been calculated*************//
    //*********** Calculating feature matrix *******//
    vector<MatrixXf> features;
    for(int i=images.size()-1;i>=0;i--)
	{
		input=images[i];
		if(input.channels()==3)
			cvtColor(input,input,CV_BGR2GRAY);
		resize(input,input, Size(m,n), 1.0, 1.0, INTER_CUBIC);
		MatrixXf A(m,n),B;
		A=copy_cv2eigen(input);
		A=A-mean;
		B=A*X;
		//input=copy_eigen2cv(A);
		//imshow("out",input);
		//waitKey(0);
	}

}
