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

#define n 120
#define m 120

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

MatrixXf pca2d::copy_cv2eigen(Mat src,int type=0)
{
    MatrixXf dst(src.rows,src.cols);
    if(type==0)
    {
        for(int i=0;i<src.rows;i++)
        {
            for(int j=0;j<src.cols;j++)     dst(i,j)=src.at<uchar>(i,j);
        }

    }
    else if(type==5)
    {
        //cout<<"Type 5 called"<<endl;
        for(int i=0;i<src.rows;i++)
        {
            for(int j=0;j<src.cols;j++)     dst(i,j)=src.at<float>(i,j);
        }

    }
	return dst;
}

Mat pca2d::copy_eigen2cv(MatrixXf src,int type=0)
{
    Mat dst(src.rows(),src.cols(),type,Scalar(0));
    if(type==0)
    {
        for(int i=0;i<src.rows();i++)
        {
            for(int j=0;j<src.cols();j++)    dst.at<uchar>(i,j)=(int)src(i,j);
        }

    }
    else if(type==5)
    {
        //cout<<"Type 5 called"<<endl;
        for(int i=0;i<src.rows();i++)
        {
            for(int j=0;j<src.cols();j++)    dst.at<float>(i,j)=src(i,j);
        }
    }
	return dst;
}

void pca2d::train(vector<Mat> images,vector<int> labels,int num_evecs,string filename)
{
	int num_images=images.size();
	Mat input;
	MatrixXf mean(m,n);
	MatrixXf G;
    MatrixXf A;
	mean=MatrixXf::Zero(m,n);


	//--------------Taking input images and calculating their mean------------------//
	for(int i=images.size()-1;i>=0;i--)
	{
		input=images[i];
        //cout<<input.rows<<"*"<<input.cols<<endl;
		if(input.channels()==3)
			cvtColor(input,input,CV_BGR2GRAY);
		if(input.rows!=m||input.cols!=n){
            resize(input,input, Size(n,m) , 0.0, 0.0, INTER_CUBIC);
            images[i]=input;
		}
		A=copy_cv2eigen(input);
		mean=mean+A;
	}

	mean=mean/num_images;
	// input=copy_eigen2cv(mean);
 //    cvNamedWindow("mean",WINDOW_NORMAL);
	// imshow("mean",input);
	// waitKey(0);

	//-------------Calculating covariance matrix in G-------------------//
	G=MatrixXf::Zero(n,n);
	for(int i=images.size()-1;i>=0;i--)
	{
		input=images[i];
		if(input.channels()==3)
			cvtColor(input,input,CV_BGR2GRAY);
		if(input.rows!=m||input.cols!=n){
            resize(input,input, Size(n,m) , 0.0, 0.0, INTER_CUBIC);
        }
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
    //cout<<endl<<"sum"<<total_sum;
    MatrixXf X,temp;
    X=initial_evec.block(0,0,initial_evec.rows(),num_evecs);
    //cout<<"Building X...\nAdding 1st eigenvector. Eigenvalue is "<< evals(0)<<" percentage is"<<evals(0)/total_sum<<endl;
    // X=initial_evec.col(0);
    // sum+=evals(0);
    // while((sum/total_sum)<e_val_thresh)
    // {

    // 	temp.resize(X.rows(),X.cols()+1);
    //     temp<<X,initial_evec.col(num_evecs);
    //     X=temp;
    //     sum+=evals(num_evecs);
    //     num_evecs++;
    //     //cout<<"Adding "<<num_evecs<<"th eigenvector. Eigenvalue is "<<evals(num_evecs)<<" percentage is "<<sum/total_sum<<endl;
    // }

    //cout<<endl<<"final x size is "<<endl<<X.rows()<<"*"<<X.cols()<<endl;
    //cout<<" number of eigenvectors in 2dpca is "<<num_evecs<<endl;
    this->eigenvectors_X=copy_eigen2cv(X,5);
    // ********** X has been calculated*************//

    //***********Visualisation of Reconstruction***********//
    //Mat src=images[1];
    //if(src.rows!=m||src.cols!=n)     resize(src,src, Size(n,m) , 0.0, 0.0, INTER_CUBIC);
    // cvNamedWindow("Source",WINDOW_NORMAL);
    // imshow("Source",src);
    // cvNamedWindow("Reconstructed",WINDOW_NORMAL);
    // waitKey(0);
    // MatrixXf U,V,A_reconstruct;
    // A=copy_cv2eigen(src);
    // A=A-mean;
    // for(int d=1;d<=num_evecs;d++)
    // {
    //     U=X.block(0,0,X.rows(),d);
    //     V=A*U;
    //     A_reconstruct=V*(U.transpose())+mean;
    //     // cout<<endl<<" size of X is "<<X.rows()<<"*"<<X.cols()<<endl;
    //     // cout<<endl<<" size of U is "<<U.rows()<<"*"<<U.cols()<<endl;
    //     // cout<<endl<<" size of V is "<<V.rows()<<"*"<<V.cols()<<endl;
    //     src=copy_eigen2cv(A_reconstruct);
    //     cvNamedWindow("Reconstructed",WINDOW_NORMAL);
    //     imshow("Reconstructed",src);
    //     waitKey(10);

    // }

    //*********** Calculating feature matrix *******//
    this->features.clear();
    this->classes=labels;
    for(int i=0;i<images.size();i++)
	{
		input=images[i];
		if(input.channels()==3)
			cvtColor(input,input,CV_BGR2GRAY);
		if(input.rows!=m||input.cols!=n){
            resize(input,input, Size(n,m) , 0.0, 0.0, INTER_CUBIC);
        }
		MatrixXf A,B;
		A=copy_cv2eigen(input);
		A=A-mean;
		B=A*X;
        input=copy_eigen2cv(B,5);
        this->features.push_back(input); //Mat type data is pushed to
		//input=copy_eigen2cv(A);
		//imshow("out",input);
		//waitKey(0);
	}
    //***********Feature matrices pushed to a vector***********//

    this->mean_img=copy_eigen2cv(mean,5);
    //*************Writing model to Xml File******************//

    FileStorage fs(filename, FileStorage::WRITE);
    fs<<"Features"<< this->features;
    fs<<"Labels" << labels;
    fs<<"Mean" <<this->mean_img;
    fs<<"X" << this->eigenvectors_X;

    fs.release();
    //cout << "\nWrite Done." << endl;
}

void pca2d::load(string filename)
{

    FileStorage fs(filename, FileStorage::READ);
    fs["Features"] >>this->features;
    fs["Labels"] >>this->classes;
    fs["Mean"] >>this->mean_img;
    fs["X"] >>this->eigenvectors_X;
    fs.release();
    fs.release();
    //cout << "\n Model loaded" << endl;
}


int pca2d::predict(Mat test,double distance_thresh)
{
    // cout<<"\nPrediction started"<<endl<<" Number of training samples = "<<classes.size()<<endl;
    // cout<<" Calculating Euclidean distances..."<<endl;

    if(test.channels()==3)  cvtColor(test,test,CV_BGR2GRAY);
    if(!((test.rows==m)&&(test.cols==n)))  resize(test,test, Size(n,m) , 0.0, 0.0, INTER_CUBIC);
    vector<distances>eucl_dist_vec;
    distances temp={0,0,0};
    MatrixXf mean=copy_cv2eigen(this->mean_img,5);
    MatrixXf A=copy_cv2eigen(test);
    A-=mean;
    MatrixXf X=copy_cv2eigen(this->eigenvectors_X,5);
    MatrixXf B;
    //cout<<"A "<<A.rows()<<"*"<<A.cols()<<endl;
    //cout<<"X "<<X.rows()<<"*"<<X.cols()<<endl;
    B=A*X;
    MatrixXf A_reconstruct=B*X.transpose()+mean;
    // cvNamedWindow("Reconstructed",WINDOW_NORMAL);
    // imshow("Reconstructed",copy_eigen2cv(A_reconstruct));
    // waitKey(30);
    //cout<<"\n Sum of all elements in B is "<<B.sum()<<endl;
    MatrixXf Bclass;
    for(int i=0;i<classes.size();i++)
    {       MatrixXf Btrain=copy_cv2eigen(features[i],5);
            Btrain-=B;
            temp.dist=Btrain.squaredNorm();
            temp.label=classes[i];
            eucl_dist_vec.push_back(temp);
    }
    
    int min=0;
    for(int i=1;i<eucl_dist_vec.size();i++)
    {
        if (eucl_dist_vec[i].dist<eucl_dist_vec[min].dist)
            {
                min=i;
            }
    }
    //cout<<" distance is "<<eucl_dist_vec[min].dist<<endl;
    if(eucl_dist_vec[min].dist<distance_thresh)
        return eucl_dist_vec[min].label;
    else return -1;

    
}
