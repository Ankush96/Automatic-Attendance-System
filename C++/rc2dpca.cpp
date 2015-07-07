#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "Utils.h"
#include "rc2dpca.h"
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



MatrixXf rc2dpca::copy_cv2eigen(Mat src,int type=0)
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

Mat rc2dpca::copy_eigen2cv(MatrixXf src,int type=0)
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

void rc2dpca::train(vector<Mat> images,vector<int> labels,double e_val_thresh,string filename)
{
	int num_images=images.size();
	Mat input;
	MatrixXf mean(m,n);
	MatrixXf G;
    MatrixXf G2;
    MatrixXf A;
	mean=MatrixXf::Zero(m,n);
	//--------------Taking input images and calculating their mean------------------//
	for(int i=images.size()-1;i>=0;i--)
	{
		input=images[i];
        //cout<<" image entered. "<< input.rows<<"*"<<input.cols<< " "<< m << " * " <<n<<endl;
		if(input.channels()==3)
			cvtColor(input,input,CV_BGR2GRAY);
		if(input.rows!=m||input.cols!=n)
        {
            resize(input,input, Size(n,m) , 1.0, 1.0, INTER_CUBIC);
            images[i]=input;
        }
            
		A=copy_cv2eigen(input);
		mean=mean+A;
	}

	mean=mean/num_images;
	// input=copy_eigen2cv(mean);
    // cvNamedWindow("mean",WINDOW_NORMAL);
	// imshow("mean",input);
	// waitKey(0);
    this->mean_img=copy_eigen2cv(mean,5);

	//-------------Calculating covariance matrix in G-------------------//
	G=MatrixXf::Zero(n,n);
	for(int i=images.size()-1;i>=0;i--)
	{
		input=images[i];
		if(input.channels()==3)
			cvtColor(input,input,CV_BGR2GRAY);
		if(input.rows!=m||input.cols!=n)
            resize(input,input, Size(n,m) , 1.0, 1.0, INTER_CUBIC);

		A=copy_cv2eigen(input);
		A=A-mean;
		G=G+A.transpose()*A;
	}
	G=G/num_images;
	//**************Finding out the eigenvectors of G***************//
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
    int num_evecs=1;
    MatrixXf X,temp;

    //cout<<"Building X...\nAdding 1st eigenvector. Eigenvalue is "<< evals(0)<<" percentage is"<<evals(0)/total_sum<<endl;
    X=initial_evec.col(0);
    sum+=evals(0);
    while((sum/total_sum)<e_val_thresh)
    {

    	temp.resize(X.rows(),X.cols()+1);
        temp<<X,initial_evec.col(num_evecs);
        X=temp;
        sum+=evals(num_evecs);
        num_evecs++;
        //cout<<"Adding "<<num_evecs<<"th eigenvector. Eigenvalue is "<<evals(num_evecs)<<" percentage is "<<sum/total_sum<<endl;
    }

    //cout<<endl<<"final x size is "<<endl<<X.rows()<<"*"<<X.cols()<<endl;
    //cout<<" number of eigenvectors in X is "<<num_evecs<<endl;
    this->eigenvectors_X=copy_eigen2cv(X,5);
    // ********** X has been calculated*************//

    //

    //***********Visualisation of Reconstruction***********//
    // Mat src=images[1];
    // cvNamedWindow("Source",WINDOW_NORMAL);
    // imshow("Source",src);
    // cvNamedWindow("Reconstructed",WINDOW_NORMAL);
    // waitKey(0);
    // MatrixXf U,V,A_reconstruct;

    // A=copy_cv2eigen(src);
    // A=A-mean;
    // char text[10];
    // for(int d=1;d<=num_evecs;d++)
    // {
    //     U=X.block(0,0,X.rows(),d);
    //     V=A*U;
    //     A_reconstruct=V*(U.transpose())+mean;
    //     src=copy_eigen2cv(A_reconstruct);

    //     //sprintf(text," d=%d ",d);
    //     //cvNamedWindow(text,WINDOW_NORMAL);
    //     //imshow(text,src);
    //     imshow("Reconstructed",src);
    //     waitKey(10);

    // }
    // waitKey(0);

    //*********** Calculating feature matrix Y *******//
    this->features.clear();
    this->classes=labels;
    vector<MatrixXf> Y_vector;
    for(int i=0;i<images.size();i++)
	{
		input=images[i];
		if(input.channels()==3)
			cvtColor(input,input,CV_BGR2GRAY);
		if(input.rows!=m||input.cols!=n){
            resize(input,input, Size(n,m) , 1.0, 1.0, INTER_CUBIC);
        }
		MatrixXf A,B;
		A=copy_cv2eigen(input);
		A=A-mean;
		B=A*X;
        // ---------- B serves the purpose of Y . Pushing the Y into a vector -------------//
        Y_vector.push_back(B);
        //input=copy_eigen2cv(B,5);
        //this->features.push_back(input); //Mat type data is pushed to
	}
    //***********Feature matrices pushed to a vector***********//

    //--------------Calculating mean of Y------------------------//
    MatrixXf mean_y;
    mean_y=MatrixXf::Zero(m,X.cols());
    for(int i=Y_vector.size()-1;i>=0;i--)
    {
        mean_y=mean_y+Y_vector[i];
    }

    mean_y=mean_y/num_images;
    this->mean_feature_y=copy_eigen2cv(mean_y,5);

    //------------------------------------------------------------//

    //-------------Calculating covariance matrix G2-----------------//
    G2=MatrixXf::Zero(mean_y.rows(),mean_y.rows());
    for(int i=Y_vector.size()-1;i>=0;i--)
    {
        MatrixXf Y=Y_vector[i];
        //cout<<" Y IS "<<Y.rows()<<"*"<<Y.cols()<<endl;
        Y=Y-mean_y;
        G2=G2+Y*Y.transpose();
    }
    G2=G2/num_images;
    //------------------------------------------------------------//

    //------------Calculating eigenvectors to stack in U-------------//
    es.compute(G2);
    complex_eval=es.eigenvalues();
    initial_evec=es.eigenvectors().real();

    for(int i=0;i<complex_eval.size();i++)
    {
        if(std::fabs(complex_eval(i).imag())>0.001) complex_eval(i).real()=0;
    }
    evals=complex_eval.real();
    quicksort(evals,0,evals.size()-1,evals.size(),initial_evec);
    total_sum=evals.sum();
    sum=0;
    // cout<<endl<<"sum"<<total_sum;
    num_evecs=1;
    MatrixXf U;

    //cout<<"Building U...\nAdding 1st eigenvector. Eigenvalue is "<< evals(0)<<" percentage is"<<evals(0)/total_sum<<endl;
    U=initial_evec.col(0);
    sum+=evals(0);
    while((sum/total_sum)<e_val_thresh)
    {

        temp.resize(U.rows(),U.cols()+1);
        temp<<U,initial_evec.col(num_evecs);
        U=temp;
        sum+=evals(num_evecs);
        num_evecs++;
        //cout<<"Adding "<<num_evecs<<"th eigenvector. Eigenvalue is "<<evals(num_evecs)<<" percentage is "<<sum/total_sum<<endl;
    }

    //cout<<endl<<"final U size is "<<endl<<U.rows()<<"*"<<U.cols()<<endl;
    //cout<<" number of eigenvectors in U is "<<num_evecs<<endl;
    this->eigenvectors_U=copy_eigen2cv(U,5);
    //------------------------------------------------------------------//

    //------------------Calculating final features----------------------//
    this->features.clear();
    this->classes=labels;
    for(int i=0;i<images.size();i++)
    {
        input=images[i];
        if(input.channels()==3)
            cvtColor(input,input,CV_BGR2GRAY);
        if(input.rows!=m||input.cols!=n){
            resize(input,input, Size(n,m) , 1.0, 1.0, INTER_CUBIC);
        }
        MatrixXf A,C;
        A=copy_cv2eigen(input);
        A=A-mean;
        C=U.transpose()*A*X;
        input=copy_eigen2cv(C,5);
        this->features.push_back(input); //Mat type data is pushed to
    }

    //----------------------------------------------------------------

    //-------------Image reconstruction-----------------------------//

    // Mat src=images[1];
    // cvNamedWindow("Source",WINDOW_NORMAL);
    // imshow("Source",src);
    // cvNamedWindow("Reconstructed",WINDOW_NORMAL);
    // waitKey(0);
    // MatrixXf C,A_reconstruct;

    // A=copy_cv2eigen(src);
    // A=A-mean;
    // for(int d=1;d<=X.cols();d++)
    // {

    //     C=U.transpose()*A*X;
    //     A_reconstruct=(U*C+mean_y)*X.transpose()+mean;
    //     src=copy_eigen2cv(A_reconstruct);

    //     //sprintf(text," d=%d ",d);
    //     //cvNamedWindow(text,WINDOW_NORMAL);
    //     //imshow(text,src);
    //     imshow("Reconstructed",src);
    //     waitKey(10);

    // }
    // waitKey(0);
    //--------------------------------------------------------------//

    //*************Writing model to Xml File******************//

    FileStorage fs(filename, FileStorage::WRITE);
    fs<<"Features"<< this->features;
    fs<<"Labels" << labels;
    fs<<"MeanA" <<this->mean_img;
    fs<<"MeanY" <<this->mean_feature_y;
    fs<<"U" << this->eigenvectors_U;
    fs<<"X" << this->eigenvectors_X;

    fs.release();
    //cout << "\nWrite Done." << endl;
}

void rc2dpca::load(string filename)
{

    FileStorage fs(filename, FileStorage::READ);
    fs["Features"] >>this->features;
    fs["Labels"] >>this->classes;
    fs["MeanA"] >>this->mean_img;
    fs["MeanY"] >>this->mean_feature_y;
    fs["X"] >>this->eigenvectors_X;
    fs["U"] >>this->eigenvectors_U;
    fs.release();
    fs.release();
    //cout << "\n Model loaded" << endl;
}


int rc2dpca::predict(Mat test,double distance_thresh)
{
    // cout<<"\nPrediction started"<<endl<<" Number of training samples = "<<classes.size()<<endl;
    // cout<<" Calculating Euclidean distances..."<<endl;
    
    if(test.channels()==3)  cvtColor(test,test,CV_BGR2GRAY);
    if(!((test.rows==m)&&(test.cols==n)))  resize(test,test, Size(n,m) , 1.0, 1.0, INTER_CUBIC);
    vector<distances> eucl_dist_vec;
    distances temp={0,0,0};
    MatrixXf MeanA=copy_cv2eigen(this->mean_img,5);
    MatrixXf MeanY=copy_cv2eigen(this->mean_feature_y,5);
    MatrixXf A=copy_cv2eigen(test);
    A-=MeanA;
    MatrixXf X=copy_cv2eigen(this->eigenvectors_X,5);
    MatrixXf U=copy_cv2eigen(this->eigenvectors_U,5);
    MatrixXf C;
    //cout<<"A "<<A.rows()<<"*"<<A.cols()<<endl;
    //cout<<"X "<<X.rows()<<"*"<<X.cols()<<endl;
    C=U.transpose()*A*X;
    //MatrixXf A_reconstruct=B*X.transpose()+mean;
    //cvNamedWindow("Reconstructed",WINDOW_NORMAL);
    //imshow("Reconstructed",copy_eigen2cv(A_reconstruct));
    //waitKey(100);

    for(int i=0;i<classes.size();i++)
    {
        MatrixXf Ctrain=copy_cv2eigen(features[i],5);
        Ctrain-=C;
        temp.dist=Ctrain.squaredNorm();
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

