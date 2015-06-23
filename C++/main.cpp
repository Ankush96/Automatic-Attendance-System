
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

using namespace cv;
using namespace Eigen;



int main()
{
    vector<Mat> images;
    vector<int> labels;
    dir_read("../orl_faces",12,images,labels);
    pca2d model;
    model.train(images,labels,0.8);

   //  MatrixXf ones = MatrixXf::Random(7,7);
   //  EigenSolver<MatrixXf> es(ones);
   //  VectorXcf complex_eval=es.eigenvalues();
   //  MatrixXf initial_evec=es.eigenvectors().real();
   //  //cout<<"initial evec"<<endl<<initial_evec<<endl;


   //  for(int i=0;i<complex_eval.size();i++) {   if(std::fabs(complex_eval(i).imag())>0) complex_eval(i).real()=0;}

   //  VectorXf evals=complex_eval.real();
   //  cout<< endl << endl<<evals << endl<<endl<<initial_evec<<endl;
   //  //quicksort(evals,0,evals.size()-1,evals.size(),initial_evec);

   // // cout << "The eigenvalues of the 3x3 matrix of ones are:"
   //  cout<< endl<<endl<<evals <<endl<< endl<<initial_evec<<endl;

//const EigenvalueType *res=es.eigenvalues();
//cout<<res<<endl;
//cout << "The first eigenvector of the 3x3 matrix of ones is:"<<endl;
//cout<< endl<<es.eigenvectors().col(0) << endl;
//Vector<complex> res=es.eigenvectors().col(0);
//cout<<endl<<res<<endl;
    return 0;
}


