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



void pca2d::train(vector<Mat> images,vector<int> labels)
{
	int num_images=images.size();
	Mat input;
	MatrixXd A;
	for(int i=images.size()-1;i>=0;i--)
	{
		input=images[i];
		//cv2eigen(input,A);
		//eigen2cv(A,input);
		A=Map<MatrixXd>(reinterpret_cast<double*>(input.data),input.rows,input.cols);
		cout<<A<<endl;
		//imshow("out,",input);
		waitKey(30);

	}
	cout<<num_images<<endl;
}
