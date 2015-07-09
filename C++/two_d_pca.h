#ifndef __2DPCA_H_INCLUDED__
#define __2DPCA_H_INCLUDED__


#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

void swap(Eigen::VectorXf &arr, const int a, const int b,Eigen::MatrixXf &evec);

int partition(Eigen::VectorXf &arr, const int left, const int right,Eigen::MatrixXf &evec);

void quicksort(Eigen::VectorXf &arr, const int left, const int right, const int sz,Eigen::MatrixXf &evec);

struct distances{
	double dist;
	int label;
	int class_count;
};

class pca2d
{
	private:
		std::vector<cv::Mat> features;
		std::vector<int> classes;
		cv::Mat mean_img;
		cv::Mat eigenvectors_X;

	public:
		void train(std::vector<cv::Mat>,std::vector<int>,double,std::string);
		void load(std::string);
		int predict(cv::Mat,double distance_thresh=200000000); //keep one zero less

		cv::Mat copy_eigen2cv(Eigen::MatrixXf src,int);
		Eigen::MatrixXf copy_cv2eigen(cv::Mat,int);

};

#endif