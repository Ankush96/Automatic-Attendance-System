#include <vector>
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
		vector<Mat> features;
		vector<int> classes;
		Mat mean_img;
		Mat eigenvectors_X;

	public:
		void train(vector<Mat>,vector<int>,double,string);
		void load(string);
		int predict(Mat);

		Mat copy_eigen2cv(Eigen::MatrixXf src,int);
		Eigen::MatrixXf copy_cv2eigen(Mat,int);

};
