
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

void swap(Eigen::VectorXf &arr, const int a, const int b,Eigen::MatrixXf &evec);
int partition(Eigen::VectorXf &arr, const int left, const int right,Eigen::MatrixXf &evec);
void quicksort(Eigen::VectorXf &arr, const int left, const int right, const int sz,Eigen::MatrixXf &evec);

class pca2d
{
	private:


	public:
		void train(vector<Mat>,vector<int>,double);
		Mat copy_eigen2cv(Eigen::MatrixXf src);
		Eigen::MatrixXf copy_cv2eigen(Mat);

};
