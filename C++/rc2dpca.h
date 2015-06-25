#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

class rc2dpca
{
	private:
		std::vector<cv::Mat> features;
		std::vector<int> classes;
		cv::Mat mean_img;
		cv::Mat mean_feature_y;
		cv::Mat eigenvectors_X;
		cv::Mat eigenvectors_U;

	public:
		void train(std::vector<cv::Mat>,std::vector<int>,double,std::string);
		void load(std::string);
		int predict(cv::Mat,double distance_thresh=5000000);

		cv::Mat copy_eigen2cv(Eigen::MatrixXf src,int);
		Eigen::MatrixXf copy_cv2eigen(cv::Mat,int);

};
