///*
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "Stage-segment.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            Mat tmp1=imread(path, 0);
            Mat tmp2;
            resize(tmp1, tmp2, Size(120, 120), 1.0, 1.0, INTER_CUBIC);
            ///tmp2=clahe(tmp2);
            equalizeHist(tmp2,tmp2);
            //cvNamedWindow("image",WINDOW_NORMAL);
            //imshow("image",tmp2);
            //waitKey(0);
            //cvtColor(tmp2,tmp2,CV_BGR2GRAY);
            images.push_back(tmp2);
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int main3()
{

    string fn_csv = "samples.csv";

    vector<Mat> images;
    vector<int> labels;

    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }


    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }




     Ptr<FaceRecognizer> lbp = createLBPHFaceRecognizer();
     Ptr<FaceRecognizer> ef =createEigenFaceRecognizer();
     Ptr<FaceRecognizer> ff =createFisherFaceRecognizer();
   //  cvNamedWindow("Face",WINDOW_NORMAL);
    // imshow("Face",images[1]);
    // cvWaitKey(0);

    lbp->train(images, labels);

    lbp->save("lbp.xml");
    ef->train(images, labels);

    ef->save("ef.xml");
    ff->train(images, labels);

    ff->save("ff.xml");
    // The following line predicts the label of a given
    // test image:


    //
    // To get the confidence of a prediction call the model with:
    //
    //      int predictedLabel = -1;
    //      double confidence = 0.0;
    //      model->predict(testSample, predictedLabel, confidence);
    //
  //  string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
  //  cout << result_message << endl;
    // Sometimes you'll need to get/set internal model data,
    // which isn't exposed by the public cv::FaceRecognizer.
    // Since each cv::FaceRecognizer is derived from a
    // cv::Algorithm, you can query the data.
    //
    // First we'll use it to set the threshold of the FaceRecognizer
    // to 0.0 without retraining the model. This can be useful if
    // you are evaluating the model:
    //
    //model->set("threshold", 0.0);
    // Now the threshold of this model is set to 0.0. A prediction
    // now returns -1, as it's impossible to have a distance below
    // it
    //predictedLabel = model->predict(testSample);
    //cout << "Predicted class = " << predictedLabel << endl;
    // Show some informations about the model, as there's no cool
    // Model data to display as in Eigenfaces/Fisherfaces.
    // Due to efficiency reasons the LBP images are not stored
    // within the model:
    //cout << "Model Information:" << endl;
    //string model_info = format("\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
     //       model->getInt("radius"),
    //        model->getInt("neighbors"),
     //       model->getInt("grid_x"),
    //        model->getInt("grid_y"),
    //        model->getDouble("threshold"));
    //cout << model_info << endl;
    //// We could get the histograms for example:
   // vector<Mat> histograms = model->getMatVector("histograms");
    // But should I really visualize it? Probably the length is interesting:
    //cout << "Size of the histograms: " << histograms[0].total() << endl;

    return 0;
}
//*/
