#ifndef IMAGESVM_H
#define IMAGESVM_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;
namespace svm
{
	void TrainAndPredict(Mat &train_data, Mat &test_data);
}
#endif // !IMAGESVM_H
