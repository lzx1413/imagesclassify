#ifndef  IMAGEFEATURES_H
#define  IMAGEFEATURES_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;
namespace feature
{
	const double EPS = 2.2204e-16;
	vector<Point2f> DectectHarrisLaplace(const Mat &src);
	Mat getHOGKernel(Size& ksize, double sigma);
	vector<Point2f> GetKeyPoint(const Mat& src);
	vector<Point2f> MixKeyPoint(vector<Point2f> keypoint1, vector<Point2f> keypoint2);
	vector<int> GetImageFeatures(vector<Mat> &images, Mat& featuresUnclustered);
	Mat GetVocabulary(const Mat &features, int center_num);
	Mat GetVocabularyFrequency(Mat & features, int center_num, vector<int> image_label_list);
	void NormalizeCols(Mat &features);
	Mat GaussianKernel2D(int dim, float sigma);
	vector<float> CalculateHueSift(Mat& himage, Point keypoint);
	float CalSimilarity(Mat&vocabulary, Mat &single_feature);

}
#endif