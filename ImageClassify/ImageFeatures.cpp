#include"ImageFeatures.h"
//#include "Log/LogManager.h"
//#include "Log/Log.h"
#include<opencv2/nonfree/features2d.hpp>
static const int CLASS_NUM = 38;
static const int NUM_IMAGES_PERCLASS = 15;
#include <thread>
#include <QDebug>
//static CLog *mylog = LogManager::OpenLog("D:/log.txt");
vector<Point2f> feature::DectectHarrisLaplace(const Mat& imgSrc)
{
	Mat gray;
	vector<Point2f> hl_point;
	if (imgSrc.channels() == 3)
	{
		cvtColor(imgSrc, gray, CV_BGR2GRAY);
	}
	else
	{
		gray = imgSrc.clone();
	}
	gray.convertTo(gray, CV_64F);
	double dSigmaStart = 1.5;
	double dSigmaStep = 1.2;
	int iSigmaNb = 13;

	vector<double> dvecSigma(iSigmaNb);
	for (int i = 0; i < iSigmaNb; i++)
	{
		dvecSigma[i] = dSigmaStart + i*dSigmaStep;
	}
	vector<Mat> harrisArray(iSigmaNb);

	for (int i = 0; i < iSigmaNb; i++)
	{
		double iSigmaI = dvecSigma[i];
		double iSigmaD = 0.7 * iSigmaI;

		int iKernelSize = 6 * round(iSigmaD) + 1;
		Mat dx(1, iKernelSize, CV_64F);
		for (int k = 0; k < iKernelSize; k++)
		{
			int pCent = (iKernelSize - 1) / 2;
			int x = k - pCent;
			dx.at<double>(0, i) = x * exp(-x*x / (2 * iSigmaD*iSigmaD)) / (iSigmaD*iSigmaD*iSigmaD*sqrt(2 * CV_PI));
		}
		Mat dy = dx.t();
		Mat Ix, Iy;
		filter2D(gray, Ix, CV_64F, dx);
		filter2D(gray, Iy, CV_64F, dy);
		Mat Ix2, Iy2, Ixy;
		Ix2 = Ix.mul(Ix);
		Iy2 = Iy.mul(Iy);
		Ixy = Ix.mul(Iy);
		int gSize = 6 * round(iSigmaI) + 1;
		Mat gaussKernel = getGaussianKernel(gSize, iSigmaI);
		filter2D(Ix2, Ix2, CV_64F, gaussKernel);
		filter2D(Iy2, Iy2, CV_64F, gaussKernel);
		filter2D(Ixy, Ixy, CV_64F, gaussKernel);
		double alpha = 0.06;
		Mat detM = Ix2.mul(Iy2) - Ixy.mul(Ixy);
		Mat trace = Ix2 + Iy2;
		Mat cornerStrength = detM - alpha * trace.mul(trace);
		double maxStrength;
		minMaxLoc(cornerStrength, NULL, &maxStrength, NULL, NULL);
		Mat dilated;
		Mat localMax;
		dilate(cornerStrength, dilated, Mat());
		compare(cornerStrength, dilated, localMax, CMP_EQ);


		Mat cornerMap;
		double qualityLevel = 0.2;
		double thresh = qualityLevel * maxStrength;
		cornerMap = cornerStrength > thresh;
		bitwise_and(cornerMap, localMax, cornerMap);
		harrisArray[i] = cornerMap.clone();
	}

	/*计算尺度归一化Laplace算子*/
	vector<Mat> laplaceSnlo(iSigmaNb);
	for (int i = 0; i < iSigmaNb; i++)
	{
		double iSigmaL = dvecSigma[i];
		Size kSize = Size(6 * floor(iSigmaL) + 1, 6 * floor(iSigmaL) + 1);
		Mat hogKernel = getHOGKernel(kSize, iSigmaL);
		filter2D(gray, laplaceSnlo[i], CV_64F, hogKernel);
		laplaceSnlo[i] *= (iSigmaL * iSigmaL);
	}
	Mat corners(gray.size(), CV_8U, Scalar(0));
	for (int i = 0; i < iSigmaNb; i++)
	{
		for (int r = 0; r < gray.rows; r++)
		{
			for (int c = 0; c < gray.cols; c++)
			{
				if (i == 0)
				{
					if (harrisArray[i].at<uchar>(r, c) > 0 && laplaceSnlo[i].at<double>(r, c) > laplaceSnlo[i + 1].at<double>(r, c))
					{
						corners.at<uchar>(r, c) = 255;
						hl_point.push_back(Point(c, r));
					}
				}
				else if (i == iSigmaNb - 1)
				{
					if (harrisArray[i].at<uchar>(r, c) > 0 && laplaceSnlo[i].at<double>(r, c) > laplaceSnlo[i - 1].at<double>(r, c))
					{
						corners.at<uchar>(r, c) = 255;
						hl_point.push_back(Point(c, r));
					}
				}
				else
				{
					if (harrisArray[i].at<uchar>(r, c) > 0 &&
						laplaceSnlo[i].at<double>(r, c) > laplaceSnlo[i + 1].at<double>(r, c) &&
						laplaceSnlo[i].at<double>(r, c) > laplaceSnlo[i - 1].at<double>(r, c))
					{
						corners.at<uchar>(r, c) = 255;
						hl_point.push_back(Point(c, r));
					}
				}
			}
		}
	}
	return hl_point;

}
Mat feature::getHOGKernel(Size& ksize, double sigma)
{
	Mat kernel(ksize, CV_64F);
	Point centPoint = Point((ksize.width - 1) / 2, ((ksize.height - 1) / 2));
	for (int i = 0; i < kernel.rows; i++)
	{
		double* pData = kernel.ptr<double>(i);
		for (int j = 0; j < kernel.cols; j++)
		{
			double param = -((i - centPoint.y) * (i - centPoint.y) + (j - centPoint.x) * (j - centPoint.x)) / (2 * sigma*sigma);
			pData[j] = exp(param);
		}
	}
	double maxValue;
	minMaxLoc(kernel, NULL, &maxValue);
	for (int i = 0; i < kernel.rows; i++)
	{
		double* pData = kernel.ptr<double>(i);
		for (int j = 0; j < kernel.cols; j++)
		{
			if (pData[j] < EPS* maxValue)
			{
				pData[j] = 0;
			}
		}
	}

	double sumKernel = sum(kernel)[0];
	if (sumKernel != 0)
	{
		kernel = kernel / sumKernel;
	}
	for (int i = 0; i < kernel.rows; i++)
	{
		double* pData = kernel.ptr<double>(i);
		for (int j = 0; j < kernel.cols; j++)
		{
			double addition = ((i - centPoint.y) * (i - centPoint.y) + (j - centPoint.x) * (j - centPoint.x) - 2 * sigma*sigma) / (sigma*sigma*sigma*sigma);
			pData[j] *= addition;
		}
	}
	sumKernel = sum(kernel)[0];
	kernel -= (sumKernel / (ksize.width  * ksize.height));

	return kernel;
}

vector<Point2f> feature::GetKeyPoint(const Mat& src)
{
	vector<Point2f> keypoint;
	keypoint = DectectHarrisLaplace(src);
	SiftFeatureDetector detector(200);
	vector<KeyPoint> sift_key_point;
	detector.detect(src, sift_key_point);
	vector<Point2f> sift_key_point2f;
	KeyPoint::convert(sift_key_point, sift_key_point2f);
	return MixKeyPoint(keypoint, sift_key_point2f);
}

vector<Point2f> feature::MixKeyPoint(vector<Point2f>keypoint1, vector<Point2f>keypoint2)
{
	RNG rng;
	for (int i = 0; i < keypoint2.size(); i++)
	{
		int j = 0;
		for (j = 0; j < keypoint1.size(); j++)
		{
			if (keypoint2[i].x == keypoint1[j].x || keypoint1[j].y == keypoint2[i].y)
			{
				break;
			}
		}
		if (j == keypoint1.size())
		{
			if (rng.uniform(0.f, 1.f) > 0.1)
			{

				keypoint1.push_back(keypoint2[i]);
			}
		}

	}
	return keypoint1;
}

vector<int> feature::GetImageFeatures(vector<Mat> &images, Mat &featuresUnclustered)//TODO:将改进后的色彩特征向量融合进去
{
	vector<KeyPoint> keypoints;
	vector<int> image_labels(images.size());
	Mat descriptor;
	SiftDescriptorExtractor computor;
	for (int image_size = 0; image_size < images.size(); image_size++)
	{
		Mat gray_image;
		Mat hsv_image;
		cvtColor(images[image_size], gray_image, CV_RGB2GRAY);
		cvtColor(images[image_size], hsv_image, CV_RGB2HSV);
		vector<Point2f> keypoint2f = GetKeyPoint(gray_image);
		image_labels[image_size] = keypoint2f.size();
		KeyPoint::convert(keypoint2f, keypoints);
		cout << image_size << " " << keypoints.size() << endl;
		computor.compute(gray_image, keypoints, descriptor);
		Mat descriptorall = Mat::zeros(descriptor.rows, descriptor.cols + 16, descriptor.type());
		for (int i = 0; i < descriptor.rows; i++)
		{
			for (int j = 0; j < 129; j++)
			{
				if (j < 128)
				{
					descriptorall.at<float>(i, j) = descriptor.at<float>(i, j);
				}
				else
				{
					Point pt = keypoints[i].pt;
					if (pt.y + 10 > hsv_image.rows || pt.x + 10 > hsv_image.cols || pt.x < 9 || pt.y < 9)
					{
						cout << "error out of range" << endl;
						continue;
					}
					else
					{
						vector<float>huesift = CalculateHueSift(hsv_image, pt);
						for (int k = 0; k < 16; k++)
						{
							descriptorall.at<float>(i, 128 + k) = huesift.at(k);
						}
					}
					descriptorall.at<float>(i, 128) = hsv_image.at<uchar>(pt.y, pt.x);
				}
			}
		}
		featuresUnclustered.push_back(descriptorall);
	}
	return image_labels;
}

Mat feature::GetVocabulary(const Mat &features, int center_num)
{
	Mat labels, normalizedfeatures;
	normalize(features, normalizedfeatures);
	Mat vocabulary = Mat::zeros(center_num, features.cols, features.type());
	kmeans(normalizedfeatures, center_num, labels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, vocabulary);
	cout << labels << endl;
	return vocabulary;
}

Mat feature::GetVocabularyFrequency(Mat & features, int center_num, vector<int> image_label_list)
{
	Mat labels;
	Mat normalized_frequency;
	NormalizeCols(features);
	Mat vocabulary = Mat::zeros(center_num, features.cols, features.type());
	kmeans(features, center_num, labels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, vocabulary);
	cout << labels << endl;
	vector<float> v(center_num);
	int label_num = 0;
	int label_block = 0;
	for (int class_num = 0; class_num < CLASS_NUM; class_num++)
	{
		for (int image_num = 0; image_num < NUM_IMAGES_PERCLASS; image_num++)
		{
			label_block = label_block + image_label_list[class_num*NUM_IMAGES_PERCLASS +image_num];

			for (; label_num < label_block; label_num++)
			{
				int label = labels.at<int>(label_num, 0);
				float  weight = CalSimilarity(vocabulary.row(label), features.row(label_num));
				v[label] = v[label] + weight;

			}

			Mat frequency(v, true);
			for_each(v.begin(), v.end(), [&](uchar t){t = 0; });
			Mat single_normalized_frequency(1, center_num, CV_32FC1);
			for (int i = 0; i < center_num; i++)
			{
				single_normalized_frequency.at<float>(0, i) = frequency.at<float>(i, 0) / static_cast<float>(image_label_list[class_num*NUM_IMAGES_PERCLASS +image_num] + 1);
			}
			normalized_frequency.push_back(single_normalized_frequency);
		}
	}
	NormalizeCols(normalized_frequency);
	return normalized_frequency;
}

void feature::NormalizeCols(Mat &features)
{
	Mat single_col = Mat::zeros(features.rows, 1, features.type());
	for (int i = 0; i < features.cols; i++)
	{
		single_col = features.colRange(i, i + 1);
		normalize(single_col, single_col, 1.0, 0.0, NORM_MINMAX);
		features.col(i) = single_col;

	}
	for (int i = 0; i < features.rows; i++){
		float* ptrData = (float*)features.ptr<float>(i);
		for (int j = 0; j < features.cols; j++){
			if (ptrData[j] < 0){
				ptrData[j] *= -1;
			}
		}
	}
}

Mat feature::GaussianKernel2D(int dim, float sigma)
{
	CvMat* mat = cvCreateMat(dim, dim, CV_32FC1);
	CvMat* mat1 = cvCreateMat(dim, dim, CV_32FC1);
	CvMat* mat2 = cvCreateMat(dim, dim, CV_32FC1);
#define Mat2(ROW,COL) ((float *)(mat2->data.fl + mat2->step/sizeof(float) *(ROW)))[(COL)]  
#define Mat1(ROW,COL) ((float *)(mat1->data.fl + mat1->step/sizeof(float) *(ROW)))[(COL)] 
#define Mat3(ROW,COL) ((float *)(mat->data.fl + mat->step/sizeof(float) *(ROW)))[(COL)] 
	float s2 = sigma * sigma;
	float m = 1.0 / (sqrt(2.0 * CV_PI) * sigma);
	if ((dim % 2) != 0)   //dim is a  odd
	{
		int c = dim / 2;
		for (int i = 0; i < dim; i++)
		{
			for (int j = 0; j < dim; j++)
			{
				printf("%d %d %d/n", c, i, j);
				float v = m * exp(-(1.0*i*i + 1.0*j*j) / (2.0 * s2));
				//float v = exp(-(1.0*i*i + 1.0*j*j) / s2); 
				Mat3(c + i, c + j) = v;
				Mat3(c - i, c + j) = v;
				Mat3(c + i, c - j) = v;
				Mat3(c - i, c - j) = v;
			}
		}
	}
	else
	{
		for (int i = 0; i < dim; i++)
		{
			for (int j = 0; j < dim; j++)
			{
				Mat1(i, j) = i;
				Mat2(j, i) = i;
			}
		}
		for (int i = 0; i < dim; i++)
		{
			for (int j = 0; j < dim; j++)
			{
				Mat3(i, j) = m * exp(-(((Mat1(i, j) - (float)(dim - 1) / 2)*(Mat1(i, j) - (float)(dim - 1) / 2) + (Mat2(i, j) \
					- (float)(dim - 1) / 2)*(Mat2(i, j) - (float)(dim - 1) / 2)) / (2.0 * s2)));
			}
		}
	}
	cv::Mat nn = cv::Mat(mat, true);
	normalize(nn, nn, 1.0, 0, NORM_MINMAX);
	return nn;
}

vector<float> feature::CalculateHueSift(Mat& himage, cv::Point keypoint)
{
	vector<float> hsift(16, 0);
	
		//CLog *mylog = LogManager::OpenLog("D:/log.txt");
		Mat gauss_kernel = GaussianKernel2D(16, 20);
		cv::Rect range = cvRect(keypoint.x - 7, keypoint.y - 7, 16, 16);


		Mat roi_image;


		
	        

	

try{
		for (int i = keypoint.x-7; i <=keypoint.x+8; i++)
		{
			for (int j = keypoint.y-7; j <=keypoint.y+8; j++)
			{
				for (int k = 0; k < 16; k++)
				{
					if (himage.at<uchar>(j, i) >= 16 * k&&himage.at<uchar>(j, i) < 16 * (k + 1))
					{
						hsift[k] = hsift[k] + gauss_kernel.at<float>(j - keypoint.y + 7, i-keypoint.x + 7);
						break;
					}
				}
			}
		}
	
		throw "error";
}
catch (...){ qDebug() << "error"; }

	return hsift;
}

float feature::CalSimilarity(Mat&vocabulary, Mat &single_feature)
{
	float inner_product = 0;
	float module1 = 0, module2 = 0;
	for (int i = 0; i < vocabulary.cols; i++)
	{
		inner_product = inner_product + vocabulary.at<float>(0, i)*single_feature.at<float>(0, i);
		module1 = module1 + vocabulary.at<float>(0, i)*vocabulary.at<float>(0, i);
		module2 = module2 + single_feature.at<float>(0, i)*single_feature.at<float>(0, i);
	}
	float similarity = inner_product / (sqrt(module2*module1));
	return similarity;
}