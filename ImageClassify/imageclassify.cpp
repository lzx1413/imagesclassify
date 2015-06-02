#include "imageclassify.h"
#include <opencv2/highgui/highgui.hpp>
#include<QDir>
#include<iostream>
#include<QDebug>
#include"tools/utls.h"
#include "ImageFeatures.h"
#include "ImageSVM.h"
#include<opencv2/contrib/contrib.hpp>
#include<thread>
inline static bool CheckInArray(int number, int a[], int count)
{
	for (int i = 0; i < count;i++)
	{
		if (number ==a[i] )
		{
			return true;
		}
	} 
    return false;
}
ImageClassify::ImageClassify(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	mylog->ClearLogFile();
}

ImageClassify::~ImageClassify()
{

}
void ImageClassify::GetImageFromFile()
{
	TickMeter time;
	time.start();
	QDir image_file("D:/Images");
	std::vector<cv::Mat> images;
	std::vector<int> labels;
	if (!image_file.exists())
	{
		mylog->WriteLog("can not get image file", CLog::LL_ERROR);

	}
	image_file.setFilter(QDir::Files | QDir::NoSymLinks);
	image_file.setSorting(QDir::NoSort);
	QFileInfoList image_list = image_file.entryInfoList();
	int label = 1;
	for (int i = 0; i < image_list.size();i++)
	{
		qDebug() << image_list.at(i).fileName() << endl;
		labels.push_back(label);
		if (!((i+1)%30))
		{
			label ++;
		}
		std::string image_name = "D:/Images/" + image_list.at(i).fileName().toStdString();
		images.push_back(cv::imread(image_name));
   }
	
	std::vector<cv::Mat>::iterator image_pointer = images.begin();
	int train_num = 1;
	int test_num = 1;
	for (int i = 1; image_pointer!=images.end();i++)
	{
		int rand_array[15] = { 0 };
		GetRandArray(rand_array, 15);
		int j = 0;
		for (;j < 30;j++)
		{
		
			if (CheckInArray(j, rand_array, 15))
			{
				image_for_train.push_back(*(image_pointer + j));
				qDebug() << "add a train image" << train_num << endl;
				label_for_train.push_back(i);
				train_num++;
			}
			else
			{
				image_for_test.push_back(*(image_pointer + j));
				label_for_test.push_back(i);
				qDebug()<< "add a test image" << test_num << endl;
			}
		
		}
	    image_pointer = image_pointer + j;
	}
	//for_each(image_for_train.begin(), image_for_train.end(), [&](cv::Mat img){cv::imshow("image for train", img); cv::waitKey(); });
	time.stop();
	qDebug() << "images load complete" << time.getTimeSec() << endl;


}
Mat ImageClassify::mergeRows(Mat A, Mat B)
{
	//CV_ASSERT(A.cols == B.cols&&A.type() == B.type());
	int totalRows = A.rows + B.rows;

	Mat mergedDescriptors(totalRows, A.cols, A.type());
	Mat submat = mergedDescriptors.rowRange(0, A.rows);
	A.copyTo(submat);
	submat = mergedDescriptors.rowRange(A.rows, totalRows);
	B.copyTo(submat);
	return mergedDescriptors;
}
void ImageClassify::ImageClassifyProcess()
{
	std::thread t1,t2;
	Mat train_image_features, test_image_features;
	TickMeter time;
	time.start();
	
 //   vector<int>train_labels = feature::GetImageFeatures(image_for_train,train_image_features);
	//Mat train_labels_(train_labels, true);
	//time.stop();
	//FileStorage fs(".\\train_feature.xml", FileStorage::WRITE);
	//fs << "train_feature" << train_image_features;
	//fs << "train_labels" << train_labels_;
	//qDebug() << "train image features got" << time.getTimeSec()<<endl;
	//time.reset();
	//time.start();
	//vector<int>test_labels = feature::GetImageFeatures(image_for_test,test_image_features);
	//Mat test_labels_(test_labels, true);
	//fs << "test_feature" << test_image_features;
	//fs << "test_labels" << test_labels_;
	//fs.release();
	//time.stop();
	//qDebug() << "test image features got" <<time.getTimeSec()<< endl;
	//time.reset();
	time.start();
	FileStorage fs1(".\\train_feature.xml", FileStorage::READ);
	Mat train_labels1, test_labels1;
	fs1["train_feature" ]>> train_image_features;
	fs1["train_labels"] >> train_labels1;
	fs1["test_feature"] >> test_image_features;
	fs1["test_labels"] >> test_labels1;
	vector<int> train_labels2, test_labels2;
	train_labels2.assign((int*)train_labels1.datastart, (int*)train_labels1.dataend);
	test_labels2.assign((int*)test_labels1.datastart, (int*)test_labels1.dataend);
	test_labels2.insert(test_labels2.end(), train_labels2.begin(), train_labels2.end());
	//vector<int> labels2(train_labels2.size() + test_labels2.size());
	//merge(train_labels2.begin(),train_labels2.end(),test_labels2.begin(),test_labels2.end(),labels2.begin());
	Mat image_features = mergeRows(train_image_features, test_image_features);
	fs1.release();
	Mat vector_for_image = feature::GetVocabularyFrequency(image_features, 300, test_labels2);
	Mat vector_for_train = vector_for_image.rowRange(0, 285);
	time.stop();
	qDebug()<< "train image vector got" <<time.getTimeSec()<< endl;
	time.reset();
	time.start();
	Mat vector_for_test = vector_for_image.rowRange(285, vector_for_image.rows);
	time.stop();
	qDebug() << "test image vector got" <<time.getTimeSec()<< endl;
	time.reset();
	time.start();
	FileStorage fs6(".\\train_feature2.xml", FileStorage::WRITE);
	fs6 << "train_feature" << vector_for_train;
	fs6 << "test_feature" << vector_for_test;
	svm::TrainAndPredict(vector_for_train, vector_for_test);
	time.stop();
	qDebug() << "finish svm" << time.getTimeSec()<< endl;
	getchar();
}
