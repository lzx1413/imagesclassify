#ifndef IMAGECLASSIFY_H
#define IMAGECLASSIFY_H
#include "Log/LogManager.h"
#include "Log/Log.h"
#include <vector>
#include <QtWidgets/QMainWindow>
#include "ui_imageclassify.h"
#include <opencv2/core/core.hpp>

class ImageClassify : public QMainWindow
{
	Q_OBJECT

public:
	ImageClassify(QWidget *parent = 0);
	~ImageClassify();
	void ImageClassifyProcess();
//private slots:
void GetImageFromFile();

private:
	Ui::ImageClassifyClass ui;
	std::vector<int> label_for_train;
	std::vector<cv::Mat>image_for_train;
	std::vector<int>label_for_test;
	std::vector<cv::Mat>image_for_test;
	CLog *mylog = LogManager::OpenLog("D:/log.txt");
	cv::Mat mergeRows(cv::Mat A, cv::Mat B);

};

#endif // IMAGECLASSIFY_H
