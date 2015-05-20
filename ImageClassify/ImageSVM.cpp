#include "ImageSVM.h"
#include <opencv2/ml/ml.hpp>
void svm::TrainAndPredict(Mat &train_data, Mat &test_data)
{
	vector<int> labels;
	for (int i = 0; i < 19;i++)
	{
		for (int j = 0; j < 15;j++)
		{
			labels.push_back(i + 1);
		}
	}
	Mat label_mat(labels, true);
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	CvSVM SVM;
	SVM.train(train_data, label_mat, Mat(), Mat(), params);
	CvMat test_data_ = test_data;
	CvMat predict_labels;
	//SVM.predict(&test_data_, &predict_labels);
	//Mat predict_labels_(const CvMat* predict_labels, bool copyData = false);
	//cout << "predict labels" << predict_labels_ << endl;
}