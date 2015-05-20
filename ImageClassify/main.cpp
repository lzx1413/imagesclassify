#include "imageclassify.h"
#include <QtWidgets/QApplication>
int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	ImageClassify w;
	w.GetImageFromFile();
	w.ImageClassifyProcess();
	w.show();
	return a.exec();
}
