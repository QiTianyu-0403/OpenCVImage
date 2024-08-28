#include "mainwindow.h"

#include <QApplication>

int main( int argc, char* argv[] )
{
  std::vector<std::vector<cv::Point>> contours;

  QApplication a( argc, argv );
  MainWindow w;
  w.show();
  return QApplication::exec();
}
