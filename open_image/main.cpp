#include "mainwindow.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    vector<vector<Point>> contours;

    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
