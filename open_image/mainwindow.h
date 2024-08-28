#pragma once
// #pragma execution_character_set("utf-8")//display chinese words

#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <QFileDialog>
#include <QVector>
// #include <iostream>
// #include <algorithm>
// #include <limits>
#include <qmath.h>
// #include <vector>
#include <opencv2/calib3d/calib3d.hpp>
// #include "opencv2/calib3d.hpp"
//  #include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>
// #include <stdio.h>
// #include <fstream>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui.hpp>
// #include <opencv2/optflow/motempl.hpp>
// #include <opencv2/xfeatures2d.hpp>
// #include "opencv2/imgcodecs/legacy/constants_c.h"
// using namespace cv;
// using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
  // Q_OBJECT
private:
  Ui::MainWindow* ui;
  cv::Mat srcImg, grayImg, noiseImg;
  std::string openFile( const char* filterDesc );

public:
  MainWindow( QWidget* parent = nullptr );
  ~MainWindow();

private slots:
  void on_pushButton_clicked();

  //    void on_checkBox_stateChanged();

  void on_select_files_clicked();

  void on_gray_leval_clicked();

  void on_gray_balance_clicked();

  void on_grad_sharpen_clicked();

  void on_laplace_sharpen_clicked();

  void on_roberts_edge_clicked();

  void on_sobel_edge_clicked();

  void on_prewitt_clicked();

  void on_laplace_edge_clicked();

  void on_salt_noise_clicked();

  void on_guass_noise_clicked();

  void on_krisch_edge_clicked();

  void on_Canny_clicked();

  void on_average_filter_clicked();

  void on_middle_filter_clicked();

  void on_window_filter_clicked();

  void on_gauss_filter_clicked();

  void on_form_filter_clicked();

  void on_affine_clicked();

  void on_perspective_clicked();

  void on_threshold_seg_clicked();

  void on_OSTU_clicked();

  void on_Kittler_clicked();

  void on_frame_diff_clicked();

  void on_mix_guass_clicked();

  void on_circle_lbp_clicked();

  void on_target_det_clicked();

  void on_model_check_clicked();

  void on_cloaking_clicked();

  void on_SIFT_clicked();

  void on_orb_clicked();

  void on_color_fit_clicked();

  static void on_svm_test_clicked();

  void on_word_test_clicked();

  void on_Haar_1_clicked();

  void on_Haar_2_clicked();

  void on_gaber_clicked();

  void on_face_haar_clicked();

  static void on_camera2_clicked();

  void on_camera2_2_clicked();
};