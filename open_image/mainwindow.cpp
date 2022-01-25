#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

Mat gray_to_level(Mat gray)//灰度直方图函数
{
    QVector<int> pixel(256,0);

    for(int i = 0 ; i < gray.rows ; i++)
        for(int j = 0 ; j < gray.cols ; j++){
            pixel[gray.at<uchar>(i,j)]++;
        }

    Mat gray_level;
    gray_level.create(350, 256, CV_8UC1);

    int max_rows = 0;
    for(int i = 0;i <= 255;i++){
        if(pixel[i] > max_rows){
            max_rows = pixel[i];
        }
    }

    for(int i = 0;i < 256;i++){
        for(int j = 0;j < 350 ; j++){
            gray_level.at<uchar>(j,i) = 255;
        }
    }

    for(int i = 0;i < 256;i++){
        for(int j = 0;j < 350 - int(320.*float(pixel[i])/float(max_rows)) ; j++){
            gray_level.at<uchar>(j,i) = 0;
        }
    }

    return gray_level;

}

QVector<int> gray2vector(Mat gray){
    QVector<int> pixel(256,0);

    for(int i = 0 ; i < gray.rows ; i++)
        for(int j = 0 ; j < gray.cols ; j++){
            pixel[gray.at<uchar>(i,j)]++;
        }
    return pixel;
}

Mat addSaltNoise(const Mat srcImage, int n)
{
    Mat dstImage = srcImage.clone();
    for (int k = 0; k < n; k++)
    {
        //随机取值行列
        int i = rand() % dstImage.rows;
        int j = rand() % dstImage.cols;
        //图像通道判定
        if (dstImage.channels() == 1)
        {
            dstImage.at<uchar>(i, j) = 255;		//盐噪声
        }
        else
        {
            dstImage.at<Vec3b>(i, j)[0] = 255;
            dstImage.at<Vec3b>(i, j)[1] = 255;
            dstImage.at<Vec3b>(i, j)[2] = 255;
        }
    }
    for (int k = 0; k < n; k++)
    {
        //随机取值行列
        int i = rand() % dstImage.rows;
        int j = rand() % dstImage.cols;
        //图像通道判定
        if (dstImage.channels() == 1)
        {
            dstImage.at<uchar>(i, j) = 0;		//椒噪声
        }
        else
        {
            dstImage.at<Vec3b>(i, j)[0] = 0;
            dstImage.at<Vec3b>(i, j)[1] = 0;
            dstImage.at<Vec3b>(i, j)[2] = 0;
        }
    }
    return dstImage;
}

double generateGaussianNoise(double mu, double sigma)
{
    //定义小值
    const double epsilon = std::numeric_limits<double>::min();
    static double z0, z1;
    static bool flag = false;
    flag = !flag;
    //flag为假构造高斯随机变量X
    if (!flag)
        return z1 * sigma + mu;
    double u1, u2;
    //构造随机变量
    do
    {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while (u1 <= epsilon);
    //flag为真构造高斯随机变量
    z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI*u2);
    z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI*u2);
    return z0*sigma + mu;
}

//为图像添加高斯噪声
Mat addGaussianNoise(Mat &srcImag)
{
    Mat dstImage = srcImag.clone();
    for (int i = 0; i < dstImage.rows; i++)
    {
        for (int j = 0; j < dstImage.cols; j++)
        {
            //添加高斯噪声
            dstImage.at<Vec3b>(i, j)[0] = saturate_cast<uchar>(dstImage.at<Vec3b>(i, j)[0] + generateGaussianNoise(2, 0.8) * 32);
            dstImage.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(dstImage.at<Vec3b>(i, j)[1] + generateGaussianNoise(2, 0.8) * 32);
            dstImage.at<Vec3b>(i, j)[2] = saturate_cast<uchar>(dstImage.at<Vec3b>(i, j)[2] + generateGaussianNoise(2, 0.8) * 32);
        }
    }
    return dstImage;
}

//canny双阈值处理
void DoubleThreshold(Mat &imageIput,double lowThreshold,double highThreshold)
{
    for(int i=0;i<imageIput.rows;i++)
    {
        for(int j=0;j<imageIput.cols;j++)
        {
            if(imageIput.at<uchar>(i,j)>highThreshold)
            {
                imageIput.at<uchar>(i,j)=255;
            }
            if(imageIput.at<uchar>(i,j)<lowThreshold)
            {
                imageIput.at<uchar>(i,j)=0;
            }
        }
    }
}

//canny双阈值连接
void DoubleThresholdLink(Mat &imageInput,double lowThreshold,double highThreshold)
{
    for(int i=1;i<imageInput.rows-1;i++)
    {
        for(int j=1;j<imageInput.cols-1;j++)
        {
            if(imageInput.at<uchar>(i,j)>lowThreshold&&imageInput.at<uchar>(i,j)<255)
            {
                if(imageInput.at<uchar>(i-1,j-1)==255||imageInput.at<uchar>(i-1,j)==255||imageInput.at<uchar>(i-1,j+1)==255||
                    imageInput.at<uchar>(i,j-1)==255||imageInput.at<uchar>(i,j)==255||imageInput.at<uchar>(i,j+1)==255||
                    imageInput.at<uchar>(i+1,j-1)==255||imageInput.at<uchar>(i+1,j)==255||imageInput.at<uchar>(i+1,j+1)==255)
                {
                    imageInput.at<uchar>(i,j)=255;
                    DoubleThresholdLink(imageInput,lowThreshold,highThreshold); //递归调用
                }
                else
            {
                    imageInput.at<uchar>(i,j)=0;
            }
            }
        }
    }
}

int OSTU(QVector<int> hist){
    float u0, u1, w0, w1; int count0, t, maxT; float devi, maxDevi = 0; //方差及最大方差 int i, sum = 0;
    int i, sum = 0;
    for (i = 0; i < 256; i++){ sum = sum + hist[i]; }

    for (t = 0; t < 255; t++){
        u0 = 0; count0 = 0;
        for (i = 0; i <= t; i++){ //阈值为t时，c0组的均值及产生的概率;
            u0 += i * hist[i];
            count0 += hist[i];
        }
        u0 = u0 / count0;
        w0 = (float)count0/sum;
        for (i = t + 1; i < 256; i++){ //阈值为t时，c1组的均值及产生的概率
            u1 += i * hist[i];
        }
        u1 = u1 / (sum - count0); w1 = 1 - w0;
        devi = w0 * w1 * (u1 - u0) * (u1 - u0); //两类间方差
        if (devi > maxDevi) //记录最大的方差及最佳位置
        {
            maxDevi = devi;
            maxT = t;
        }
    }
    return maxT;
}

void elbp(Mat& src, Mat &dst, int radius, int neighbors)
{

    for (int n = 0; n<neighbors; n++)
    {
        // 采样点的计算
        float x = static_cast<float>(-radius * sin(2.0*CV_PI*n / static_cast<float>(neighbors)));
        float y = static_cast<float>(radius * cos(2.0*CV_PI*n / static_cast<float>(neighbors)));
        // 上取整和下取整的值
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // 小数部分
        float ty = y - fy;
        float tx = x - fx;
        // 设置插值权重
        float w1 = (1 - tx) * (1 - ty);
        float w2 = tx  * (1 - ty);
        float w3 = (1 - tx) * ty;
        float w4 = tx * ty;
        // 循环处理图像数据
        for (int i = radius; i < src.rows - radius; i++)
        {
            for (int j = radius; j < src.cols - radius; j++)
            {
                // 计算插值
                float t = static_cast<float>(w1*src.at<uchar>(i + fy, j + fx) + w2*src.at<uchar>(i + fy, j + cx) + w3*src.at<uchar>(i + cy, j + fx) + w4*src.at<uchar>(i + cy, j + cx));
                // 进行编码
                dst.at<uchar>(i - radius, j - radius) += ((t > src.at<uchar>(i, j)) || (abs(t - src.at<uchar>(i, j)) < numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}

void elbp1(Mat& src, Mat &dst)
{
    // 循环处理图像数据
    for (int i = 1; i < src.rows - 1; i++)
    {
        for (int j = 1; j < src.cols - 1; j++)
        {
            uchar tt = 0;
            int tt1 = 0;
            uchar u = src.at<uchar>(i, j);
            if (src.at<uchar>(i - 1, j - 1)>u) { tt += 1 << tt1; }
            tt1++;
            if (src.at<uchar>(i - 1, j)>u) { tt += 1 << tt1; }
            tt1++;
            if (src.at<uchar>(i - 1, j + 1)>u) { tt += 1 << tt1; }
            tt1++;
            if (src.at<uchar>(i, j + 1)>u) { tt += 1 << tt1; }
            tt1++;
            if (src.at<uchar>(i + 1, j + 1)>u) { tt += 1 << tt1; }
            tt1++;
            if (src.at<uchar>(i + 1, j)>u) { tt += 1 << tt1; }
            tt1++;
            if (src.at<uchar>(i + 1, j - 1)>u) { tt += 1 << tt1; }
            tt1++;
            if (src.at<uchar>(i - 1, j)>u) { tt += 1 << tt1; }
            tt1++;

            dst.at<uchar>(i - 1, j - 1) = tt;
        }
    }
}


void MainWindow::on_pushButton_clicked()//选择文件
{
    QString testFileName = QFileDialog::getOpenFileName(this,tr(""),"../../../../open_image","files(*)");
    srcImg = imread(testFileName.toStdString());
    cvtColor(srcImg, grayImg, CV_BGR2GRAY);

    Mat temp;
    QImage Qtemp;
    cvtColor(srcImg, temp, CV_BGR2RGB);//BGR convert to RGB
    Qtemp = QImage((const unsigned char*)(temp.data), temp.cols, temp.rows, temp.step, QImage::Format_RGB888);

    ui->label->setPixmap(QPixmap::fromImage(Qtemp));
    Qtemp = Qtemp.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label->setScaledContents(true);
    ui->label->resize(Qtemp.size());
    ui->label->show();

}

void MainWindow::on_select_files_clicked()//BGR转灰度
{
    //Mat gray;
    grayImg.create(srcImg.rows, srcImg.cols, CV_8UC1);
    QImage Qtemp;

    for(int i = 0 ; i < srcImg.rows ; i++)
        for(int j = 0 ; j < srcImg.cols ; j++){
            grayImg.at<uchar>(i,j) = (int)0.11 * srcImg.at<Vec3b>(i,j)[0]
                                        + 0.59 * srcImg.at<Vec3b>(i,j)[1]
                                        + 0.3 * srcImg.at<Vec3b>(i,j)[2];
        }

    Qtemp = QImage((const uchar*)(grayImg.data), grayImg.cols, grayImg.rows, grayImg.cols*grayImg.channels(), QImage::Format_Indexed8);
    ui->label_1->setPixmap(QPixmap::fromImage(Qtemp));
    Qtemp = Qtemp.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_1->setScaledContents(true);
    ui->label_1->resize(Qtemp.size());
    ui->label_1->show();
}

void MainWindow::on_gray_leval_clicked()//灰度直方图
{

    //Mat gray;

    QImage Qtemp;

    Qtemp = QImage((const uchar*)(grayImg.data), grayImg.cols, grayImg.rows, grayImg.cols*grayImg.channels(), QImage::Format_Indexed8);
    ui->label_1->setPixmap(QPixmap::fromImage(Qtemp));
    Qtemp = Qtemp.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_1->setScaledContents(true);
    ui->label_1->resize(Qtemp.size());
    ui->label_1->show();

    Mat gray_level;
    gray_level = gray_to_level(grayImg);

    imshow("gray_level",gray_level);
    waitKey(0);
    cv::destroyWindow("gray_level");
    waitKey(1);

}

void MainWindow::on_gray_balance_clicked()
{
    Mat balance,gray2Img;
    gray2Img.create(srcImg.rows, srcImg.cols, CV_8UC1);
    balance.create(srcImg.rows, srcImg.cols, CV_8UC1);
    QImage Qtemp;

//    for(int i = 0 ; i < srcImg.rows ; i++)
//        for(int j = 0 ; j < srcImg.cols ; j++){
//            grayImg.at<uchar>(i,j) = (int)0.11 * srcImg.at<Vec3b>(i,j)[0]
//                                        + 0.59 * srcImg.at<Vec3b>(i,j)[1]
//                                        + 0.3 * srcImg.at<Vec3b>(i,j)[2];
//        }
    QVector<int> pixel(256,0);
    QVector<float> pixel_gray(256,0.);
    float sum=0;

    for(int i = 0 ; i < grayImg.rows ; i++)
        for(int j = 0 ; j < grayImg.cols ; j++){
            pixel[grayImg.at<uchar>(i,j)]++;
        }

    for(int i = 0 ; i < pixel.size() ; i++){
        sum += pixel[i];
    }

    for(int i = 0 ; i < 256 ; i++){
        float num = 0;
        for(int j = 0 ; j <= i ; j++){
            num += pixel[j];
        }
        pixel_gray[i] = 255*num/sum;
    }

    for(int i = 0 ; i < srcImg.rows ; i++)
        for(int j = 0 ; j < srcImg.cols ; j++){
            balance.at<uchar>(i,j) = pixel_gray[grayImg.at<uchar>(i,j)];
        }

    gray2Img = balance;

    Qtemp = QImage((const uchar*)(gray2Img.data), gray2Img.cols, gray2Img.rows, gray2Img.cols*gray2Img.channels(), QImage::Format_Indexed8);
    ui->label_3->setPixmap(QPixmap::fromImage(Qtemp));
    Qtemp = Qtemp.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_3->setScaledContents(true);
    ui->label_3->resize(Qtemp.size());
    ui->label_3->show();
}

void MainWindow::on_grad_sharpen_clicked()
{
    Mat grad,gray2Img;
    gray2Img.create(srcImg.rows, srcImg.cols, CV_8UC1);
    QImage Qtemp,Qtemp2;
    grad.create(gray2Img.rows, gray2Img.cols, CV_8UC1);
    for(int i = 0 ; i < gray2Img.rows - 1 ; i++)
        for(int j = 0 ; j < gray2Img.cols - 1 ; j++){
            grad.at<uchar>(i,j) = saturate_cast<uchar>(max(fabs(grayImg.at<uchar>(i+1, j)-grayImg.at<uchar>(i,j)),fabs(grayImg.at<uchar>(i, j+1)-grayImg.at<uchar>(i,j))));
            gray2Img.at<uchar>(i,j) = saturate_cast<uchar>(grayImg.at<uchar>(i,j) - grad.at<uchar>(i,j));
        }
    //imshow("grad",grad);

    Qtemp = QImage((const uchar*)(gray2Img.data), gray2Img.cols, gray2Img.rows, gray2Img.cols*gray2Img.channels(), QImage::Format_Indexed8);
    ui->label_3->setPixmap(QPixmap::fromImage(Qtemp));
    Qtemp = Qtemp.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_3->setScaledContents(true);
    ui->label_3->resize(Qtemp.size());
    ui->label_3->show();

    Qtemp2 = QImage((const uchar*)(grad.data), grad.cols, grad.rows, grad.cols*grad.channels(), QImage::Format_Indexed8);
    ui->label_2->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_2->setScaledContents(true);
    ui->label_2->resize(Qtemp2.size());
    ui->label_2->show();

}

void MainWindow::on_laplace_sharpen_clicked()
{
    Mat gradimg,gray2Img;
    QImage Qtemp,Qtemp2;
    gray2Img.create(grayImg.rows, grayImg.cols, CV_8UC1);
    gradimg.create(grayImg.rows, grayImg.cols, CV_8UC1);
    for (int i = 1; i < srcImg.rows - 1; i++)
    {
        for (int j = 1; j < srcImg.cols - 1; j++)
        {
            gradimg.at<uchar>(i, j) = saturate_cast<uchar>(- 4 * grayImg.at<uchar>(i, j) + grayImg.at<uchar>(i + 1 , j)
                                                          + grayImg.at<uchar>(i, j + 1) + grayImg.at<uchar>(i - 1, j)
                                                          + grayImg.at<uchar>(i, j - 1));
            gray2Img.at<uchar>(i, j) = saturate_cast<uchar>(5 * grayImg.at<uchar>(i, j) - grayImg.at<uchar>(i + 1, j)
                                                        - grayImg.at<uchar>(i, j + 1) - grayImg.at<uchar>(i - 1, j)
                                                        - grayImg.at<uchar>(i, j - 1));
        }
    }
    Qtemp = QImage((const uchar*)(gray2Img.data), gray2Img.cols, gray2Img.rows, gray2Img.cols*gray2Img.channels(), QImage::Format_Indexed8);
    ui->label_3->setPixmap(QPixmap::fromImage(Qtemp));
    Qtemp = Qtemp.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_3->setScaledContents(true);
    ui->label_3->resize(Qtemp.size());
    ui->label_3->show();

    Qtemp2 = QImage((const uchar*)(gradimg.data), gradimg.cols, gradimg.rows, gradimg.cols*gradimg.channels(), QImage::Format_Indexed8);
    ui->label_2->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_2->setScaledContents(true);
    ui->label_2->resize(Qtemp2.size());
    ui->label_2->show();
}

void MainWindow::on_roberts_edge_clicked()
{
    Mat gradimg,gray2Img;
    QImage Qtemp,Qtemp2;
    gray2Img.create(grayImg.rows, grayImg.cols, CV_8UC1);
    gradimg.create(grayImg.rows, grayImg.cols, CV_8UC1);
    for (int i = 0; i < srcImg.rows - 1; i++)
    {
        for (int j = 0; j < srcImg.cols - 1; j++)
        {
            gradimg.at<uchar>(i, j) = saturate_cast<uchar>(fabs(grayImg.at<uchar>(i, j) - grayImg.at<uchar>(i + 1, j + 1)) + fabs(grayImg.at<uchar>(i + 1, j) - grayImg.at<uchar>(i, j + 1)));
            gray2Img.at<uchar>(i, j) = saturate_cast<uchar>(grayImg.at<uchar>(i, j) - gradimg.at<uchar>(i, j));
        }
    }
    Qtemp = QImage((const uchar*)(gray2Img.data), gray2Img.cols, gray2Img.rows, gray2Img.cols*gray2Img.channels(), QImage::Format_Indexed8);
    ui->label_3->setPixmap(QPixmap::fromImage(Qtemp));
    Qtemp = Qtemp.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_3->setScaledContents(true);
    ui->label_3->resize(Qtemp.size());
    ui->label_3->show();

    Qtemp2 = QImage((const uchar*)(gradimg.data), gradimg.cols, gradimg.rows, gradimg.cols*gradimg.channels(), QImage::Format_Indexed8);
    ui->label_2->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_2->setScaledContents(true);
    ui->label_2->resize(Qtemp2.size());
    ui->label_2->show();
}

void MainWindow::on_sobel_edge_clicked()
{
    Mat gradimg,gray2Img,f_x,f_y;
    QImage Qtemp,Qtemp2;
    gray2Img.create(grayImg.rows, grayImg.cols, CV_8UC1);
    gradimg.create(grayImg.rows, grayImg.cols, CV_8UC1);
    f_x.create(grayImg.rows, grayImg.cols, CV_8UC1);
    f_y.create(grayImg.rows, grayImg.cols, CV_8UC1);
    for (int i = 1; i < srcImg.rows - 1; i++)
    {
        for (int j = 1; j < srcImg.cols - 1; j++)
        {
            f_x.at<uchar>(i, j) = saturate_cast<uchar>(fabs(grayImg.at<uchar>(i + 1, j - 1) + 2*grayImg.at<uchar>(i + 1, j) + grayImg.at<uchar>(i + 1, j + 1)
                                                            - grayImg.at<uchar>(i - 1, j - 1) - 2*grayImg.at<uchar>(i - 1, j) - grayImg.at<uchar>(i - 1, j + 1)));
            f_y.at<uchar>(i, j) = saturate_cast<uchar>(fabs(grayImg.at<uchar>(i - 1, j + 1) + 2*grayImg.at<uchar>(i, j + 1) + grayImg.at<uchar>(i + 1, j + 1)
                                                            - grayImg.at<uchar>(i - 1, j - 1) - 2*grayImg.at<uchar>(i, j - 1) - grayImg.at<uchar>(i + 1, j - 1)));
            gradimg.at<uchar>(i, j) = f_x.at<uchar>(i, j) + f_y.at<uchar>(i, j);
            gray2Img.at<uchar>(i, j) = saturate_cast<uchar>(grayImg.at<uchar>(i, j) - gradimg.at<uchar>(i, j));
        }
    }
    Qtemp = QImage((const uchar*)(gray2Img.data), gray2Img.cols, gray2Img.rows, gray2Img.cols*gray2Img.channels(), QImage::Format_Indexed8);
    ui->label_3->setPixmap(QPixmap::fromImage(Qtemp));
    Qtemp = Qtemp.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_3->setScaledContents(true);
    ui->label_3->resize(Qtemp.size());
    ui->label_3->show();

    Qtemp2 = QImage((const uchar*)(gradimg.data), gradimg.cols, gradimg.rows, gradimg.cols*gradimg.channels(), QImage::Format_Indexed8);
    ui->label_2->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_2->setScaledContents(true);
    ui->label_2->resize(Qtemp2.size());
    ui->label_2->show();
}

void MainWindow::on_prewitt_clicked()
{
    Mat gradimg,gray2Img,f_x,f_y;
    QImage Qtemp,Qtemp2;
    gray2Img.create(grayImg.rows, grayImg.cols, CV_8UC1);
    gradimg.create(grayImg.rows, grayImg.cols, CV_8UC1);
    f_x.create(grayImg.rows, grayImg.cols, CV_8UC1);
    f_y.create(grayImg.rows, grayImg.cols, CV_8UC1);
    for (int i = 1; i < srcImg.rows - 1; i++)
    {
        for (int j = 1; j < srcImg.cols - 1; j++)
        {
            f_x.at<uchar>(i, j) = saturate_cast<uchar>(fabs(grayImg.at<uchar>(i + 1, j - 1) + grayImg.at<uchar>(i + 1, j) + grayImg.at<uchar>(i + 1, j + 1)
                                                            - grayImg.at<uchar>(i - 1, j - 1) - grayImg.at<uchar>(i - 1, j) - grayImg.at<uchar>(i - 1, j + 1)));
            f_y.at<uchar>(i, j) = saturate_cast<uchar>(fabs(grayImg.at<uchar>(i - 1, j + 1) + grayImg.at<uchar>(i, j + 1) + grayImg.at<uchar>(i + 1, j + 1)
                                                            - grayImg.at<uchar>(i - 1, j - 1) - grayImg.at<uchar>(i, j - 1) - grayImg.at<uchar>(i + 1, j - 1)));
            gradimg.at<uchar>(i, j) = max(f_x.at<uchar>(i, j),f_y.at<uchar>(i, j));
            gray2Img.at<uchar>(i, j) = saturate_cast<uchar>(grayImg.at<uchar>(i, j) - gradimg.at<uchar>(i, j));
        }
    }
    Qtemp = QImage((const uchar*)(gray2Img.data), gray2Img.cols, gray2Img.rows, gray2Img.cols*gray2Img.channels(), QImage::Format_Indexed8);
    ui->label_3->setPixmap(QPixmap::fromImage(Qtemp));
    Qtemp = Qtemp.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_3->setScaledContents(true);
    ui->label_3->resize(Qtemp.size());
    ui->label_3->show();

    Qtemp2 = QImage((const uchar*)(gradimg.data), gradimg.cols, gradimg.rows, gradimg.cols*gradimg.channels(), QImage::Format_Indexed8);
    ui->label_2->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_2->setScaledContents(true);
    ui->label_2->resize(Qtemp2.size());
    ui->label_2->show();
}

void MainWindow::on_laplace_edge_clicked()
{
    Mat gradimg,gray2Img;
    QImage Qtemp,Qtemp2;
    gray2Img.create(grayImg.rows, grayImg.cols, CV_8UC1);
    gradimg.create(grayImg.rows, grayImg.cols, CV_8UC1);
    for (int i = 1; i < srcImg.rows - 1; i++)
    {
        for (int j = 1; j < srcImg.cols - 1; j++)
        {
            gradimg.at<uchar>(i, j) = saturate_cast<uchar>(- 4 * grayImg.at<uchar>(i, j) + grayImg.at<uchar>(i + 1 , j)
                                                          + grayImg.at<uchar>(i, j + 1) + grayImg.at<uchar>(i - 1, j)
                                                          + grayImg.at<uchar>(i, j - 1));
            gray2Img.at<uchar>(i, j) = saturate_cast<uchar>(5 * grayImg.at<uchar>(i, j) - grayImg.at<uchar>(i + 1, j)
                                                        - grayImg.at<uchar>(i, j + 1) - grayImg.at<uchar>(i - 1, j)
                                                        - grayImg.at<uchar>(i, j - 1));
        }
    }
    Qtemp = QImage((const uchar*)(gray2Img.data), gray2Img.cols, gray2Img.rows, gray2Img.cols*gray2Img.channels(), QImage::Format_Indexed8);
    ui->label_3->setPixmap(QPixmap::fromImage(Qtemp));
    Qtemp = Qtemp.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_3->setScaledContents(true);
    ui->label_3->resize(Qtemp.size());
    ui->label_3->show();

    Qtemp2 = QImage((const uchar*)(gradimg.data), gradimg.cols, gradimg.rows, gradimg.cols*gradimg.channels(), QImage::Format_Indexed8);
    ui->label_2->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_2->setScaledContents(true);
    ui->label_2->resize(Qtemp2.size());
    ui->label_2->show();
}

void MainWindow::on_salt_noise_clicked()
{
    Mat salt;
    Mat temp;
    salt.create(srcImg.rows, srcImg.cols, CV_8UC3);
    salt = addSaltNoise(srcImg,800);
    QImage Qtemp2;
    cvtColor(salt, temp, CV_BGR2RGB);//BGR convert to RGB

    noiseImg=temp.clone();

    Qtemp2 = QImage((const unsigned char*)(temp.data), temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
    ui->label_2->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_2->setScaledContents(true);
    ui->label_2->resize(Qtemp2.size());
    ui->label_2->show();

}

void MainWindow::on_guass_noise_clicked()
{
    Mat salt;
    Mat temp;
    salt.create(srcImg.rows, srcImg.cols, CV_8UC3);
    salt = addGaussianNoise(srcImg);
    QImage Qtemp2;
    cvtColor(salt, temp, CV_BGR2RGB);//BGR convert to RGB
    noiseImg=temp.clone();

    Qtemp2 = QImage((const unsigned char*)(temp.data), temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
    ui->label_2->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_2->setScaledContents(true);
    ui->label_2->resize(Qtemp2.size());
    ui->label_2->show();

}

void MainWindow::on_krisch_edge_clicked()
{
    Mat gradimg,gray2Img,f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8;
    QImage Qtemp,Qtemp2;
    gray2Img.create(grayImg.rows, grayImg.cols, CV_8UC1);
    gradimg.create(grayImg.rows, grayImg.cols, CV_8UC1);
    f_1.create(grayImg.rows, grayImg.cols, CV_8UC1);
    f_2.create(grayImg.rows, grayImg.cols, CV_8UC1);
    f_3.create(grayImg.rows, grayImg.cols, CV_8UC1);
    f_4.create(grayImg.rows, grayImg.cols, CV_8UC1);
    f_5.create(grayImg.rows, grayImg.cols, CV_8UC1);
    f_6.create(grayImg.rows, grayImg.cols, CV_8UC1);
    f_7.create(grayImg.rows, grayImg.cols, CV_8UC1);
    f_8.create(grayImg.rows, grayImg.cols, CV_8UC1);
    for (int i = 1; i < srcImg.rows - 1; i++)
    {
        for (int j = 1; j < srcImg.cols - 1; j++)
        {
            f_1.at<uchar>(i, j) = saturate_cast<uchar>(fabs(-5*grayImg.at<uchar>(i - 1, j - 1) -5* grayImg.at<uchar>(i - 1, j) -5* grayImg.at<uchar>(i - 1, j + 1)
                                                           +3*grayImg.at<uchar>(i, j - 1) +3*grayImg.at<uchar>(i, j + 1)
                                                           + 3*grayImg.at<uchar>(i + 1, j - 1) + 3*grayImg.at<uchar>(i + 1, j) + 3*grayImg.at<uchar>(i + 1, j + 1)));
            f_2.at<uchar>(i, j) = saturate_cast<uchar>(fabs(3*grayImg.at<uchar>(i - 1, j - 1) -5* grayImg.at<uchar>(i - 1, j) -5* grayImg.at<uchar>(i - 1, j + 1)
                                                           +3*grayImg.at<uchar>(i, j - 1) -5*grayImg.at<uchar>(i, j + 1)
                                                           + 3*grayImg.at<uchar>(i + 1, j - 1) + 3*grayImg.at<uchar>(i + 1, j) + 3*grayImg.at<uchar>(i + 1, j + 1)));
            f_3.at<uchar>(i, j) = saturate_cast<uchar>(fabs(3*grayImg.at<uchar>(i - 1, j - 1) +3* grayImg.at<uchar>(i - 1, j) -5* grayImg.at<uchar>(i - 1, j + 1)
                                                           +3*grayImg.at<uchar>(i, j - 1) -5*grayImg.at<uchar>(i, j + 1)
                                                           + 3*grayImg.at<uchar>(i + 1, j - 1) + 3*grayImg.at<uchar>(i + 1, j) -5*grayImg.at<uchar>(i + 1, j + 1)));
            f_4.at<uchar>(i, j) = saturate_cast<uchar>(fabs(3*grayImg.at<uchar>(i - 1, j - 1) +3* grayImg.at<uchar>(i - 1, j) +3* grayImg.at<uchar>(i - 1, j + 1)
                                                           +3*grayImg.at<uchar>(i, j - 1) -5*grayImg.at<uchar>(i, j + 1)
                                                           + 3*grayImg.at<uchar>(i + 1, j - 1) -5*grayImg.at<uchar>(i + 1, j) -5*grayImg.at<uchar>(i + 1, j + 1)));
            f_5.at<uchar>(i, j) = saturate_cast<uchar>(fabs(3*grayImg.at<uchar>(i - 1, j - 1) +3* grayImg.at<uchar>(i - 1, j) +3* grayImg.at<uchar>(i - 1, j + 1)
                                                           +3*grayImg.at<uchar>(i, j - 1) +3*grayImg.at<uchar>(i, j + 1)
                                                           -5*grayImg.at<uchar>(i + 1, j - 1) -5*grayImg.at<uchar>(i + 1, j) -5*grayImg.at<uchar>(i + 1, j + 1)));
            f_6.at<uchar>(i, j) = saturate_cast<uchar>(fabs(3*grayImg.at<uchar>(i - 1, j - 1) +3* grayImg.at<uchar>(i - 1, j) +3* grayImg.at<uchar>(i - 1, j + 1)
                                                           -5*grayImg.at<uchar>(i, j - 1) +3*grayImg.at<uchar>(i, j + 1)
                                                           -5*grayImg.at<uchar>(i + 1, j - 1) -5*grayImg.at<uchar>(i + 1, j) +3*grayImg.at<uchar>(i + 1, j + 1)));
            f_7.at<uchar>(i, j) = saturate_cast<uchar>(fabs(-5*grayImg.at<uchar>(i - 1, j - 1) +3* grayImg.at<uchar>(i - 1, j) +3* grayImg.at<uchar>(i - 1, j + 1)
                                                           -5*grayImg.at<uchar>(i, j - 1) +3*grayImg.at<uchar>(i, j + 1)
                                                           -5*grayImg.at<uchar>(i + 1, j - 1) +3*grayImg.at<uchar>(i + 1, j) +3*grayImg.at<uchar>(i + 1, j + 1)));
            f_8.at<uchar>(i, j) = saturate_cast<uchar>(fabs(-5*grayImg.at<uchar>(i - 1, j - 1) -5* grayImg.at<uchar>(i - 1, j) +3* grayImg.at<uchar>(i - 1, j + 1)
                                                           -5*grayImg.at<uchar>(i, j - 1) +3*grayImg.at<uchar>(i, j + 1)
                                                           +3*grayImg.at<uchar>(i + 1, j - 1) +3*grayImg.at<uchar>(i + 1, j) +3*grayImg.at<uchar>(i + 1, j + 1)));
            QVector<int> goal = {f_1.at<uchar>(i, j),f_2.at<uchar>(i, j),f_3.at<uchar>(i, j),f_4.at<uchar>(i, j),f_5.at<uchar>(i, j),f_6.at<uchar>(i, j),f_7.at<uchar>(i, j),f_8.at<uchar>(i, j)};
            int max_8 = 0;
            for(int i = 0 ; i<8;i++){
                if(goal[i] > max_8){
                    max_8 = goal[i];
                }
            }

            gradimg.at<uchar>(i, j) = max_8;
            gray2Img.at<uchar>(i, j) = saturate_cast<uchar>(grayImg.at<uchar>(i, j) - gradimg.at<uchar>(i, j));
        }
    }
    Qtemp = QImage((const uchar*)(gray2Img.data), gray2Img.cols, gray2Img.rows, gray2Img.cols*gray2Img.channels(), QImage::Format_Indexed8);
    ui->label_3->setPixmap(QPixmap::fromImage(Qtemp));
    Qtemp = Qtemp.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_3->setScaledContents(true);
    ui->label_3->resize(Qtemp.size());
    ui->label_3->show();

    Qtemp2 = QImage((const uchar*)(gradimg.data), gradimg.cols, gradimg.rows, gradimg.cols*gradimg.channels(), QImage::Format_Indexed8);
    ui->label_2->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_2->setScaledContents(true);
    ui->label_2->resize(Qtemp2.size());
    ui->label_2->show();
}

void MainWindow::on_Canny_clicked()
{
    Mat gauss,f_x,f_y,gradimg,max_control,gray2Img;
    QImage Qtemp,Qtemp2;
    gauss.create(grayImg.rows, grayImg.cols, CV_8UC1);
    gradimg.create(grayImg.rows, grayImg.cols, CV_8UC1);
    gray2Img.create(grayImg.rows, grayImg.cols, CV_8UC1);
    f_x.create(grayImg.rows, grayImg.cols, CV_8UC1);
    f_y.create(grayImg.rows, grayImg.cols, CV_8UC1);
    QVector<double> direction((grayImg.rows-1)*(grayImg.rows-1),0);
    //高斯处理
    for (int i = 0; i < grayImg.rows - 1; i++)
    {
        for (int j = 0; j < grayImg.cols - 1; j++){
            gauss.at<uchar>(i,j) = saturate_cast<uchar>(fabs((0.751136*grayImg.at<uchar>(i - 1, j - 1) +0.123841* grayImg.at<uchar>(i - 1, j) +0.0751136* grayImg.at<uchar>(i - 1, j + 1)
                                                         +0.123841*grayImg.at<uchar>(i, j - 1) +0.20418*grayImg.at<uchar>(i, j) + 0.123841*grayImg.at<uchar>(i, j + 1)
                                                         + 0.0751136*grayImg.at<uchar>(i + 1, j - 1) + 0.123841*grayImg.at<uchar>(i + 1, j) + 0.0751136*grayImg.at<uchar>(i + 1, j + 1))));
        }
    }
    //sobel处理
    int k = 0;
    for (int i = 1; i < gauss.rows - 1; i++)
    {
        for (int j = 1; j < gauss.cols - 1; j++)
        {
            f_x.at<uchar>(i, j) = saturate_cast<uchar>(fabs(grayImg.at<uchar>(i + 1, j - 1) + 2*grayImg.at<uchar>(i + 1, j) + grayImg.at<uchar>(i + 1, j + 1)
                                                            - grayImg.at<uchar>(i - 1, j - 1) - 2*grayImg.at<uchar>(i - 1, j) - grayImg.at<uchar>(i - 1, j + 1)));
            f_y.at<uchar>(i, j) = saturate_cast<uchar>(fabs(grayImg.at<uchar>(i - 1, j + 1) + 2*grayImg.at<uchar>(i, j + 1) + grayImg.at<uchar>(i + 1, j + 1)
                                                            - grayImg.at<uchar>(i - 1, j - 1) - 2*grayImg.at<uchar>(i, j - 1) - grayImg.at<uchar>(i + 1, j - 1)));
            gradimg.at<uchar>(i, j) = sqrt(pow(f_x.at<uchar>(i, j),2)+pow(f_y.at<uchar>(i, j),2));

            if(f_x.at<uchar>(i, j)==0)
            {
                direction[k]=atan(f_y.at<uchar>(i, j)/0.0000000001)*57.3;  //防止除数为0异常
            }
            else {
                direction[k]=atan(f_y.at<uchar>(i, j)/f_x.at<uchar>(i, j))*57.3;
            }
            direction[k]+=90;
            k++;
        }
    }
    //极大值抑制
//    double m,s;
//    Mat Mat_mean,Mat_vari;
//    meanStdDev(gradimg,Mat_mean,Mat_vari);
//    m = Mat_mean.at<double>(0,0);
//    s = Mat_vari.at<double>(0,0);
//    std::cout<<"m"<< m <<"    "<< "s" << s << std::endl;
//    m = m+s;
//    s = 0.4*m;
//    std::cout<<"m"<< m <<"    "<< "s" << s << std::endl;
    max_control=gradimg.clone();
    k = 0;
    for (int i = 1; i < gradimg.rows - 1; i++)
    {
        for (int j = 1; j < gradimg.cols - 1; j++){
            int value00=gradimg.at<uchar>((i-1),j-1);
            int value01=gradimg.at<uchar>((i-1),j);
            int value02=gradimg.at<uchar>((i-1),j+1);
            int value10=gradimg.at<uchar>((i),j-1);
            int value11=gradimg.at<uchar>((i),j);
            int value12=gradimg.at<uchar>((i),j+1);
            int value20=gradimg.at<uchar>((i+1),j-1);
            int value21=gradimg.at<uchar>((i+1),j);
            int value22=gradimg.at<uchar>((i+1),j+1);

            if(direction[k]>0&&direction[k]<=45)
            {
                if(value11<=(value12+(value02-value12)*tan(direction[i*max_control.rows+j]))||(value11<=(value10+(value20-value10)*tan(direction[i*max_control.rows+j]))))
                {
                    max_control.at<uchar>(i,j)=0;
                }
            }

            if(direction[k]>45&&direction[k]<=90)
            {
                if(value11<=(value01+(value02-value01)/tan(direction[i*max_control.cols+j]))||value11<=(value21+(value20-value21)/tan(direction[i*max_control.cols+j])))
                {
                    max_control.at<uchar>(i,j)=0;

                }
            }

            if(direction[k]>90&&direction[k]<=135)
            {
                if(value11<=(value01+(value00-value01)/tan(180-direction[i*max_control.cols+j]))||value11<=(value21+(value22-value21)/tan(180-direction[i*max_control.cols+j])))
                {
                    max_control.at<uchar>(i,j)=0;
                }
            }
            if(direction[k]>135&&direction[k]<=180)
            {
                if(value11<=(value10+(value00-value10)*tan(180-direction[i*max_control.cols+j]))||value11<=(value12+(value22-value11)*tan(180-direction[i*max_control.cols+j])))
                {
                    max_control.at<uchar>(i,j)=0;
                }
            }
            k++;
        }
    }
    DoubleThreshold(max_control,10,40);
    DoubleThresholdLink(max_control,10,40);

    for (int i = 0; i < grayImg.rows - 1; i++)
    {
        for (int j = 0; j < grayImg.cols - 1; j++){
            gray2Img.at<uchar>(i, j) = saturate_cast<uchar>(grayImg.at<uchar>(i, j) - max_control.at<uchar>(i, j));
        }
    }

    Qtemp2 = QImage((const uchar*)(max_control.data), max_control.cols, max_control.rows, max_control.cols*max_control.channels(), QImage::Format_Indexed8);
    ui->label_2->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_2->setScaledContents(true);
    ui->label_2->resize(Qtemp2.size());
    ui->label_2->show();

    Qtemp = QImage((const uchar*)(gray2Img.data), gray2Img.cols, gray2Img.rows, gray2Img.cols*gray2Img.channels(), QImage::Format_Indexed8);
    ui->label_3->setPixmap(QPixmap::fromImage(Qtemp));
    Qtemp = Qtemp.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_3->setScaledContents(true);
    ui->label_3->resize(Qtemp.size());
    ui->label_3->show();
}

void MainWindow::on_average_filter_clicked()
{
    Mat filterImg;
    QImage Qtemp,Qtemp2;

    filterImg = noiseImg.clone();

    for(int i = 1 ; i < noiseImg.rows - 1 ; i++)
        for(int j = 1 ; j < noiseImg.cols - 1 ; j++){
            for(int k = 0 ; k < 3 ; k++){
                filterImg.at<Vec3b>(i,j)[k] = saturate_cast<uchar>((noiseImg.at<Vec3b>(i - 1,j - 1)[k] + noiseImg.at<Vec3b>(i - 1,j)[k] + noiseImg.at<Vec3b>(i - 1,j + 1)[k]
                                                   +noiseImg.at<Vec3b>(i,j - 1)[k] + noiseImg.at<Vec3b>(i,j)[k] + noiseImg.at<Vec3b>(i,j + 1)[k]
                                                   +noiseImg.at<Vec3b>(i + 1,j - 1)[k] + noiseImg.at<Vec3b>(i + 1,j)[k] + noiseImg.at<Vec3b>(i + 1,j + 1)[k])/9);
            }
        }

    Qtemp2 = QImage((const unsigned char*)(filterImg.data), filterImg.cols, filterImg.rows, filterImg.step, QImage::Format_RGB888);
    ui->label_3->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_3->setScaledContents(true);
    ui->label_3->resize(Qtemp2.size());
    ui->label_3->show();
}

void MainWindow::on_middle_filter_clicked()
{
    Mat filterImg;
    QImage Qtemp,Qtemp2;
    QVector<double> middle(9,0);

    filterImg = noiseImg.clone();

    for(int i = 1 ; i < noiseImg.rows - 1 ; i++)
        for(int j = 1 ; j < noiseImg.cols - 1 ; j++){
            for(int k = 0 ; k < 3 ; k++){
                middle[0] = noiseImg.at<Vec3b>(i - 1 , j - 1)[k];
                middle[1] = noiseImg.at<Vec3b>(i - 1 , j)[k];
                middle[2] = noiseImg.at<Vec3b>(i - 1 , j + 1)[k];
                middle[3] = noiseImg.at<Vec3b>(i , j - 1)[k];
                middle[4] = noiseImg.at<Vec3b>(i , j)[k];
                middle[5] = noiseImg.at<Vec3b>(i , j + 1)[k];
                middle[6] = noiseImg.at<Vec3b>(i + 1 , j - 1)[k];
                middle[7] = noiseImg.at<Vec3b>(i + 1 , j)[k];
                middle[8] = noiseImg.at<Vec3b>(i + 1 , j + 1)[k];

                std::sort(middle.begin(),middle.end());

                filterImg.at<Vec3b>(i,j)[k] = saturate_cast<uchar>(middle[5]);
            }
        }

    Qtemp2 = QImage((const unsigned char*)(filterImg.data), filterImg.cols, filterImg.rows, filterImg.step, QImage::Format_RGB888);
    ui->label_3->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_3->setScaledContents(true);
    ui->label_3->resize(Qtemp2.size());
    ui->label_3->show();
}

void MainWindow::on_window_filter_clicked()
{
    Mat filterImg;
    QImage Qtemp,Qtemp2;
    QVector<double> window(8,0),minImg(8,0);

    filterImg = noiseImg.clone();

    for(int i = 1 ; i < noiseImg.rows - 1 ; i++)
        for(int j = 1 ; j < noiseImg.cols - 1 ; j++){
            for(int k = 0 ; k < 3 ; k++){
                window[0] = (noiseImg.at<Vec3b>(i - 1 , j - 1)[k] + noiseImg.at<Vec3b>(i - 1 , j)[k] + noiseImg.at<Vec3b>(i , j - 1)[k] + noiseImg.at<Vec3b>(i , j)[k])/4;
                window[1] = (noiseImg.at<Vec3b>(i - 1 , j)[k] + noiseImg.at<Vec3b>(i - 1 , j + 1)[k] + noiseImg.at<Vec3b>(i , j)[k] + noiseImg.at<Vec3b>(i , j + 1)[k])/4;
                window[2] = (noiseImg.at<Vec3b>(i , j)[k] + noiseImg.at<Vec3b>(i , j + 1)[k] + noiseImg.at<Vec3b>(i + 1 , j)[k] + noiseImg.at<Vec3b>(i + 1 , j + 1)[k])/4;
                window[3] = (noiseImg.at<Vec3b>(i , j - 1)[k] + noiseImg.at<Vec3b>(i , j)[k] + noiseImg.at<Vec3b>(i + 1 , j - 1)[k] + noiseImg.at<Vec3b>(i + 1 , j)[k])/4;
                window[4] = (noiseImg.at<Vec3b>(i - 1 , j - 1)[k] + noiseImg.at<Vec3b>(i - 1 , j)[k] + noiseImg.at<Vec3b>(i - 1 , j + 1)[k] + noiseImg.at<Vec3b>(i, j - 1)[k] + noiseImg.at<Vec3b>(i, j)[k] + noiseImg.at<Vec3b>(i, j + 1)[k])/6;
                window[5] = (noiseImg.at<Vec3b>(i + 1 , j - 1)[k] + noiseImg.at<Vec3b>(i + 1 , j)[k] + noiseImg.at<Vec3b>(i + 1 , j + 1)[k] + noiseImg.at<Vec3b>(i, j - 1)[k] + noiseImg.at<Vec3b>(i, j)[k] + noiseImg.at<Vec3b>(i, j + 1)[k])/6;
                window[6] = (noiseImg.at<Vec3b>(i - 1 , j)[k] + noiseImg.at<Vec3b>(i - 1 , j + 1)[k] + noiseImg.at<Vec3b>(i , j)[k] + noiseImg.at<Vec3b>(i, j + 1)[k] + noiseImg.at<Vec3b>(i + 1, j)[k] + noiseImg.at<Vec3b>(i + 1, j + 1)[k])/6;
                window[7] = (noiseImg.at<Vec3b>(i - 1 , j - 1)[k] + noiseImg.at<Vec3b>(i - 1 , j)[k] + noiseImg.at<Vec3b>(i , j)[k] + noiseImg.at<Vec3b>(i, j - 1)[k] + noiseImg.at<Vec3b>(i + 1, j)[k] + noiseImg.at<Vec3b>(i + 1, j - 1)[k])/6;

                for(int n = 0 ; n < 8 ; n++){
                    minImg[n] = pow(window[n] - noiseImg.at<Vec3b>(i , j)[k],2);
                }
                auto smallest = std::min_element(std::begin(minImg), std::end(minImg));
                int position = std::distance(std::begin(minImg), smallest);
                filterImg.at<Vec3b>(i , j)[k] = saturate_cast<uchar>(window[position]);
            }
        }

    Qtemp2 = QImage((const unsigned char*)(filterImg.data), filterImg.cols, filterImg.rows, filterImg.step, QImage::Format_RGB888);
    ui->label_3->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_3->setScaledContents(true);
    ui->label_3->resize(Qtemp2.size());
    ui->label_3->show();
}

void MainWindow::on_gauss_filter_clicked()
{
    Mat filterImg;
    QImage Qtemp,Qtemp2;

    filterImg = noiseImg.clone();

    for(int i = 1 ; i < noiseImg.rows - 1 ; i++)
        for(int j = 1 ; j < noiseImg.cols - 1 ; j++){
            for(int k = 0 ; k < 3 ; k++){
                filterImg.at<Vec3b>(i,j)[k] = saturate_cast<uchar>((noiseImg.at<Vec3b>(i - 1,j - 1)[k] + 2*noiseImg.at<Vec3b>(i - 1,j)[k] + noiseImg.at<Vec3b>(i - 1,j + 1)[k]
                                                   +2*noiseImg.at<Vec3b>(i,j - 1)[k] + 4*noiseImg.at<Vec3b>(i,j)[k] + 2*noiseImg.at<Vec3b>(i,j + 1)[k]
                                                   +noiseImg.at<Vec3b>(i + 1,j - 1)[k] + 2*noiseImg.at<Vec3b>(i + 1,j)[k] + noiseImg.at<Vec3b>(i + 1,j + 1)[k])/16);
            }
        }

    Qtemp2 = QImage((const unsigned char*)(filterImg.data), filterImg.cols, filterImg.rows, filterImg.step, QImage::Format_RGB888);
    ui->label_3->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_3->setScaledContents(true);
    ui->label_3->resize(Qtemp2.size());
    ui->label_3->show();
}

void MainWindow::on_form_filter_clicked()
{
    Mat filterImg,temp,RGB;
    QImage Qtemp,Qtemp2;

    Mat element=getStructuringElement(MORPH_RECT,Size(15,15));
    cvtColor(srcImg, RGB, CV_BGR2RGB);
    erode(RGB,temp,element);
    dilate(temp,filterImg,element);


    Qtemp2 = QImage((const unsigned char*)(filterImg.data), filterImg.cols, filterImg.rows, filterImg.step, QImage::Format_RGB888);
    ui->label_3->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_3->setScaledContents(true);
    ui->label_3->resize(Qtemp2.size());
    ui->label_3->show();

}

void MainWindow::on_affine_clicked()
{
    QImage Qtemp,Qtemp2;
    Point2f srcTri[3], dstTri[3];
    Mat rot_mat(2, 3, CV_32FC1);
    Mat warp_mat(2, 3, CV_32FC1);
    Mat dst,RGB;
    cvtColor(srcImg, RGB, CV_BGR2RGB);

    dst = Mat::zeros(RGB.rows, RGB.cols, RGB.type());

    srcTri[0] = Point2f (0,0);
    srcTri[1] = Point2f(RGB.cols - 1,0); //缩小一个像素
    srcTri[2]= Point2f(0,RGB.rows - 1);

    dstTri[0] = Point2f(RGB.cols * 0.0,RGB.rows * 0.33);
    dstTri[1] = Point2f(RGB.cols * 0.85,RGB.rows * 0.25);
    dstTri[2] = Point2f(RGB.cols* 0.15,RGB.rows* 0.7);

    warp_mat = getAffineTransform(srcTri, dstTri);

    warpAffine(RGB, dst, warp_mat,RGB.size());

    Qtemp2 = QImage((const unsigned char*)(dst.data), dst.cols, dst.rows, dst.step, QImage::Format_RGB888);
    ui->label_3->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_3->setScaledContents(true);
    ui->label_3->resize(Qtemp2.size());
    ui->label_3->show();
}

void MainWindow::on_perspective_clicked()
{
    QImage Qtemp,Qtemp2;
    Point2f srcQuad[4],dstQuad[4];
    Mat warp_matrix(3,3,CV_32FC1);
    Mat dst,RGB;
    cvtColor(srcImg, RGB, CV_BGR2RGB);
    dst = Mat::zeros(RGB.rows,RGB.cols,RGB.type());

    srcQuad[0]=Point2f(0,0); //src top left
    srcQuad[1] =Point2f(RGB.cols -1,0); //src top right
    srcQuad[2]=Point2f(0, RGB.rows-1); //src bottom left
    srcQuad[3]=Point2f(RGB.cols -1, RGB.rows-1); //src bot right

    dstQuad[0]=Point2f(RGB.cols*0.05,RGB.rows*0.33); //dst top left
    dstQuad[1]=Point2f(RGB.cols*0.9,RGB.rows*0.25); //dst top right
    dstQuad[2]=Point2f(RGB.cols*0.2,RGB.rows*0.7); //dst bottom left
    dstQuad[3]=Point2f(RGB.cols*0.8,RGB.rows*0.9); //dst bot right

    warp_matrix=getPerspectiveTransform(srcQuad,dstQuad);
    warpPerspective(RGB,dst,warp_matrix,RGB.size());

    Qtemp2 = QImage((const unsigned char*)(dst.data), dst.cols, dst.rows, dst.step, QImage::Format_RGB888);
    ui->label_3->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_3->setScaledContents(true);
    ui->label_3->resize(Qtemp2.size());
    ui->label_3->show();
}

void MainWindow::on_threshold_seg_clicked()
{
    QImage Qtemp;
    Mat targetImg;
    targetImg.create(grayImg.rows, grayImg.cols, CV_8UC1);

    for(int i = 0; i < grayImg.rows ; i++){
        for(int j = 0; j < grayImg.cols ; j++){
            if(grayImg.at<uchar>(i, j)>100){
                targetImg.at<uchar>(i,j) = 255;
            }
            else{targetImg.at<uchar>(i,j) = 0;}
        }
    }
    Qtemp = QImage((const uchar*)(targetImg.data), targetImg.cols, targetImg.rows, targetImg.cols*targetImg.channels(), QImage::Format_Indexed8);
    ui->label_2->setPixmap(QPixmap::fromImage(Qtemp));
    Qtemp = Qtemp.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_2->setScaledContents(true);
    ui->label_2->resize(Qtemp.size());
    ui->label_2->show();

}

void MainWindow::on_OSTU_clicked()
{
    QVector<int> hist(256,0);

    for(int i = 0 ; i < grayImg.rows ; i++)
        for(int j = 0 ; j < grayImg.cols ; j++){
            hist[grayImg.at<uchar>(i,j)]++;
        }
    int T;
    T = OSTU(hist);
    std::cout<<"OSTU:"<<T<<std::endl;

    QImage Qtemp;
    Mat targetImg;
    targetImg.create(grayImg.rows, grayImg.cols, CV_8UC1);

    for(int i = 0; i < grayImg.rows ; i++){
        for(int j = 0; j < grayImg.cols ; j++){
            if(grayImg.at<uchar>(i, j)>T){
                targetImg.at<uchar>(i,j) = 255;
            }
            else{targetImg.at<uchar>(i,j) = 0;}
        }
    }
    Qtemp = QImage((const uchar*)(targetImg.data), targetImg.cols, targetImg.rows, targetImg.cols*targetImg.channels(), QImage::Format_Indexed8);
    ui->label_2->setPixmap(QPixmap::fromImage(Qtemp));
    Qtemp = Qtemp.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_2->setScaledContents(true);
    ui->label_2->resize(Qtemp.size());
    ui->label_2->show();


}

void MainWindow::on_Kittler_clicked()
{
    QImage Qtemp;
    Mat targetImg,temp;
    temp = grayImg.clone();
    targetImg.create(grayImg.rows, grayImg.cols, CV_8UC1);

    int Grads,sumGrads = 0,sumGrayGrads = 0,KT;

    for (int i=1;i<temp.rows-1;i++)
    {
        uchar* previous=temp.ptr<uchar>(i-1); // previous row
        uchar* current=temp.ptr<uchar>(i); // current row
        uchar* next=temp.ptr<uchar>(i+1); // next row
        for(int j=1;j<temp.cols-1;j++)
        {   //求水平或垂直方向的最大梯度
            Grads=MAX(abs(previous[j]-next[j]),abs(current[j-1]-current[j+1]));
            sumGrads  += Grads;
            sumGrayGrads += Grads*(current[j]); //梯度与当前点灰度的积
        }
    }
    KT=sumGrayGrads/sumGrads;
    std::cout<<"OSTU:"<<KT<<std::endl;

    for(int i = 0; i < grayImg.rows ; i++){
        for(int j = 0; j < grayImg.cols ; j++){
            if(grayImg.at<uchar>(i, j)>KT){
                targetImg.at<uchar>(i,j) = 255;
            }
            else{targetImg.at<uchar>(i,j) = 0;}
        }
    }
    Qtemp = QImage((const uchar*)(targetImg.data), targetImg.cols, targetImg.rows, targetImg.cols*targetImg.channels(), QImage::Format_Indexed8);
    ui->label_2->setPixmap(QPixmap::fromImage(Qtemp));
    Qtemp = Qtemp.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_2->setScaledContents(true);
    ui->label_2->resize(Qtemp.size());
    ui->label_2->show();

}

void MainWindow::on_frame_diff_clicked()
{
    Mat pFrame1,pFrame2, pFrame3;  //当前帧

    VideoCapture pCapture;

    int nFrmNum;

    Mat pframe;
    pCapture = VideoCapture("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/cat.avi");
    pCapture >> pframe;

    Mat pFrImg1,pFrImg2,pFrImg3;   //当前帧

    pFrImg1.create(pframe.size(), CV_8UC1);
    pFrImg2.create(pframe.size(), CV_8UC1);
    pFrImg3.create(pframe.size(), CV_8UC1);

    Mat pFrMat1, pFrMat2, pFrMat3;

    nFrmNum = 0;
    while (1)
    {
        nFrmNum++;

            pCapture >> pFrame1;
            if (pFrame1.data == NULL)
                return;
            pCapture >> pFrame2;
            pCapture >> pFrame3;

            cvtColor(pFrame1, pFrImg1, CV_BGR2GRAY);
            cvtColor(pFrame2, pFrImg2, CV_BGR2GRAY);
            cvtColor(pFrame3, pFrImg3, CV_BGR2GRAY);

            absdiff(pFrImg1, pFrImg2, pFrMat1);//当前帧跟前面帧相减
            absdiff(pFrImg2, pFrImg3, pFrMat2);//当前帧与后面帧相减
                                                          //二值化前景图
            threshold(pFrMat1, pFrMat1, 10, 255.0, CV_THRESH_BINARY);
            threshold(pFrMat2, pFrMat2, 10, 255.0, CV_THRESH_BINARY);

            Mat element = getStructuringElement(0, cv::Size(3, 3));
            Mat element1 = getStructuringElement(0, cv::Size(5, 5));
            //膨胀化前景图
            erode(pFrMat1, pFrMat1, element);
            erode(pFrMat2, pFrMat2, element);

            dilate(pFrMat1, pFrMat1, element1);
            dilate(pFrMat2, pFrMat2, element1);

            dilate(pFrMat1, pFrMat1, element1);
            dilate(pFrMat2, pFrMat2, element1);

            //imshow("diff1", pFrMat1);
            imshow("diff2", pFrMat2);

            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            //当前帧与前面帧相减后提取的轮廓线
            findContours(pFrMat2, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
            double Maxarea = 0;
            int numi=0;
            for (size_t i = 0; i < contours.size(); ++i)
            {
                double area = contourArea(contours[i], false);
                if (area > Maxarea)
                {
                    Maxarea = area;
                    numi = i;
                }
            }
            if (numi!=0)
                drawContours(pFrame2, contours, numi, Scalar(0, 0, 255), 2);

            Mat resultImage = Mat::zeros(pFrMat2.size(), CV_8U);

            imshow("src", pFrame2);
            //waitKey(10);
            if (waitKey(1)!=-1)
                    break;
    }
    pCapture.release();

    // Closes all the frames
    destroyAllWindows();
}

void MainWindow::on_mix_guass_clicked()
{
    Mat greyimg;
    Mat foreground, foreground2;
    Ptr<BackgroundSubtractorKNN> ptrKNN = createBackgroundSubtractorKNN(100, 400, true);
    Ptr<BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2(100, 25, true);
    namedWindow("Extracted Foreground");
    VideoCapture pCapture;
    Mat pframe;
    pCapture = VideoCapture("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/pets2001.avi");

    while (1)
    {
        pCapture >> pframe;
        if (pframe.data == NULL)
            return;
        cvtColor(pframe, greyimg, CV_BGR2GRAY);
        long long t = getTickCount();
        ptrKNN->apply(pframe, foreground, 0.01);
        long long t1 = getTickCount();
        mog2->apply(greyimg, foreground2, -1);
        long long t2 = getTickCount();
        //_cprintf("t1 = %f t2 = %f\n", (t1 - t) / getTickFrequency(), (t2 - t1) / getTickFrequency());
        cout<<"t1 = "<<(t1 - t) / getTickFrequency()<<"t2 = "<<(t2 - t1) / getTickFrequency()<<endl;
        imshow("Extracted Foreground", foreground);
        imshow("Extracted Foreground2", foreground2);
        imshow("video", pframe);
        if (waitKey(1)!=-1)
                break;
    }
    destroyAllWindows();
}

void MainWindow::on_circle_lbp_clicked()
{
    QImage Qtemp0,Qtemp,Qtemp1,Qtemp2;

    Mat Img = imread("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/lena.jpg");

    Mat temp;
    cvtColor(Img, temp, CV_BGR2RGB);//BGR convert to RGB
    Qtemp = QImage((const unsigned char*)(temp.data), temp.cols, temp.rows, temp.step, QImage::Format_RGB888);

    ui->label->setPixmap(QPixmap::fromImage(Qtemp));
    Qtemp = Qtemp.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label->setScaledContents(true);
    ui->label->resize(Qtemp.size());
    ui->label->show();

    Mat img = cv::imread("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/lena.jpg", 0);
    //namedWindow("image");
    //imshow("image", img);

    Qtemp = QImage((const uchar*)(img.data), img.cols, img.rows, img.cols*img.channels(), QImage::Format_Indexed8);
    ui->label_1->setPixmap(QPixmap::fromImage(Qtemp));
    Qtemp = Qtemp.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_1->setScaledContents(true);
    ui->label_1->resize(Qtemp.size());
    ui->label_1->show();

    int radius, neighbors;
    radius = 1;
    neighbors = 8;

    //创建一个LBP
    //注意为了溢出，我们行列都在原有图像上减去2个半径
    Mat dst = Mat(img.rows - 2 * radius, img.cols - 2 * radius, CV_8UC1, Scalar(0));
    elbp1(img, dst);
    //namedWindow("normal");
    //imshow("normal", dst);
    Qtemp1 = QImage((const uchar*)(dst.data), dst.cols, dst.rows, dst.cols*dst.channels(), QImage::Format_Indexed8);
    ui->label_2->setPixmap(QPixmap::fromImage(Qtemp1));
    Qtemp1 = Qtemp1.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_2->setScaledContents(true);
    ui->label_2->resize(Qtemp1.size());
    ui->label_2->show();

    Mat dst1 = Mat(img.rows - 2 * radius, img.cols - 2 * radius, CV_8UC1, Scalar(0));
    elbp(img, dst1, 1, 8);
    //namedWindow("circle");
    //imshow("circle", dst1);
    Qtemp2 = QImage((const uchar*)(dst1.data), dst1.cols, dst1.rows, dst1.cols*dst1.channels(), QImage::Format_Indexed8);
    ui->label_3->setPixmap(QPixmap::fromImage(Qtemp2));
    Qtemp2 = Qtemp2.scaled(250, 250, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->label_3->setScaledContents(true);
    ui->label_3->resize(Qtemp2.size());
    ui->label_3->show();

}

void MainWindow::on_target_det_clicked()
{
    QImage Qtemp,Qtemp1;

    Mat temp0 = imread("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/lena.jpg");
    Mat temp1 = imread("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/lena-1.jpg");
    Mat Img0,Img1,Img2;

    cvtColor(temp0, Img0, COLOR_BGR2HSV);
    cvtColor(temp1, Img1, COLOR_BGR2HSV);
    Mat box = Img1.clone();

    int h_bins = 50;
    int s_bins = 60;
    int histSize[] = { h_bins,s_bins };
    float h_ranges[] = { 0,180 };
    float s_ranges[] = { 0,256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0,1 };

    double max = 0.;
    int x_ray,y_ray;


    for(int i = 0; i < Img0.rows-Img1.rows-1; i++){
        for(int j = 0; j < Img0.cols-Img1.cols-1; j++){
            for(int x = i; x < Img1.rows + i; x++){
                for(int y = j; y < Img1.cols + j; y++){
                    box.at<Vec3b>(x-i, y-j) = Img0.at<Vec3b>(x, y);
                }
            }
            MatND hist_src0;
            MatND hist_src1;

            calcHist(&box, 1, channels, Mat(), hist_src0, 2, histSize, ranges, true, false);
            normalize(hist_src0, hist_src0, 0, 1, NORM_MINMAX, -1, Mat());

            calcHist(&Img1, 1, channels, Mat(), hist_src1, 2, histSize, ranges, true, false);
            normalize(hist_src1, hist_src1, 0, 1, NORM_MINMAX, -1, Mat());

            double src_src = compareHist(hist_src0, hist_src1, CV_COMP_CORREL);

            cout << "src compare with src correlation value : " << src_src << endl;

            if(src_src > max){
                max = src_src;
                x_ray = i;
                y_ray = j;
            }
        }
    }
    cout << "************************************"  << endl;

    Rect rect(x_ray, y_ray, Img1.rows, Img1.cols);
    rectangle(Img0, rect, Scalar(255, 0, 0),1, LINE_8,0);
    imshow("check",Img0);
    imshow("Img1",Img1);

    waitKey(0);
    cv::destroyWindow("check");
    cv::destroyWindow("Img1");
    waitKey(1);
}

void MainWindow::on_model_check_clicked()
{
    QImage Qtemp,Qtemp1;
    double minVal; double maxVal; Point minLoc; Point maxLoc;

    Mat Img0 = imread("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/lena.jpg");
    Mat Img1 = imread("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/lena-1.jpg");

    Mat result;

    matchTemplate(Img0, Img1, result, TM_SQDIFF);

    normalize(result, result, 1, 0, NORM_MINMAX);
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    rectangle(Img0,Rect(minLoc.x,minLoc.y,Img1.cols,Img1.rows),1,8,0 );

    imshow("src", Img0);
    imshow("template", Img1);
    imshow("0", result);
    waitKey(0);
    cv::destroyWindow("src");
    cv::destroyWindow("template");
    cv::destroyWindow("0");
    waitKey(1);

}

void MainWindow::on_cloaking_clicked()
{
    // TODO: 在此添加控件通知处理程序代码
    // Create a VideoCapture object and open the input file
    VideoCapture cap;
    cap.open("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/input-color.mp4");

    // Check if camera opened successfully
    if (!cap.isOpened()) {
        cout << "Error opening video stream or file" << endl;
        return;
    }

    Mat background, background1;
    for (int i = 0; i<60; i++)
    {
        cap >> background;
    }

    //flip(background,background,1);

    for (int i = 0; i<100; i++)
    {
        cap >> background1;
    }
    while (1)
    {
        long t = getTickCount();

        Mat frame;
        // Capture frame-by-frame
        cap >> frame;


        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        Mat hsv;
        //flip(frame,frame,1);
        cvtColor(frame, hsv, COLOR_BGR2HSV);

        Mat mask1, mask2;
        inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), mask1);   //H为0-10的分量
        inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);//H为170-180的分量


        mask1 = mask1 + mask2;

        Mat kernel = Mat::ones(3, 3, CV_32F);
        morphologyEx(mask1, mask1, MORPH_OPEN, kernel);//开
        morphologyEx(mask1, mask1, MORPH_DILATE, kernel);//膨胀

        bitwise_not(mask1, mask2);

        Mat res1, res2, final_output;
        bitwise_and(frame, frame, res1, mask2);//not red
        //imshow("res1 !!!", res1);
        bitwise_and(background, background, res2, mask1);//red

    //	long t1 = getTickCount();
        //imshow("res2 !!!",res2);
        //addWeighted(res1, 1, res2, 1, 0, final_output);
        add(res1, res2, final_output);

        imshow("Magic !!!", final_output);
        // Display the resulting frame
        //imshow( "Frame", frame );

        // Press  ESC on keyboard to exit
        char c = (char)waitKey(5);
        if (c == 27)
            break;
        // Also relese all the mat created in the code to avoid memory leakage.
        frame.release(), hsv.release(), mask1.release(), mask2.release(), res1.release(), res2.release(), final_output.release();

        long t1 = getTickCount();

        cout << "t1 =  " << (t1 - t) / getTickFrequency()*1000 << "ms\n"<<endl;

        if (waitKey(1)!=-1)
                break;
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();
}

void MainWindow::on_SIFT_clicked()
{
    Mat src1 = imread("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/1.1.jpg", 1);
    Mat src2 = imread("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/1.2.jpg", 1);
    imshow("src1", src1);
    imshow("src2", src2);

    if (!src1.data || !src2.data)
    {
        //_cprintf(" --(!) Error reading images \n");
        return;
    }

    //sift feature detect
    Ptr<SIFT> siftdetector = SIFT::create();
    vector<KeyPoint> kp1, kp2;

    siftdetector->detect(src1, kp1);
    siftdetector->detect(src2, kp2);
    Mat des1, des2;//descriptor
    siftdetector->compute(src1, kp1, des1);
    siftdetector->compute(src2, kp2, des2);
    Mat res1, res2;

    drawKeypoints(src1, kp1, res1);//在内存中画出特征点
    drawKeypoints(src2, kp2, res2);

    //_cprintf("size of description of Img1: %d\n",kp1.size());
    //_cprintf("size of description of Img2: %d\n",kp2.size());

    Mat transimg1, transimg2;
    transimg1 = res1.clone();
    transimg2 = res2.clone();

    char str1[20], str2[20];
    sprintf(str1, "%d", kp1.size());
    sprintf(str2, "%d", kp2.size());

    const char* str = str1;
    putText(transimg1, str1, Point(280, 230), 0, 1.0,Scalar(255, 0, 0),2);//在图片中输出字符

    str = str2;
    putText(transimg2, str2, Point(280, 230), 0, 1.0,Scalar(255, 0, 0),2);//在图片中输出字符

                                                                            //imshow("Description 1",res1);
    imshow("descriptor1", transimg1);
    imshow("descriptor2", transimg2);

    BFMatcher matcher(NORM_L2, true);
    vector<DMatch> matches;
    matcher.match(des1, des2, matches);
    Mat img_match;
    drawMatches(src1, kp1, src2, kp2, matches, img_match);//,Scalar::all(-1),Scalar::all(-1),vector<char>(),drawmode);
    //_cprintf("number of matched points: %d\n",matches.size());
    imshow("matches", img_match);
    waitKey(0);
    cv::destroyWindow("matches");
    cv::destroyWindow("descriptor1");
    cv::destroyWindow("descriptor2");
    cv::destroyWindow("src1");
    cv::destroyWindow("src2");
    waitKey(1);
}

void MainWindow::on_orb_clicked()
{
    Mat obj = imread("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/1.1.jpg");   //载入目标图像
    Mat scene = imread("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/1.2.jpg"); //载入场景图像
    if (obj.empty() || scene.empty())
    {
        cout << "Can't open the picture!\n";
        return;
    }
    vector<KeyPoint> obj_keypoints, scene_keypoints;
    Mat obj_descriptors, scene_descriptors;
    Ptr<ORB> detector = ORB::create();

    detector->detect(obj, obj_keypoints);
    detector->detect(scene, scene_keypoints);
    detector->compute(obj, obj_keypoints, obj_descriptors);
    detector->compute(scene, scene_keypoints, scene_descriptors);

    BFMatcher matcher(NORM_HAMMING, true); //汉明距离做为相似度度量
    vector<DMatch> matches;
    matcher.match(obj_descriptors, scene_descriptors, matches);
    Mat match_img;
    drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, match_img);
    imshow("match_img", match_img);

    //保存匹配对序号
    vector<int> queryIdxs(matches.size()), trainIdxs(matches.size());
    for (size_t i = 0; i < matches.size(); i++)
    {
        queryIdxs[i] = matches[i].queryIdx;
        trainIdxs[i] = matches[i].trainIdx;
    }

    Mat H12;   //变换矩阵

    vector<Point2f> points1;
    KeyPoint::convert(obj_keypoints, points1, queryIdxs);
    vector<Point2f> points2;
    KeyPoint::convert(scene_keypoints, points2, trainIdxs);
    int ransacReprojThreshold = 5;  //拒绝阈值


    H12 = findHomography(Mat(points1), Mat(points2), RANSAC, ransacReprojThreshold);
    vector<char> matchesMask(matches.size(), 0);
    Mat points1t;
    perspectiveTransform(Mat(points1), points1t, H12);
    for (size_t i1 = 0; i1 < points1.size(); i1++)  //保存‘内点’
    {
        if (norm(points2[i1] - points1t.at<Point2f>((int)i1, 0)) <= ransacReprojThreshold) //给内点做标记
        {
            matchesMask[i1] = 1;
        }
    }
    Mat match_img2;   //滤除‘外点’后
    drawMatches(obj, obj_keypoints, scene, scene_keypoints, matches, match_img2, Scalar(0, 0, 255), Scalar::all(-1), matchesMask);

    //画出目标位置
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point(0, 0); obj_corners[1] = Point(obj.cols, 0);
    obj_corners[2] = Point(obj.cols, obj.rows); obj_corners[3] = Point(0, obj.rows);
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform(obj_corners, scene_corners, H12);
    //line( match_img2, scene_corners[0] + Point2f(static_cast<float>(obj.cols), 0),scene_corners[1] + Point2f(static_cast<float>(obj.cols), 0),Scalar(0,0,255),2);
    //line( match_img2, scene_corners[1] + Point2f(static_cast<float>(obj.cols), 0),scene_corners[2] + Point2f(static_cast<float>(obj.cols), 0),Scalar(0,0,255),2);
    //line( match_img2, scene_corners[2] + Point2f(static_cast<float>(obj.cols), 0),scene_corners[3] + Point2f(static_cast<float>(obj.cols), 0),Scalar(0,0,255),2);
    //line( match_img2, scene_corners[3] + Point2f(static_cast<float>(obj.cols), 0),scene_corners[0] + Point2f(static_cast<float>(obj.cols), 0),Scalar(0,0,255),2);
    line(match_img2, Point2f((scene_corners[0].x + static_cast<float>(obj.cols)), (scene_corners[0].y)), Point2f((scene_corners[1].x + static_cast<float>(obj.cols)), (scene_corners[1].y)), Scalar(0, 0, 255), 2);
    line(match_img2, Point2f((scene_corners[1].x + static_cast<float>(obj.cols)), (scene_corners[1].y)), Point2f((scene_corners[2].x + static_cast<float>(obj.cols)), (scene_corners[2].y)), Scalar(0, 0, 255), 2);
    line(match_img2, Point2f((scene_corners[2].x + static_cast<float>(obj.cols)), (scene_corners[2].y)), Point2f((scene_corners[3].x + static_cast<float>(obj.cols)), (scene_corners[3].y)), Scalar(0, 0, 255), 2);
    line(match_img2, Point2f((scene_corners[3].x + static_cast<float>(obj.cols)), (scene_corners[3].y)), Point2f((scene_corners[0].x + static_cast<float>(obj.cols)), (scene_corners[0].y)), Scalar(0, 0, 255), 2);

    float A_th;
    A_th = atan(abs((scene_corners[3].y - scene_corners[0].y) / (scene_corners[3].x - scene_corners[0].x)));
    A_th = 90-180 * A_th / 3.14;

    imshow("match_img2", match_img2);

    //line( scene, scene_corners[0],scene_corners[1],Scalar(0,0,255),2);
    //line( scene, scene_corners[1],scene_corners[2],Scalar(0,0,255),2);
    //line( scene, scene_corners[2],scene_corners[3],Scalar(0,0,255),2);
    //line( scene, scene_corners[3],scene_corners[0],Scalar(0,0,255),2);

    imshow("scense",scene);

    Mat rotimage;
    Mat rotate = getRotationMatrix2D(Point(scene.cols/2,scene.rows/2), A_th, 1);
    warpAffine(scene,rotimage,rotate, scene.size());
    imshow("rotimage", rotimage);


    //方法三 透视变换
    Point2f src_point[4];
    Point2f dst_point[4];
    src_point[0].x = scene_corners[0].x;
    src_point[0].y = scene_corners[0].y;
    src_point[1].x = scene_corners[1].x;
    src_point[1].y = scene_corners[1].y;
    src_point[2].x = scene_corners[2].x;
    src_point[2].y = scene_corners[2].y;
    src_point[3].x = scene_corners[3].x;
    src_point[3].y = scene_corners[3].y;


    dst_point[0].x = 0;
    dst_point[0].y = 0;
    dst_point[1].x = obj.cols;
    dst_point[1].y = 0;
    dst_point[2].x = obj.cols;
    dst_point[2].y = obj.rows;
    dst_point[3].x = 0;
    dst_point[3].y = obj.rows;

    Mat newM(3, 3, CV_32FC1);
    newM=getPerspectiveTransform(src_point, dst_point);

    Mat dst = scene.clone();

    warpPerspective(scene, dst, newM, obj.size());

    imshow("obj", obj);
    imshow("dst", dst);

    Mat resultimg=dst.clone();

    absdiff(obj, dst, resultimg);//当前帧跟前面帧相减

    imshow("result", resultimg);

    imshow("dst", dst);
    imshow("src", obj);

    waitKey(0);
    cv::destroyWindow("match_img");
    cv::destroyWindow("match_img2");
    cv::destroyWindow("obj");
    cv::destroyWindow("result");
    cv::destroyWindow("dst");
    cv::destroyWindow("src");
    cv::destroyWindow("scense");
    cv::destroyWindow("rotimage");
    waitKey(1);
}

//输入图像
Mat img_color;
//灰度值归一化
Mat bgr_color;
//HSV图像
Mat hsv_color;
//色度
int hsv_hmin = 0;
int hsv_hmin_Max = 360;
int hsv_hmax = 360;
int hsv_hmax_Max = 360;
//饱和度
int hsv_smin = 0;
int hsv_smin_Max = 255;
int hsv_smax = 255;
int hsv_smax_Max = 255;
//亮度
int hsv_vmin = 106;
int hsv_vmin_Max = 255;
int hsv_vmax = 250;
int hsv_vmax_Max = 255;
//显示原图的窗口
string windowName = "src";
//输出图像的显示窗口
string dstName = "dst";
//输出图像
Mat dst_color;
//回调函数
void callBack(int, void*)
{
    //输出图像分配内存
    dst_color = Mat::zeros(img_color.size(), CV_32FC3);
    //掩码
    Mat mask;
    inRange(hsv_color, Scalar(hsv_hmin, hsv_smin / float(hsv_smin_Max), hsv_vmin / float(hsv_vmin_Max)), Scalar(hsv_hmax, hsv_smax / float(hsv_smax_Max), hsv_vmax / float(hsv_vmax_Max)), mask);
    //只保留
    for (int r = 0; r < bgr_color.rows; r++)
    {
        for (int c = 0; c < bgr_color.cols; c++)
        {
            if (mask.at<uchar>(r, c) == 255)
            {
                dst_color.at<Vec3f>(r, c) = bgr_color.at<Vec3f>(r, c);
            }
        }
    }
    //输出图像
    imshow(dstName, dst_color);
    //imshow("mast", mask);
    //保存图像
    //dst_color.convertTo(dst_color, CV_8UC3, 255.0, 0);
    //imwrite("F://program//image//HSV_inRange.jpg", dst_color)

}


void MainWindow::on_color_fit_clicked()
{
    img_color = imread("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/color.jpg");
    if (!img_color.data || img_color.channels() != 3)
        return;
    namedWindow(windowName, CV_WINDOW_AUTOSIZE);
    imshow(windowName, img_color);
    //彩色图像的灰度值归一化
    img_color.convertTo(bgr_color, CV_32FC3, 1.0 / 255, 0);
    //颜色空间转换
    cvtColor(bgr_color, hsv_color, COLOR_BGR2HSV);
    //定义输出图像的显示窗口
    namedWindow(dstName, WINDOW_GUI_EXPANDED);
    //调节色度 H
    createTrackbar("hmin", dstName, &hsv_hmin, hsv_hmin_Max, callBack);
    createTrackbar("hmax", dstName, &hsv_hmax, hsv_hmax_Max, callBack);
    //调节饱和度 S
    createTrackbar("smin", dstName, &hsv_smin, hsv_smin_Max, callBack);
    createTrackbar("smax", dstName, &hsv_smax, hsv_smax_Max, callBack);
    //调节亮度 V
    createTrackbar("vmin", dstName, &hsv_vmin, hsv_vmin_Max, callBack);
    createTrackbar("vmax", dstName, &hsv_vmax, hsv_vmax_Max, callBack);
    callBack(0, 0);
    waitKey(0);
    cv::destroyWindow(dstName);
    cv::destroyWindow(windowName);
    waitKey(1);
}

void MainWindow::on_svm_test_clicked()
{
    int iWidth = 512, iheight = 512;
//    Mat matImg = Mat::zeros(iheight, iWidth, CV_8UC3);//三色通道
    Mat matImg = Mat(iheight, iWidth, CV_8UC3, Scalar(0, 255, 255));//三色通道
                                                              //1.获取样本
    double labels[5] = { 1.0, -1.0, -1.0, -1.0,1.0 }; //样本数据
    Mat labelsMat(5, 1, CV_32SC1, labels);     //样本标签
    float trainingData[5][2] = { { 501, 300 },{ 255, 10 },{ 501, 255 },{ 10, 501 },{ 450,500 } }; //Mat结构特征数据
    Mat trainingDataMat(5, 2, CV_32FC1, trainingData);   //Mat结构标签
                                                         //2.设置SVM参数
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);//可以处理非线性分割的问题
    svm->setKernel(ml::SVM::POLY);//径向基函数SVM::LINEAR
                                        /*svm->setGamma(0.01);
                                        svm->setC(10.0);*/
                                        //算法终止条件
    svm->setDegree(1.0);
    svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 100, 1e-6));
    //3.训练支持向量
    svm->train(trainingDataMat, ml::SampleTypes::ROW_SAMPLE, labelsMat);
    //4.保存训练器
    svm->save("mnist_svm.xml");
    //5.导入训练器
    //Ptr<SVM> svm1 = StatModel::load<SVM>("mnist_dataset/mnist_svm.xml");

    //读取测试数据
    Mat sampleMat;
    Vec3b green(0, 255, 0), blue(255, 0, 0);
    for (int i = 0; i < matImg.rows; i++)
    {
        for (int j = 0; j < matImg.cols; j++)
        {
            sampleMat = (Mat_<float>(1, 2) << j, i);
            float fRespone = svm->predict(sampleMat);
            if (fRespone == 1)
            {
                matImg.at<cv::Vec3b>(i, j) = green;
            }
            else if (fRespone == -1)
            {
                matImg.at<cv::Vec3b>(i, j) = blue;
            }
        }
    }

    for (int i = 0; i < matImg.rows; i++)
    {
        for (int j = 0; j < matImg.cols; j++)
        {
            if(i>525-(1./2.)*j){
                matImg.at<cv::Vec3b>(i, j) = green;
            }
            else{matImg.at<cv::Vec3b>(i, j) = blue;}
        }
    }
    // Show the training data
    int thickness = -1;
    int lineType = 8;
    for (int i = 0; i < trainingDataMat.rows; i++)
    {
        if (labels[i] == 1)
        {
            circle(matImg, Point(trainingData[i][0], trainingData[i][1]), 5, Scalar(0, 0, 0), thickness, lineType);
        }
        else
        {
            circle(matImg, Point(trainingData[i][0], trainingData[i][1]), 5, Scalar(255, 255, 255), thickness, lineType);
        }
    }

    //显示支持向量点
    thickness = 2;
    lineType = 8;
    Mat vec = svm->getSupportVectors();
    int nVarCount = svm->getVarCount();//支持向量的维数
    //_cprintf("vec.rows=%d vec.cols=%d\n", vec.rows, vec.cols);
    for (int i = 0; i < vec.rows; ++i)
    {
        int x = (int)vec.at<float>(i, 0);
        int y = (int)vec.at<float>(i, 1);
        //_cprintf("vec.at=%d %f,%f\n", i,vec.at<float>(i, 0), vec.at<float>(i, 1));
        //_cprintf("x=%d,y=%d\n", x, y);
        circle(matImg, Point(x, y), 6, Scalar(0, 0, 255), thickness, lineType);
    }


    imshow("circle", matImg); // show it to the user
    waitKey(0);
    cv::destroyWindow("circle");
    waitKey(1);
}

void MainWindow::on_word_test_clicked()
{
    Ptr<ml::SVM> svm1 = ml::SVM::load("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/sample/SVM_HOG.xml");

    if (svm1->empty())
    {
        cout<< "load svm detector failed!!!\n"<< endl;
        return;
    }

    Mat testimg;
    testimg = imread("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/sample/9/0.png");
    cv::resize(testimg, testimg, Size(28, 28), 1);
    imshow("src", testimg);
    //waitKey(0);

    HOGDescriptor hog(Size(14, 14), Size(7, 7), Size(1, 1), Size(7, 7), 9);
    vector<float> imgdescriptor;
    hog.compute(testimg, imgdescriptor, Size(5, 5));
    Mat sampleMat;
    sampleMat.create(1, imgdescriptor.size(), CV_32FC1);

    for (int i = 0; i < imgdescriptor.size(); i++)
    {
        sampleMat.at<float>(0, i) = imgdescriptor[i];//第num个样本的特征向量中的第i个元素
    }
    int ret = svm1->predict(sampleMat);
    cout << "ret= " << ret <<endl;

    waitKey(0);
    cv::destroyWindow("src");
    waitKey(1);
}


double compute_sum_of_rect(Mat src,Rect r){
    int x=r.x;
    int y=r.y;
    int width=r.width;
    int height=r.height;
    double sum;
//这里使用Mat::at函数需要注意第一参数为行数对应的y和高度height，第二个参数对应才是列数对应的x和宽度width
    sum=src.at<double>(y,x)+src.at<double>(y+height,x+width)
            -src.at<double>(y+height,x)-src.at<double>(y,x+width);

    return sum;
}

void MainWindow::on_Haar_1_clicked()
{
    Mat src_img;
    src_img = imread("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/lena.jpg");
    if(src_img.empty()){
        cout<<"error.could not find"<<endl;
        return;
    }
    namedWindow("src_img",CV_WINDOW_AUTOSIZE);
    imshow("src_img",src_img);
    Mat gray_img;
    cvtColor(src_img,gray_img,COLOR_BGR2GRAY);
    namedWindow("gray_img",CV_WINDOW_AUTOSIZE);
    imshow("gray_img",gray_img);
    Mat sum_img = Mat::zeros(gray_img.rows + 1, gray_img.cols + 1,CV_32FC1);
    //Mat sqsum_img = Mat::zeros(gray_img.rows + 1, gray_img.cols + 1,CV_64FC1);
    integral(grayImg,sum_img,CV_64F);

    int step_x = 8, step_y = 8;
    double sum;
    Rect rect1, rect2;
    Mat dst = Mat::zeros(src_img.size(),CV_32FC1);
    for(int i = 0; i < src_img.rows; i = i + step_x){
        for(int j = 0; j < src_img.cols; j = j + step_y){
            rect1 = Rect(j, i, 2, 4);
            rect2 = Rect(j + 2, i, 2, 4);
            sum = compute_sum_of_rect(gray_img,rect1) - compute_sum_of_rect(gray_img,rect2);
            dst.at<float>(i,j) = sum;
        }
    }

    Mat dst_8;
    convertScaleAbs(dst,dst_8);
    imshow("dst",dst_8);

    waitKey(0);
    cv::destroyWindow("src_img");
    cv::destroyWindow("gray_img");
    cv::destroyWindow("dst");
    waitKey(1);
}

void MainWindow::on_Haar_2_clicked()
{
    Mat src_img;
    src_img = imread("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/lena.jpg");
    if(src_img.empty()){
        cout<<"error.could not find"<<endl;
        return;
    }
    namedWindow("src_img",CV_WINDOW_AUTOSIZE);
    imshow("src_img",src_img);
    Mat gray_img;
    cvtColor(src_img,gray_img,COLOR_BGR2GRAY);
    namedWindow("gray_img",CV_WINDOW_AUTOSIZE);
    imshow("gray_img",gray_img);
    Mat sum_img = Mat::zeros(gray_img.rows + 1, gray_img.cols + 1,CV_32FC1);
    //Mat sqsum_img = Mat::zeros(gray_img.rows + 1, gray_img.cols + 1,CV_64FC1);
    integral(grayImg,sum_img,CV_64F);

    int step_x = 8, step_y = 8;
    double sum;
    Rect rect1, rect2;
    Mat dst = Mat::zeros(src_img.size(),CV_32FC1);
    for(int i = 0; i < src_img.rows; i = i + step_x){
        for(int j = 0; j < src_img.cols; j = j + step_y){
            rect1 = Rect(j, i, 2, 4);
            rect2 = Rect(j, i + 2, 2, 4);
            sum = compute_sum_of_rect(gray_img,rect1) - compute_sum_of_rect(gray_img,rect2);
            dst.at<float>(i,j) = sum;
        }
    }

    Mat dst_8;
    convertScaleAbs(dst,dst_8);
    imshow("dst",dst_8);

    waitKey(0);
    cv::destroyWindow("src_img");
    cv::destroyWindow("gray_img");
    cv::destroyWindow("dst");
    waitKey(1);
}

void MainWindow::on_gaber_clicked()
{
    Mat src = imread("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/lena.jpg", IMREAD_GRAYSCALE);
    namedWindow("input", CV_WINDOW_AUTOSIZE);
    imshow("input", src);
    Mat src_f;
    src.convertTo(src_f, CV_32F);
    // 参数初始化
    int kernel_size = 3;
    double sigma = 1.0, lambd = CV_PI / 8, gamma = 0.5, psi = 0;
    vector<Mat> destArray;
    double theta[4];
    Mat temp;
    // theta 法线方向
    theta[0] = 0;
    theta[1] = CV_PI / 4;
    theta[2] = CV_PI / 2;
    theta[3] = CV_PI - CV_PI / 4;
    // gabor 纹理检测器，可以更多，
    // filters = number of thetas * number of lambd
    // 这里lambad只取一个值，所有4个filter
    for (int i = 0; i<4; i++)
    {
        Mat kernel1;
        Mat dest;
        kernel1 = getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta[i], lambd, gamma, psi, CV_32F);
        filter2D(src_f, dest, CV_32F, kernel1);
        destArray.push_back(dest);
    }
    // 显示与保存
    Mat dst1, dst2, dst3, dst4;
    convertScaleAbs(destArray[0], dst1);
    //imwrite("F://program//image//gabor1.jpg", dst1);
    convertScaleAbs(destArray[1], dst2);
    //imwrite("F://program//image//gabor2.jpg", dst2);
    convertScaleAbs(destArray[2], dst3);
    //imwrite("F://program//image//gabor3.jpg", dst3);
    convertScaleAbs(destArray[3], dst4);
    //imwrite("F://program//image//gabor4.jpg", dst4);
    imshow("gabor1", dst1);
    imshow("gabor2", dst2);
    imshow("gabor3", dst3);
    imshow("gabor4", dst4);
    // 合并结果
    add(destArray[0], destArray[1], destArray[0]);
    add(destArray[2], destArray[3], destArray[2]);
    add(destArray[0], destArray[2], destArray[0]);
    Mat dst;
    convertScaleAbs(destArray[0], dst, 0.2, 0);
    // 二值化显示
    Mat gray, binary;
    // cvtColor(dst, gray, COLOR_BGR2GRAY);
    threshold(dst, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    imshow("result", dst);
    imshow("binary", binary);
    //imwrite("F://program//image//result_01.png", binary);
    waitKey(0);
    cv::destroyWindow("input");
    cv::destroyWindow("gabor1");
    cv::destroyWindow("gabor2");
    cv::destroyWindow("gabor3");
    cv::destroyWindow("gabor4");
    cv::destroyWindow("result");
    cv::destroyWindow("binary");
    waitKey(1);
}

void MainWindow::on_face_haar_clicked()
{
    String label = "Face";
    CascadeClassifier faceCascade;
    faceCascade.load("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/face-haar/haarcascade_frontalface_alt2.xml");//加载分类器
    VideoCapture capture;
    capture.open(0);// 打开摄像头
    //      capture.open("video.avi");    // 打开视频
    if (!capture.isOpened())
    {
        //_cprintf("open camera failed. \n");
        return;
    }
    Mat img, imgGray;
    vector<Rect> faces;
    while (1)
    {
        capture >> img;// 读取图像至img
        if (img.empty())continue;
        if (img.channels() == 3)
            cvtColor(img, imgGray, CV_RGB2GRAY);
        else
        {
            imgGray = img;
        }
        //double start = cv::getTickCount();
        faceCascade.detectMultiScale(imgGray, faces, 1.2, 6, 0, Size(0, 0));// 检测人脸
        //double end = cv::getTickCount();
        //_cprintf("run time: %f ms\n", (end - start));
        if (faces.size()>0)
        {
            for (int i = 0; i<faces.size(); i++)
            {
                rectangle(img, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 1, 8);
                putText(img, label, Point(faces[i].x, faces[i].y -5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255,0));
            }
        }
        imshow("CamerFace", img); // 显示
        if (waitKey(1) != -1)
            break;// delay ms 等待按键退出
    }
    cv::destroyWindow("CamerFace");
}


void calRealPoint(vector<vector<Point3f>>& obj, int boardWidth, int boardHeight, int imgNumber, int squareSize)
{
    vector<Point3f> imgpoint;
    for (int rowIndex = 0; rowIndex < boardHeight; rowIndex++)
    {
        for (int colIndex = 0; colIndex < boardWidth; colIndex++)
        {
            imgpoint.push_back(Point3f(rowIndex * squareSize, colIndex * squareSize, 0));
        }
    }
    for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
    {
        obj.push_back(imgpoint);
    }
}
Mat R, T, E, F;
Mat Rl, Rr, Pl, Pr, Q;
//映射表
Mat mapLx, mapLy, mapRx, mapRy;

Mat cameraMatrixL = (Mat_<double>(3, 3) << 530.1397548683084, 0, 338.2680507680664,
0, 530.2291152852337, 232.4902023212199,
0, 0, 1);
//获得的畸变参数
Mat distCoeffL = (Mat_<double>(5, 1) << -0.266294943795012, -0.0450330886310585, 0.0003024821418382528, -0.001243865371699451, 0.2973605735168139);

Mat cameraMatrixR = (Mat_<double>(3, 3) << 530.1397548683084, 0, 338.2680507680664,
0, 530.2291152852337, 232.4902023212199,
0, 0, 1);
Mat distCoeffR = (Mat_<double>(5, 1) << -0.266294943795012, -0.0450330886310585, 0.0003024821418382528, -0.001243865371699451, 0.2973605735168139);

void outputCameraParam(void)
{
    /*保存数据*/
    /*输出数据*/
    FileStorage fs("intrisics.yml", FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "cameraMatrixL" << cameraMatrixL << "cameraDistcoeffL" << distCoeffL << "cameraMatrixR" << cameraMatrixR << "cameraDistcoeffR" << distCoeffR;
        fs.release();
        cout << "cameraMatrixL=:" << cameraMatrixL << endl << "cameraDistcoeffL=:" << distCoeffL << endl << "cameraMatrixR=:" << cameraMatrixR << endl << "cameraDistcoeffR=:" << distCoeffR << endl;
    }
    else
    {
        cout << "Error: can not save the intrinsics!!!!" << endl;
    }

    fs.open("extrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "R" << R << "T" << T << "Rl" << Rl << "Rr" << Rr << "Pl" << Pl << "Pr" << Pr << "Q" << Q;
        cout << "R=" << R << endl << "T=" << T << endl << "Rl=" << Rl << endl << "Rr" << Rr << endl << "Pl" << Pl << endl << "Pr" << Pr << endl << "Q" << Q << endl;
        fs.release();
    }
    else
    {
        cout << "Error: can not save the extrinsic parameters\n";
    }
}


void MainWindow::on_camera2_clicked()
{
    //摄像头的分辨率
    const int imageWidth = 640;
    const int imageHeight = 480;
    //横向的角点数目
    const int boardWidth = 9;
    //纵向的角点数目
    const int boardHeight = 6;
    //总的角点数目
    const int boardCorner = boardWidth * boardHeight;
    //相机标定时需要采用的图像帧数
    const int frameNumber = 14;
    //标定板黑白格子的大小 单位是mm
    const int squareSize = 10;
    //标定板的总内角点
    const Size boardSize = Size(boardWidth, boardHeight);
    Size imageSize = Size(imageWidth, imageHeight);


    //R旋转矢量 T平移矢量 E本征矩阵 F基础矩阵
    vector<Mat> rvecs; //R
    vector<Mat> tvecs; //T
                       //左边摄像机所有照片角点的坐标集合
    vector<vector<Point2f>> imagePointL;
    //右边摄像机所有照片角点的坐标集合
    vector<vector<Point2f>> imagePointR;
    //各图像的角点的实际的物理坐标集合
    vector<vector<Point3f>> objRealPoint;
    //左边摄像机某一照片角点坐标集合
    vector<Point2f> cornerL;
    //右边摄像机某一照片角点坐标集合
    vector<Point2f> cornerR;

    Mat rgbImageL, grayImageL;
    Mat rgbImageR, grayImageR;
    Mat intrinsic;
    Mat distortion_coeff;
    //校正旋转矩阵R，投影矩阵P，重投影矩阵Q
    //映射表
    Mat mapLx, mapLy, mapRx, mapRy;
    Rect validROIL, validROIR;
    //图像校正之后，会对图像进行裁剪，其中，validROI裁剪之后的区域

    Mat img;
    int goodFrameCount = 1;
    while (goodFrameCount <= frameNumber)
    {
        char filename[100];
        /*读取左边的图像*/
        sprintf(filename, "/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/camer_cab/left%02d.jpg", goodFrameCount);
        rgbImageL = imread(filename, 1);
        imshow("chessboardL", rgbImageL);
        cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
        /*读取右边的图像*/
        sprintf(filename, "/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/camer_cab/right%02d.jpg", goodFrameCount);
        rgbImageR = imread(filename, 1);
        cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);

        bool isFindL, isFindR;
        isFindL = findChessboardCorners(rgbImageL, boardSize, cornerL);
        isFindR = findChessboardCorners(rgbImageR, boardSize, cornerR);
        if (isFindL == true && isFindR == true)
        {
            cornerSubPix(grayImageL, cornerL, Size(5, 5), Size(-1, 1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
            drawChessboardCorners(rgbImageL, boardSize, cornerL, isFindL);
            imshow("chessboardL", rgbImageL);
            imagePointL.push_back(cornerL);

            cornerSubPix(grayImageR, cornerR, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
            drawChessboardCorners(rgbImageR, boardSize, cornerR, isFindR);
            imshow("chessboardR", rgbImageR);
            imagePointR.push_back(cornerR);

            //_cprintf("the image %d is good\n",goodFrameCount);
            goodFrameCount++;
        }
        else
        {
            //_cprintf("the image is bad please try again\n");
        }

        if (waitKey(10) == 'q')
        {
            break;
        }
    }

    //计算实际的校正点的三维坐标，根据实际标定格子的大小来设置

    calRealPoint(objRealPoint, boardWidth, boardHeight, frameNumber, squareSize);
    //_cprintf("cal real successful\n");

    //标定摄像头
    double rms = stereoCalibrate(objRealPoint, imagePointL, imagePointR,
        cameraMatrixL, distCoeffL,
        cameraMatrixR, distCoeffR,
        Size(imageWidth, imageHeight), R, T, E, F, CALIB_USE_INTRINSIC_GUESS,
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));

    //_cprintf("Stereo Calibration done with RMS error = %f\n",rms);

    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl,
        Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY, -1, imageSize, &validROIL, &validROIR);

    //摄像机校正映射
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

    Mat rectifyImageL, rectifyImageR;
    cvtColor(grayImageL, rectifyImageL, CV_GRAY2BGR);
    cvtColor(grayImageR, rectifyImageR, CV_GRAY2BGR);

    imshow("RecitifyL Before", rectifyImageL);
    imshow("RecitifyR Before", rectifyImageR);

    //经过remap之后，左右相机的图像已经共面并且行对准了
    Mat rectifyImageL2, rectifyImageR2;
    remap(rectifyImageL, rectifyImageL2, mapLx, mapLy, INTER_LINEAR);
    remap(rectifyImageR, rectifyImageR2, mapRx, mapRy, INTER_LINEAR);


    imshow("rectifyImageL", rectifyImageL2);
    imshow("rectifyImageR", rectifyImageR2);

    outputCameraParam();
    //显示校正结果
    Mat canvas;
    double sf;
    int w, h;
    sf = 600. / MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width * sf);
    h = cvRound(imageSize.height * sf);
    canvas.create(h, w * 2, CV_8UC3);

    //左图像画到画布上
    Mat canvasPart = canvas(Rect(0, 0, w, h));
    cv::resize(rectifyImageL2, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
    Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),
        cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
    rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);

    //_cprintf("Painted ImageL\n");

    //右图像画到画布上
    canvasPart = canvas(Rect(w, 0, w, h));
    cv::resize(rectifyImageR2, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
    Rect vroiR(cvRound(validROIR.x*sf), cvRound(validROIR.y*sf),
        cvRound(validROIR.width*sf), cvRound(validROIR.height*sf));
    rectangle(canvasPart, vroiR, Scalar(0, 255, 0), 3, 8);

    //_cprintf("Painted ImageR\n");

    //画上对应的线条
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);

    imshow("rectified", canvas);
    //_cprintf("wait key\n");
    waitKey(0);

    cv::destroyAllWindows();
    waitKey(1);

}

int getDisparityImage(cv::Mat& disparity, cv::Mat& disparityImage, bool isColor)
{
    cv::Mat disp8u;
    disp8u = disparity;
    // 转换为伪彩色图像 或 灰度图像
    if (isColor)
    {
        if (disparityImage.empty() || disparityImage.type() != CV_8UC3 || disparityImage.size() != disparity.size())
        {
            disparityImage = cv::Mat::zeros(disparity.rows, disparity.cols, CV_8UC3);
        }
        for (int y = 0; y<disparity.rows; y++)
        {
            for (int x = 0; x<disparity.cols; x++)
            {
                uchar val = disp8u.at<uchar>(y, x);
                uchar r, g, b;

                if (val == 0)
                    r = g = b = 0;
                else
                {
                    r = 255 - val;
                    g = val < 128 ? val * 2 : (uchar)((255 - val) * 2);
                    b = val;
                }
                disparityImage.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
            }
        }
    }
    else
    {
        disp8u.copyTo(disparityImage);
    }
    return 1;
}
const int imageWidth = 640;                             //摄像头的分辨率
const int imageHeight = 480;
Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL;//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域
Rect validROIR;

Mat xyz;              //三维坐标
int blockSize = 0, uniquenessRatio = 0, numDisparities = 0;
Ptr<StereoBM> bm = StereoBM::create(16, 9);

Mat T_new = (Mat_<double>(3, 1) << -3.3269653179960471e+01, 3.7375231026230421e-01,-1.2058042444883227e-02);//T平移向量
//Mat rec = (Mat_<double>(3, 1) << -0.00306, -0.03207, 0.00206);//rec旋转向量
Mat R_new = (Mat_<double>(3, 3) << 9.9998505024526163e-01, 3.5253250461816949e-03,
    4.1798767087380161e-03, -3.4957471578341281e-03,
    9.9996894942320580e-01, -7.0625732745616225e-03,
    -4.2046447876106169e-03, 7.0478558986986593e-03,
    9.9996632377767658e-01);//R 旋转矩阵

/*****立体匹配*****/
void stereo_match(int, void*)
{
    bm->setBlockSize(2 * blockSize + 5);     //SAD窗口大小，5~21之间为宜
    bm->setROI1(validROIL);
    bm->setROI2(validROIR);
    bm->setPreFilterCap(31);
    bm->setMinDisparity(0);  //最小视差，默认值为0, 可以是负值，int型
    bm->setNumDisparities(numDisparities * 16 + 16);//视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(uniquenessRatio);//uniquenessRatio主要可以防止误匹配
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(-1);
    Mat disp, disp8, disparityImage;
    bm->compute(rectifyImageL, rectifyImageR, disp);//输入图像必须为灰度图
    disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));//计算出的视差是CV_16S格式
    reprojectImageTo3D(disp, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
    xyz = xyz * 16;
    getDisparityImage(disp8, disparityImage, true);
    imshow("disparity", disparityImage);
}

//立体匹配


void MainWindow::on_camera2_2_clicked()
{
    // TODO: 在此添加控件通知处理程序代码
    //立体校正
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R_new, T_new, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
        0, imageSize, &validROIL, &validROIR);
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

    rgbImageL = imread("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/camer_cab/left01.jpg", CV_LOAD_IMAGE_COLOR);
    cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
    rgbImageR = imread("/Users/qitianyu/Master/Semester1/Image_processing/ProjectFiles/Pro1_open_image/open_image/camer_cab/right01.jpg", CV_LOAD_IMAGE_COLOR);
    cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);

    imshow("ImageL Before Rectify", grayImageL);
    imshow("ImageR Before Rectify", grayImageR);

    /*
    经过remap之后，左右相机的图像已经共面并且行对准了
    */
    remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
    remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

    /*
    把校正结果显示出来
    */
    Mat rgbRectifyImageL, rgbRectifyImageR;
    cvtColor(rectifyImageL, rgbRectifyImageL, CV_GRAY2BGR);  //伪彩色图
    cvtColor(rectifyImageR, rgbRectifyImageR, CV_GRAY2BGR);
    //单独显示
    //rectangle(rgbRectifyImageL, validROIL, Scalar(0, 0, 255), 3, 8);
    //rectangle(rgbRectifyImageR, validROIR, Scalar(0, 0, 255), 3, 8);
    imshow("ImageL After Rectify", rgbRectifyImageL);
    imshow("ImageR After Rectify", rgbRectifyImageR);

    //显示在同一张图上
    Mat canvas;
    double sf;
    int w, h;
    sf = 600. / MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width * sf);
    h = cvRound(imageSize.height * sf);
    canvas.create(h, w * 2, CV_8UC3);   //注意通道

                                        //左图像画到画布上
    Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分
    cv::resize(rgbRectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小
    Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域
        cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
    //rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形
    cout << "Painted ImageL" << endl;

    //右图像画到画布上
    canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分
    cv::resize(rgbRectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
    Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
        cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
    //rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
    cout << "Painted ImageR" << endl;

    //画上对应的线条
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
    imshow("rectified", canvas);

    /*
    立体匹配
    */
    namedWindow("disparity", CV_WINDOW_AUTOSIZE);
    // 创建SAD窗口 Trackbar
    createTrackbar("BlockSize:\n", "disparity", &blockSize, 8, stereo_match);
    // 创建视差唯一性百分比窗口 Trackbar
    createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match);
    // 创建视差窗口 Trackbar
    createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);
    stereo_match(0, 0);

    waitKey(0);

    cv::destroyAllWindows();
    waitKey(1);

}
