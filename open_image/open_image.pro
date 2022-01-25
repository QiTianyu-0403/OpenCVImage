QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    mainwindow.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

INCLUDEPATH += /usr/local/Cellar/opencv/4.5.3_2/include/opencv4

LIBS += -L/usr/local/Cellar/opencv/4.5.3_2/lib\
-lopencv_core \
 -lopencv_highgui \
 -lopencv_imgproc \
  -lopencv_imgcodecs \
  -lopencv_shape \
  -lopencv_videoio \
  -lopencv_bgsegm \
  -lopencv_barcode \
  -lopencv_imgproc \
  -lopencv_imgcodecs \
  -lopencv_video \
  -lopencv_calib3d \
  -lopencv_features2d \
  -lopencv_ml \
  -lopencv_objdetect \
  -lopencv_optflow \
  -lopencv_xfeatures2d \


#INCLUDEPATH += /usr/local/Cellar/opencv@3/3.4.14_3/include/opencv

#LIBS += -L/usr/local/Cellar/opencv@3/3.4.14_3/lib\
#-lopencv_core \
# -lopencv_highgui \
# -lopencv_imgproc \
#  -lopencv_imgcodecs \
#  -lopencv_shape \
#  -lopencv_videoio \
#  -lopencv_bgsegm \
#  -lopencv_barcode \
#  -lopencv_imgproc \
#  -lopencv_imgcodecs \
#  -lopencv_video \
#  -lopencv_calib3d \
#  -lopencv_ccalib \
#  -lopencv_ml



