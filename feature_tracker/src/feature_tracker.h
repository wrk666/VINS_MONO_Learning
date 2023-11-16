#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);//判断跟踪的特征点是否在图像边界内

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);//去除无法跟踪的特征点
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img,double _cur_time);//对图像使用光流法进行特征点跟踪，去畸变，outlier rejection，重投影至单位球

    void setMask();//对跟踪点进行排序并去除密集点

    void addPoints();//添将新检测到的特征点n_pts，ID初始化-1，跟踪次数1

    bool updateID(unsigned int i);//更新特征点id

    void readIntrinsicParameter(const string &calib_file);//读取相机内参

    void showUndistortion(const string &name);//显示去畸变矫正后的特征点

    void rejectWithF();//通过F矩阵去除outliers

    void undistortedPoints();//对特征点的图像坐标去畸变矫正，并计算每个角点的速度

    cv::Mat mask;//图像掩码
    cv::Mat fisheye_mask;//鱼眼相机mask，用来去除边缘噪点
    cv::Mat prev_img, cur_img, forw_img;//prev：上一帧发布的图像数据；cur：光流跟的第一帧；forw：光流跟的第二帧
    vector<cv::Point2f> n_pts;//每一帧中新提取的特征点，在添加时其ids[i]会被置为-1
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;//对应的光流图像特征点
    vector<cv::Point2f> prev_un_pts, cur_un_pts;//归一化相机坐标系下的坐标
    vector<cv::Point2f> pts_velocity;//当前帧相对前一帧特征点沿x,y方向的像素移动速度(应该是当前帧的光流输出)
    vector<int> ids;//能够被跟踪到的特征点的id
    vector<int> track_cnt;//当前帧forw_img中每个特征点被追踪的时间次数
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;//相机模型
    double cur_time;
    double prev_time;

    static int n_id;//用来作为特征点id，每检测到一个新的特征点，就将n_id作为该特征点的id，然后n_id加1
};
