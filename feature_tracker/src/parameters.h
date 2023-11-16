#pragma once
#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>

extern int ROW;//图像高度
extern int COL;//图像宽度
extern int FOCAL_LENGTH;
const int NUM_OF_CAM = 1;


extern std::string IMAGE_TOPIC;
extern std::string IMU_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;//特征点最大个数
extern int MIN_DIST;//特征点之间的最小间隔
extern int WINDOW_SIZE;//滑动窗口的大小
extern int FREQ;//控制发布次数的频率
extern double F_THRESHOLD;//ransac算法的阈值
extern int SHOW_TRACK;//是否发布跟踪点的图像
extern int STEREO_TRACK;//双目跟踪则为1
extern int EQUALIZE;//光太亮或太暗则为1，进行直方图均衡化
extern int FISHEYE;//如果是鱼眼相机则为1
extern bool PUB_THIS_FRAME;//是否需要发布特征点

void readParameters(ros::NodeHandle &n);
