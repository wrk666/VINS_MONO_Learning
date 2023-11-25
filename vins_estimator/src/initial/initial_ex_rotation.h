#pragma once 

#include <vector>
#include "../parameters.h"
using namespace std;

#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Dense>
using namespace Eigen;
#include <ros/console.h>

/* This class help you to calibrate extrinsic rotation between imu and camera when your totally don't konw the extrinsic parameter */
class InitialEXRotation
{
public:
	InitialEXRotation();
    bool CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result);
private:
	Matrix3d solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres);

	//进行三角化并用深度找出正确的解
    double testTriangulation(const vector<cv::Point2f> &l,
                             const vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double> t);
    //E反解出Rt
    void decomposeE(cv::Mat E,
                    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);
    int frame_count;

    vector< Matrix3d > Rc;//Rc即Rck_ck+1
    vector< Matrix3d > Rimu;//Rimu即delta_q即qbk+1_bk即Rbk+1_bk
    vector< Matrix3d > Rc_g;//Rc_g * Rc就是整个左移右的residual
    Matrix3d ric;
};


