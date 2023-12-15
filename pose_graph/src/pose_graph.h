#pragma once

#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <string>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <queue>
#include <assert.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
#include <stdio.h>
#include <ros/ros.h>
#include "keyframe.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "utility/CameraPoseVisualization.h"
#include "utility/tic_toc.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"
#include "ThirdParty/DBoW/TemplatedDatabase.h"
#include "ThirdParty/DBoW/TemplatedVocabulary.h"


#define SHOW_S_EDGE false
#define SHOW_L_EDGE true
#define SAVE_LOOP_PATH true

using namespace DVision;
using namespace DBoW2;

class PoseGraph
{
public:
	PoseGraph();//构造
	~PoseGraph();
	void registerPub(ros::NodeHandle &n);
	void addKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop);
	void loadKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop);
	void loadVocabulary(std::string voc_path);
	void updateKeyFrameLoop(int index, Eigen::Matrix<double, 8, 1 > &_loop_info);
	KeyFrame* getKeyFrame(int index);
	nav_msgs::Path path[10];
	nav_msgs::Path base_path;
	CameraPoseVisualization* posegraph_visualization;
	void savePoseGraph();
	void loadPoseGraph();
	void publish();
	//loop i，j帧计算的Tw1_w2
	Vector3d t_drift;
	double yaw_drift;
	Matrix3d r_drift;
	// world frame( base sequence or first sequence)<----> cur sequence frame  
	Vector3d w_t_vio;
	Matrix3d w_r_vio;


private:
	int detectLoop(KeyFrame* keyframe, int frame_index);
	void addKeyFrameIntoVoc(KeyFrame* keyframe);
	void optimize4DoF();
	void updatePath();
	list<KeyFrame*> keyframelist;//所有KF
	std::mutex m_keyframelist;//一些互斥锁
	std::mutex m_optimize_buf;
	std::mutex m_path;
	std::mutex m_drift;
	std::thread t_optimization;//优化线程
	std::queue<int> optimize_buf;

	int global_index;
	int sequence_cnt;
	vector<bool> sequence_loop;
	map<int, cv::Mat> image_pool;
	int earliest_loop_index;
	int base_sequence;

	BriefDatabase db;//描述子数据库,from 3rd lib
	BriefVocabulary* voc; //词典

	ros::Publisher pub_pg_path;
	ros::Publisher pub_base_path;
	ros::Publisher pub_pose_graph;
	ros::Publisher pub_path[10];
};

template <typename T>
T NormalizeAngle(const T& angle_degrees) {
  if (angle_degrees > T(180.0))
  	return angle_degrees - T(360.0);
  else if (angle_degrees < T(-180.0))
  	return angle_degrees + T(360.0);
  else
  	return angle_degrees;
};

//是ceres::AutoDiffLocalParameterization的Functor参数，需要重载operator()
class AngleLocalParameterization {
 public:

  template <typename T>
  //重载()，进行弧度/角度运算，并归一化到[-360, 360]
  bool operator()(const T* theta_radians, const T* delta_theta_radians,
                  T* theta_radians_plus_delta) const {
    *theta_radians_plus_delta =
        NormalizeAngle(*theta_radians + *delta_theta_radians);

    return true;
  }

  //定义yaw角的更新方式  <functor_, >
  static ceres::LocalParameterization* Create() {
    return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization,
                                                     1, 1>);
  }
};

template <typename T> 
void YawPitchRollToRotationMatrix(const T yaw, const T pitch, const T roll, T R[9])
{
    //转为弧度
	T y = yaw / T(180.0) * T(M_PI);
	T p = pitch / T(180.0) * T(M_PI);
	T r = roll / T(180.0) * T(M_PI);


	R[0] = cos(y) * cos(p);
	R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
	R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
	R[3] = sin(y) * cos(p);
	R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
	R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
	R[6] = -sin(p);
	R[7] = cos(p) * sin(r);
	R[8] = cos(p) * cos(r);
};

template <typename T> 
void RotationMatrixTranspose(const T R[9], T inv_R[9])
{
	inv_R[0] = R[0];
	inv_R[1] = R[3];
	inv_R[2] = R[6];
	inv_R[3] = R[1];
	inv_R[4] = R[4];
	inv_R[5] = R[7];
	inv_R[6] = R[2];
	inv_R[7] = R[5];
	inv_R[8] = R[8];
};

template <typename T> 
void RotationMatrixRotatePoint(const T R[9], const T t[3], T r_t[3])
{
	r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
	r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
	r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
};

//4DoF pose graph的factor，
struct FourDOFError
{
	FourDOFError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i)
				  :t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i){}

	template <typename T>
	bool operator()(const T* const yaw_i, const T* ti, const T* yaw_j, const T* tj, T* residuals) const
	{
		T t_w_ij[3];
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		// euler to rotation
		T w_R_i[9];
		YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
		// rotation transpose
		T i_R_w[9];
		RotationMatrixTranspose(w_R_i, i_R_w);
		// rotation matrix rotate point
		T t_i_ij[3];
		RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

		residuals[0] = (t_i_ij[0] - T(t_x));
		residuals[1] = (t_i_ij[1] - T(t_y));
		residuals[2] = (t_i_ij[2] - T(t_z));
		residuals[3] = NormalizeAngle(yaw_j[0] - yaw_i[0] - T(relative_yaw));

		return true;
	}

	//4为residual，输入参数分别为1(yaw_i)，3(tw2_i)，1(yaw_j)，3(tw2_j)，注意这里的w在优化前是w2，在优化后由于保持了全局一致性，就变为w1
	static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
									   const double relative_yaw, const double pitch_i, const double roll_i) 
	{
      //分别是sequential和loop边的yaw(1)和t(3)的residual
	  return (new ceres::AutoDiffCostFunction<
	          FourDOFError, 4, 1, 3, 1, 3>(
	          	new FourDOFError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
	}

	double t_x, t_y, t_z;
	double relative_yaw, pitch_i, roll_i;

};

struct FourDOFWeightError
{
    //构造，weight=1，yaw的weight sacle为1/10
	FourDOFWeightError(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i)
				  :t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i){
				  	weight = 1;
				  }

    //auto diff需要重载()来定义residual的计算方法
	template <typename T>
	bool operator()(const T* const yaw_i, const T* ti, const T* yaw_j, const T* tj, T* residuals) const
	{
		T t_w_ij[3];
		t_w_ij[0] = tj[0] - ti[0];
		t_w_ij[1] = tj[1] - ti[1];
		t_w_ij[2] = tj[2] - ti[2];

		// euler to rotation 欧拉角转旋转矩阵，[out]w_R_i
		T w_R_i[9];
		YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
		// rotation transpose 旋转矩阵转置 [out]i_R_w
		T i_R_w[9];
		RotationMatrixTranspose(w_R_i, i_R_w);
		// rotation matrix rotate point  (w)ti_j转到(i)ti_j
		T t_i_ij[3];
		RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

        // t_x,t_y和t_z都是测量值
        // residuals前三项表示平移项残差，第四项表示旋转项yaw角的残差，注意这里yaw角的残差缩小了10倍(目的是避免优化过程中对yaw角的调整过大)
		residuals[0] = (t_i_ij[0] - T(t_x)) * T(weight);
		residuals[1] = (t_i_ij[1] - T(t_y)) * T(weight);
		residuals[2] = (t_i_ij[2] - T(t_z)) * T(weight);
		residuals[3] = NormalizeAngle((yaw_j[0] - yaw_i[0] - T(relative_yaw))) * T(weight) / T(10.0);

		return true;
	}

	//自动求导
	static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
									   const double relative_yaw, const double pitch_i, const double roll_i) 
	{
	  return (new ceres::AutoDiffCostFunction<
	          FourDOFWeightError, 4, 1, 3, 1, 3>(
	          	new FourDOFWeightError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
	}

	double t_x, t_y, t_z;
	double relative_yaw, pitch_i, roll_i;
	double weight;

};