#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>


class Estimator
{
  public:
    Estimator();

    void setParameter();

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header);
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();


    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;//重力细化就是对这个变量
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    // camera与IMU的外参
    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    Vector3d Ps[(WINDOW_SIZE + 1)]; // 滑动窗口中各帧在世界坐标系下的位置
    Vector3d Vs[(WINDOW_SIZE + 1)]; // 滑动窗口中各帧在世界坐标系下的速度
    Matrix3d Rs[(WINDOW_SIZE + 1)]; // 滑动窗口中各帧在世界坐标系下的旋转
    Vector3d Bas[(WINDOW_SIZE + 1)]; // 滑动窗口中各帧对应的加速度计偏置
    Vector3d Bgs[(WINDOW_SIZE + 1)]; // 滑动窗口中各帧对应的陀螺仪偏置
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];//总共11维的指针数组，放的是每一帧的预积分类的指针,为什么要定义WINDOW_SIZE个，一个不就够了吗？反正放的是每两个IMU数据之间的预积分
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];//是一个数组，元素是vector<一个measurement中的每次积分的dt>，下面两个变量同理。
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;// 最新帧在滑动窗口中的索引（0，1，2，... ，WINDOW_SIZE-1）WINDOW_SIZE=10，等frame_count==10时就该进行marg了，以保证window内只有10帧数据
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;

    //因为ceres用的是double数组，所以在传入之前要用vector2double做类型装换
    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];//传给ceres的WINDOW内的所有pose
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];

    //用于check参数是否被加入，加了多少
    double check_para_Pose[WINDOW_SIZE + 1][SIZE_POSE] = {{0}};//传给ceres的WINDOW内的所有pose
    double check_para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS] = {{0}};
    double check_para_Feature[NUM_OF_F][SIZE_FEATURE] = {{0}};
    double check_para_Ex_Pose[NUM_OF_CAM][SIZE_POSE] = {{0}};
    double check_para_Td[1][1] = {{0}};

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;//编译器自动生成默认构造函数
    vector<double *> last_marginalization_parameter_blocks;//上次marg后的待优化参数块，是一个double数组的vector，依靠idx来访问

    // 存储所有的ImageFrame对象（每读取一帧图像就会构建ImageFrame对象）
    //时间戳map映射ImgFrame，ImageFrame是里面有该帧pose，有该帧预积分,有所有feature_id->features的mapfeatures是pair<camera_id, Mat<7,1>>
    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;//单帧临时的预积分的指针,和pre_integrations同一时间进行积分，用于在创建ImageFrame对象时，把该指针赋给该帧图像对应的pre_integration

    //relocalization variable
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;//j帧index
    int relo_frame_local_index;
    vector<Vector3d> match_points;//(i帧归一化x，y，feature_id)
    double relo_Pose[SIZE_POSE];//WINDOW内与old帧loop上的j帧的pose Tw2_bj
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;//old frame的Tw1_bi
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;//
    Quaterniond relo_relative_q;
    double relo_relative_yaw;

    int l_ = 0;

    //用于统计求解时间
    double solver_time_sum_;
    double solve_times_;

    double makeHessian_time_sum_;
    double makeHessian_times_;
    double pure_makeHessian_time_sum_;
    double pure_makeHessian_times_;
};
