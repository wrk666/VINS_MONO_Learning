#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;//条件变量，在完成将msg放入buf的操作后唤醒作用于process线程中的获取观测值数据的函数。
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;//buf是被多线程共享的，所以push或者pop时需要保证线程安全，设置互斥锁，m_buf，在操作前lock()，操作后unlock()。其他互斥锁同理。
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;// 订阅pose graph node发布的回环帧数据，存到relo_buf队列中，供重定位使用
int sum_of_wait = 0;

std::mutex m_buf;//buf互斥锁
std::mutex m_state;//state互斥锁,用于处理多个线程使用当前里程计信息（即tmp_P、tmp_Q、tmp_V）的冲突
std::mutex i_buf;//好像没用到
std::mutex m_estimator;//estimator互斥锁,用于处理多个线程使用VINS系统对象（即Estimator类的实例estimator）的冲突

double latest_time;  // 最近一次里程计信息对应的IMU时间戳
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    //这也是processIMU里相同的IMU积分
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

//更新IMU参数[P,Q,V,ba,bg,a,g]
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    //将对imu和图像数据进行初步对齐得到的数据结构，确保图像关联着对应时间戳内的所有IMU数据
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    // 直到把imu_buf或者feature_buf中的数据全部取出，才会退出while循环
    while (true)
    {
        //以下两个对时间戳的操作保证了IMUbuf两端时间戳都比对应的Img时间戳长，且基本上两帧Img之间的IMU数据都用上了
        //                  |-|-|-|-|-|             IMU
        //此帧在添加时已删除->|---------|---------|       IMG
        if (imu_buf.empty() || feature_buf.empty())//把其中一个读完才能跳出
            return measurements;
        // imu_buf队尾元素的时间戳，早于或等于feature_buf队首元素的时间戳（时间偏移补偿后），则需要等待接收IMU数据
        //                等待队尾更多的IMU数据
        //           |-|-|-|-|-|-|                       IMU
        //          |-------------|-------------|       IMG
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        // imu_buf队首元素的时间戳，晚于或等于feature_buf队首元素的时间戳（时间偏移补偿后），则需要剔除feature_buf队首多余的特征点数据
        //                 |-|-|-|-|-|-|-|               IMU
        //          |-------------|-------------|       IMG
        //     剔除这帧IMG数据
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());//emplace_back相比push_back能更好地避免内存的拷贝与移动
            imu_buf.pop();
        }
        //这里把下一个imu_msg也放进去了,但没有pop，因此当前图像帧和下一图像帧会共用这个imu_msg
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();//buf互斥锁
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();//条件锁

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);//通过lock_guard的方式构造互斥锁，在构造时加锁，析构时解锁
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
// process()是measurement_process线程的线程函数，在process()中处理VIO后端，包括IMU预积分、松耦合初始化和local BA
void process()
{
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;//是vector<pair<imu数据，img数据>>
        // unique_lock对象lk以独占所有权的方式管理mutex对象m_buf的上锁和解锁操作，所谓独占所有权，就是没有其他的 unique_lock对象同时拥有m_buf的所有权，
        // 新创建的unique_lock对象lk管理Mutex对象m_buf，并尝试调用m_buf.lock()对Mutex对象m_buf进行上锁，如果此时另外某个unique_lock对象已经管理了该Mutex对象m_buf,
        // 则当前线程将会被阻塞；如果此时m_buf本身就处于上锁状态，当前线程也会被阻塞（我猜的）。
        // 在unique_lock对象lk的声明周期内，它所管理的锁对象m_buf会一直保持上锁状态
        std::unique_lock<std::mutex> lk(m_buf);//可以unlock中途解锁，而std::lock_guard无法中途解锁，只能在作用域结束时解锁

        // std::condition_variable::wait(std::unique_lock<std::mutex>& lock, Predicate pred)的功能：
        // while (!pred()){
        //     wait(lock);
        // }
        // 当pred为false的时候，才会调用wait(lock)，阻塞当前线程，当同一条件变量在其它线程中调用了notify_*函数时，当前线程被唤醒。
        // 直到pred为ture的时候，退出while循环。
        // C++实际wait步骤：
        //1.wait()不断尝试获取互斥量锁，如果获取不到就一直在这里等待获取；（notify_one()会通知一个线程的wait(),条件变量在等待时会释放锁，在被notify时会得到锁）
        //2.如果获取到了，wait将再次对匿名函数表达式进行判断；
        //3.如果返回true，可以执行下去
        //4.如果返回false，那么wait()将解锁互斥量，并再次阻塞到本行；
        // 此处希望实现的目的是：当getmMasurements()拿到值不为空之后，就进行下面的IMU预积分等操作，否则就一直wait阻塞住

        // [&]{return (measurements = getMeasurements()).size() != 0;}是lamda表达式（匿名函数）
        // 先调用匿名函数，从缓存队列中读取IMU数据和图像特征点数据，如果measurements为空，则匿名函数返回false，调用wait(lock)，
        // 释放m_buf（为了使图像和IMU回调函数可以访问缓存队列），阻塞当前线程，等待被con.notify_one()唤醒
        // 直到measurements不为空时（成功从缓存队列获取数据），匿名函数返回true，则可以退出while循环。
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;//获得时间戳对齐的测量数据(IMUs, img_msg)s
                 });
        lk.unlock();
        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();// 最新IMU数据的时间戳
                double img_t = img_msg->header.stamp.toSec() + estimator.td;//用优化后的td来补偿Img时间戳
                if (t <= img_t)//TDOO：为什么当IMU最后的时间戳小于img_t时，不对时间戳进行处理？
                { 
                    if (current_time < 0) //第一次接受会这样
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    //3轴加计
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    //3轴角速度
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));//IMU预积分，通过中值积分得到当前PQV作为优化初值
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                // IMU时间戳晚于Img特征点数据（时间偏移补偿后）的第一帧IMU数据（也是一组measurement中的唯一一帧），对IMU数据进行插值，得到图像帧时间戳对应的IMU数据
                // 那如果IMU更早，后面没有IMU数据了为什么不解决呢？
                else
                {
                    // 时间戳的对应关系如下图所示：
                    //                                            current_time         t
                    // *               *               *               *               *     （IMU数据）
                    //                                                          |            （图像特征点数据）
                    //                                                        img_t
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // 设置重定位用的回环帧
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);//设置重定位帧(relo包括loop detection等操作)
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            //设置feature point id和具体features的map映射
            TicToc t_s;
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                // +0.5再int是四舍五入，因为int是默认向下取整的，
                // (int)2.4==2,(int)2.9==2;(int)(2.4+0.5)==(int)(2.9)=2;(int)(2.5+0.5)==(int)(3.0)=3，实现了四舍五入
                int v = img_msg->channels[0].values[i] + 0.5;//为什么要四舍五入？
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            estimator.processImage(image, img_msg->header);//视觉与IMU的初始化以及非线性优化的紧耦合

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            pubOdometry(estimator, header);//"odometry"
            pubKeyPoses(estimator, header);//"key_poses"
            pubCameraPose(estimator, header);//"camera_pose"
            pubPointCloud(estimator, header);//"history_cloud"
            pubTF(estimator, header);//"extrinsic" Tbc
            pubKeyframe(estimator);//"keyframe_point"、"keyframe_pose"
            if (relo_msg != NULL)
                pubRelocalization(estimator);//"relo_relative_pose"
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
    readParameters(n);
    estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    registerPub(n);
    //订阅topic并执行各自的回调
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    //订阅feature_tracker pub的feature topic，在callback 中 feature_buf.push(feature_msg);
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    //callbak中清空feature_buf，imu_buf，清除estimator的state和已读取的config paramter
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    //callback中relo_buf.push(points_msg);
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    std::thread measurement_process{process};//创建VIO主线程
    ros::spin();

    return 0;
}
