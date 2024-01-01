#include "estimator.h"
#include "solver/solve.h"

//#define CERES_SOLVE
uint8_t strategy = 3;//先定义为全局变量，后面再优化

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
}

//视觉测量残差的协方差矩阵
void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    //这里假设标定相机内参时的重投影误差△u=1.5 pixel，(Sigma)^(-1)=(1.5/f * I(2x2))^(-1) = (f/1.5 * I(2x2))
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;


    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

//IMU预积分：IntegrationBase类，IMU预积分具体细节
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;//保存本次measurement中的第一帧IMU数据（有啥用？）
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])//如果frame_count的积分为空则new一个预积分对象
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)//第0帧[0]没有预积分，第[0]与第[1]帧之间才有预积分
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);//调用IntegrationBase中定义的成员函数push_back，保存变量并propagate预积分
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);//保存这两帧IMU之间的时间间隔，用于预积分
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        //IMU预积分(为什么这里要重新再算一遍？push_back里面不是重新算过了吗？为什么不直接把delta_p等结果拿出直接用？)
        // 用IMU数据进行积分，当积完一个measurement中所有IMU数据后，就得到了对应图像帧在世界坐标系中的Ps、Vs、Rs（这里为什么是相对于世界坐标系呢？为什么不把关于world系的抽出来呢？）
        // 下面这一部分的积分，在没有成功完成初始化时似乎是没有意义的，因为在没有成功初始化时，对IMU数据来说是没有世界坐标系的
        // 当成功完成了初始化后，下面这一部分积分才有用，它可以通过IMU积分得到滑动窗口中最新帧在世界坐标系中的P V R
        int j = frame_count;//到后面frame_count一直为window_size即10
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;//为什么要有重力g？
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);//mid-point中值法计算a，w在k~k+1时刻内的测量值
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;//更新本次预积分的初始值
    gyr_0 = angular_velocity;
}

//实现了视觉与IMU的初始化以及非线性优化的紧耦合
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    // 把当前帧图像（frame_count）的特征点添加到f_manager.feature容器中
    // 计算第2最新帧与第3最新帧之间的平均视差（当前帧是第1最新帧），判断第2最新帧是否为KF
    // 在未完成初始化时，如果窗口没有塞满，那么是否添加关键帧的判定结果不起作用，滑动窗口要塞满
    // 只有在滑动窗口塞满后，或者初始化完成之后，才需要滑动窗口，此时才需要做关键帧判别，根据第2最新关键帧是否为关键帧选择相应的边缘化策略
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD;//如果第2新帧是KF则marg掉最老的一帧
    else
        marginalization_flag = MARGIN_SECOND_NEW;//如果第二新帧不是KF则直接丢掉最新帧的视觉measurement，并对IMU积分propogate

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
    //用于下一个measurement进行积分
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    //不知道关于外参的任何info，需要标定
    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            // 找相邻两帧(bk, bk+1)之间的tracking上的点，构建一个pair，所有pair是一个vector，即corres(pondents),first=前一帧的去畸变的归一化平面上的点，second=后一帧的
            // 要求it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            //旋转约束+SVD分解求取Ric旋转外参
            //delta_q即qbk_bk+1,是从k时刻积分到k+1，所以是qbk_bk+1(从左往右读)
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL)// 需要初始化
    {
        if (frame_count == WINDOW_SIZE)// 滑动窗口中塞满了才进行初始化(初始化并不影响KF的筛选，KF筛选仍然使用：视差>=10和tracked_num<20来判断，满足其一则是KF
        {
            bool result = false;
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1) //确保有足够的frame参与初始化，有外参，且当前帧时间戳大于初始化时间戳+0.1秒
            {
               result = initialStructure();//执行视觉惯性联合初始化
               initial_timestamp = header.stamp.toSec();
            }
            //初始化成功则进行一次非线性优化，不成功则进行滑窗操作
            if(result)
            {
                solver_flag = NON_LINEAR;//求解
                solveOdometry();//重新三角化，并后端求解
                slideWindow();
                ROS_DEBUG("Ps[0] addr: %ld", reinterpret_cast<long>(&Ps[0]));
                f_manager.removeFailures();
                ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
                
            }
            else
                slideWindow();
        }
        else
            frame_count++;//只在这里自增，自增到WINDOW_SIZE(10)之后就不再自增了，后面都是WINDOW_SIZE(10)，即后面的优化都是需要进行marg的
    }
    else//flag==NON_LINEAR,初始化完成，需要求解后端
    {
        TicToc t_solve;
        solveOdometry();
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        // 以下5种情况会判定为fail：
        // 1,2：ba或bg过大
        // 3,4,5：本次WINDOW内和上次WINDOW内的最后一帧pose(Tw_b[k])之间的relative pose的t或z或角度变化过大
        // fail之后会clear state并重启系统（重新初始化）
        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();//所有buff，预积分等都clear，erase，delete
            setParameter();//清零外参，time offset
            ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow();//根据marg flag marg掉old或者2nd，管理优化变量，数据，深度等
        ROS_DEBUG("Ps[0] addr: %ld", reinterpret_cast<long>(&Ps[0]));
        f_manager.removeFailures();//去掉未三角化出正深度的landmark
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS(本次优化且划窗之后，保存WINDOW内的所有KF的translation)
        key_poses.clear();
        //slideWindow后最后两个Ps相同，所以用11个数据无所谓
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];//保留这一WINDOW内的最新一帧的信息，供下次failureDetection()使用
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

//执行视觉惯性联合初始化,包含两部分：1. visual SfM，2.visual和IMU的align(估计gyro bias，scale，重力细化RefineGravity)
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        //遍历window内所有的ImageFrame
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;//该帧总时间
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;//速度/时间=加速度
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);//线加速度均值，因为第一帧没有，所以-1
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));//求线加速度的标准差
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)//如果加速度方差小于0.25，则证明加速度波动较小，证明IMU激励不够（TODO：这个0.25跟标定qcb旋转外参SVD的特征值的那个0.25有关系吗？）
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }

    // global sfm
    Quaterniond Q[frame_count + 1];//存放window内所有帧相对____的pose T___i
    Vector3d T[frame_count + 1];
    //把window内所有id对应的所有feature都存到一个vector<SFMFeature>中
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)//feature是list，元素是装了window内的所有该id的feature的vector，即一个feature_id对应一个vector
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;//未被三角化
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)//window内该id对应的所有的Matrix<double, 7, 1>
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));//observation: 所有观测到该特征点的图像帧ID和图像坐标
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    //选择window内第一个满足“tracking数量>20,平均视差>30”的帧(l)与最新帧之间的relative pose，是从最新帧到第l帧Tl_cur，就是下面的Tw_cur
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    l_ = l;//将l赋给成员，便于外面查看l的帧数

    //求解SfM问题：对窗口中每个图像帧求解sfm问题，得到所有图像帧相对于参考帧l的旋转四元数Q、平移向量T和特征点坐标sfm_tracked_points。
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        //如果初始化不成功，就marg掉最老的帧(在all_image_frame中把最老的帧也删掉，但是在MARGIN_SECOND_NEW时就不会删掉all_image_frame中的帧)
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame(直接用cv的库函数，没有再使用ceres构建problem)
    // 由于并不是第一次视觉初始化就能成功，此时图像帧数目有可能会超过滑动窗口的大小
    // 所以再视觉初始化的最后，要求出滑动窗口外的帧的位姿
    // 最后把世界坐标系从帧l的相机坐标系，转到帧l的IMU坐标系
    // 4.对于非滑动窗口的所有帧，提供一个初始的R,T，然后solve pnp求解pose
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );//时间戳map映射ImgFrame，ImageFrame是里面有的所有id->features的map,features是pair<camera_id, Mat<7,1>>
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i].stamp.toSec()) // all_image_frame与滑动窗口中对应的帧，SfM阶段已经计算过，无需再次计算
        {
            frame_it->second.is_key_frame = true;// 滑动窗口中所有帧都是关键帧
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();// 根据各帧相机坐标系的姿态和外参，得到用各帧IMU坐标系的姿态（对应VINS Mono论文(2018年的期刊版论文)中的公式（6））。
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }
        // 为滑动窗口外的帧提供一个初始位姿
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();//qwc^(-1)=qcw
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;// 初始化时位于滑动窗口外的帧是非关键帧
        vector<cv::Point3f> pts_3_vector;// 用于pnp解算的3D点
        vector<cv::Point2f> pts_2_vector;// 用于pnp解算的2D点
        for (auto &id_pts : frame_it->second.points) // 对于该帧中的特征点
        {
            int feature_id = id_pts.first;// 特征点id
            for (auto &i_p : id_pts.second)// 由于可能有多个相机，所以需要遍历。i_p对应着一个相机所拍图像帧的特征点信息
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())//如果找到了已经Triangulation的,说明在sfm_tracked_points中找到了相应的3D点
                {
                    // 记录该已被Triangulated的id特征点的3D位置
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);

                    // 记录该id的特征点在该帧图像中的2D位置
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1)) // pnp求解失败
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();//qwc = qcw^(-1)
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose(); // Tc0_ck * Tbc^(-1) = Tc0_bk转到c0系下看bk
        frame_it->second.T = T_pnp;
    }
    ROS_DEBUG_STREAM("\nhere l_: " << l_ <<  "\nKF[0] Rs[0]:\n" << all_image_frame[Headers[0].stamp.toSec()].R);
    if (visualInitialAlign())//视觉惯性对齐:bg，gc0，s，v的估计
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;//待优化变量[vk,vk+1,w,s],维度是(all_image_frame.size() * 3 + 2 + 1)
    //估计陀螺仪的偏置，速度、重力和尺度初始化，重力细化
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    //原文：we can get the rotation qw c0 between the world frame and the
    //camera frame c0 by rotating the gravity to the z-axis. We then
    //rotate all variables from the reference frame (·)c0 to the world
    //frame (·)w.
    // change state(以下仅对WINDOW内的frame进行操作)
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;//IMU相对于world(即c0,此时还是l帧)的pose:Tc0_b[k]
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;//Rc0_b[k]
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }
    ROS_DEBUG_STREAM("\nhere l_: " << l_
                        << "\nKF Rs[0]:\n" << Rs[0]);
    //1.梳理一下：此时all_image_frame[Headers[i].stamp.toSec()].R，T都是Tc0_bk
    //所以Ps,Rs也都是Tc0_bk

    //将三角化出的深度均设为-1，重新三角化
    VectorXd dep = f_manager.getDepthVector();//获取WINDOW内所有成功Triangulated出深度的landmark，求其逆深度
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);//重新赋深度(都是-1)

    //triangulat on cam pose , no tic
    //平移tic未标定，设为0
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));//Ps是tc0_bk(里面要转为tc_ck使用)

    double s = (x.tail<1>())(0);//取优化出的scale
    //gyro bias bg改变了，需要重新IMU预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        //对每两帧camera之间的IMU数据重新进行积分(每次积分的观测初值(acc_0,gyro_0)仍然使用之前保存的linearized_acc, linearized_gyro)
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    ROS_INFO_STREAM("TIC[0]:\n" << TIC[0].transpose());

    //2.这里将Ps转换为(c0)tb0_bk
    for (int i = frame_count; i >= 0; i--) {
        //论文式(6)，看起来Rs应该是Rc0_bk(这个时候c0应该还没变为world，所以应该是在恢复米制单位)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);//这里输入的Ps还是tc0_bk,输出的Ps是(c0)tb0_bk，是在c0系下看的这个translation
        //TIC[0]为0代表第一项 s * Ps[i] - Rs[i] * TIC[0]=s*Ps[i]，即s*tc0_b[k]=s*tc0_c[k](因为此时Ps=tc0_b[k])
        ROS_INFO_STREAM("TIC[0]:" << TIC[0].transpose()
                        << "\ns * Ps[i] - Rs[i] * TIC[0]: " << (s * Ps[i] - Rs[i] * TIC[0]).transpose()
                        << "\ns * Ps[i]: " << (s * Ps[i]).transpose()
                        << "\nl_: " << l_
                        << "\nPs[0]: " << Ps[0].transpose()//看他是否为0，如果不是0则证明我把c0和c[0]弄混了
                        << "\ns * Ps[0]: " << (s * Ps[0]).transpose());
    }

    //速度，深度处理
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);//更新bk系下的速度：Rc0_bk * (bk)vk = (c0)vk
        }
    }
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;//恢复真实世界下尺度的深度
    }

    //g是world系下的重力向量，Rs[0]是Rc0_b[0]
    ROS_DEBUG_STREAM("\nRs[0] is Rc0_b0:\n" << Rs[0]
                        <<"\nRbc^T:\n" << RIC[0].transpose());
    Matrix3d R0 = Utility::g2R(g);//求出gc0->gw(0,0,1)的pitch和roll方向的旋转R0
    ROS_DEBUG_STREAM("\nhere1 R0.yaw = \n" << Utility::R2ypr(R0).x());
    Eigen::Vector3d here1_Rs0_ypr = Utility::R2ypr(Rs[0]);
    double here1_Rs0_yaw = here1_Rs0_ypr.x();//Rs[0].yaw

    double yaw = Utility::R2ypr(R0 * Rs[0]).x();//和transformed_yaw相等，说明不是运算精度的问题，可能就是旋转之后yaw会受到影响
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    ROS_DEBUG_STREAM("\nhere2 yaw = :\n" << yaw <<
                          "\nRs[0].yaw = :\n" << here1_Rs0_yaw <<
                          "\neventually, R0.yaw = \n" << Utility::R2ypr(R0).x());
    g = R0 * g;//将估计的重力g旋转到world系：yaw * Rwc0*g^(c0)=g^w，
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;//rotdiff最后使得在world系下，b[0]真的yaw为0°
    //(PRV)w_b[k] = Rw_b[0] * (PRV)c0_b[k]
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];//(w)vb0_bk
        Vs[i] = rot_diff * Vs[i];//(w)vb0_bk
        ROS_DEBUG_STREAM("\ni=" << i <<"    Rs[i].yaw = \n" << Utility::R2ypr(Rs[i]).x());
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    return true;
}

//选择window内第一个满足tracking数量>20,平均视差>30的帧(l)与最新帧之间的relative pose，是从最新帧到第l帧
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    //对应论文V.A节
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        // 找第i帧和buffer内最后一帧，(i, WINDOW_SIZE),之间的tracking上的点，构建一个pair，
        // 所有pair是一个vector，即corres(pondents),first=前一帧的去畸变的归一化平面上的点，second=后一帧的
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)//要求两帧的共视点大于20对
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();//计算共视点的视差(欧氏距离)
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());//平均视差
            //用内参将归一化平面上的点转化到像素平面fx*X/Z + cx，cx相减抵消，z=1，所以直接就是fx*X
            //求的Rt是当前帧([WINDOW_SIZE]帧)到第l帧的坐标系变换Rl_[WINDOW_SIZE]
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
//                l = l+2;
//                ROS_DEBUG("change l to l+2 = %d", l);
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry()
{
    //需要在WINDOW满之后才进行求解，没满之前，初始化阶段pose由sfm等求解
    if (frame_count < WINDOW_SIZE)
        return;
    //
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        //在optimize和marg，在新的start_frame上重新三角化landmark
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        optimization();
    }
}

//vector转换成double数组，因为ceres使用数值数组
//Ps、Rs转变成para_Pose，Vs、Bas、Bgs转变成para_SpeedBias
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

// 优化一次之后，求出优化前后的第一帧的T，用于后面作用到所有的轨迹上去
// 数据转换，vector2double的相反过程
// 同时这里为防止优化结果往零空间变化，会根据优化前后第一帧的位姿差进行修正。
void Estimator::double2vector()
{
    //窗口第一帧优化前的位姿
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);//R[0]
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    //窗口第一帧优化后的位姿 q(wxyz)
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                     para_Pose[0][3],
                                                     para_Pose[0][4],
                                                     para_Pose[0][5]).toRotationMatrix());
    //(R_before_after).yaw（转到被减，变换到before）
    //TODO：确定到底是哪个  若是R_after_before.x()则下面使用rot_diff做的矫正就不对了,para_Pose肯定是after的
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO：了解欧拉角奇异点
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }
    // 根据位姿差做修正，即保证第一帧优化前后位姿不变(似乎只是yaw不变，那tilt呢？)
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    //外参
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    //转为逆深度，并置flag
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    //time offset
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if(relocalization_info)
    {
        //按照WINDOW内第一帧的yaw角变化对j帧进行矫正
        Matrix3d relo_r;//j帧矫正之后的T
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;//保证第[0]帧不变之后，+origin_P0转为世界系下的t

        //由于pitch和roll是可观的，所以我们在BA中一直都在优化pitch和roll，但由于yaw不可观，
        //所以即使漂了，可能还是满足我们BA的最优解，所以需要手动进行矫正
        //prev_relo_r=Tw1_bi, relo_Pose=Tw2_bi，这两个pose间的yaw和t的漂移Rw1_w2,tw1_w2
        double drift_correct_yaw;
        //yaw drift, Rw1_bi.yaw() - Rw2_bi.yaw=Rw1_w2.yaw()
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        //tw1_w2
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;


        //Tw2_bi^(-1) * Tw2_bj = Tbi_bj
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        //Rw2_bj.yaw() - Rw2_bi.yaw() = Rbi_bj.yaw()
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());


/*        //验证Tw1w2是否正确。  结果不一样，不知道为啥。
        //1.计算Rw1_w2 = Rw1_bi * Rw2_bi^(-1)
        Matrix3d Rw1_w2 = prev_relo_r * relo_r;
        //2. 计算Tw1_w2中的Rw1_w2 = Tw1_bi.R * Tbi_bj.R * Rw2_bj^(-1)
        Matrix3d Rw1_w2_prime = prev_relo_r * relo_relative_q.toRotationMatrix() * Rs[relo_frame_local_index].transpose();
        ROS_DEBUG_STREAM("\ncheck Rw1_w2:\n" << Rw1_w2 << "\nRw1_w2_prime:\n" << Rw1_w2_prime);*/

        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;

    }
}

bool Estimator::failureDetection()
{
    //最后一帧tracking上的点的数量是否足够多
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    //ba和bg都不应过大
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    //在world系下的pose的t和z变化如果过大则认为fail
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true;
    }
    //relative pose过大则fail
    //求误差的角度大小，对四元数表示的旋转，delta q有
    //delta q = [1, 1/2 delta theta]
    //delta theta = [delta q]_xyz * 2，弧度制，视情况转为degree
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;//转为degree
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

//获得当前优化参数，按照自定义solver中的排列方式排列
static void get_cur_parameter(solver::Solver& solver, double* cur_x_array) {
    for (auto it : solver.parameter_block_idx) {
        const long addr = it.first;
        const int idx = it.second;
        const int tmp_param_block_size = solver.parameter_block_size[addr];
        ROS_ASSERT_MSG(tmp_param_block_size > 0, "tmp_param_block_size = %d", tmp_param_block_size);
        memcpy( &cur_x_array[idx], reinterpret_cast<double *>(addr), sizeof(double) *(int)tmp_param_block_size);
    }
}

static bool updatePose(const double *x, const double *delta, double *x_plus_delta)
{
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = -_p + dp ;
    q = (_q.inverse() * dq).normalized();//四元数乘法并归一化

    return true;
}

//计算ceres优化iteration轮之后的delta_x, solver要传引用，否则会调用析构函数
static void cal_delta_x(solver::Solver& solver, double* x_before, double* x_after, double* delta_x) {
    for (auto it : solver.parameter_block_idx) {
        const long addr = it.first;
        const int idx = it.second;
        const int tmp_param_block_size = solver.parameter_block_size[addr];
        double tmp_delta_pose_array[SIZE_POSE];
        ROS_DEBUG_STREAM("\nidx: " << idx << ", tmp_param_block_size: " << tmp_param_block_size);
//        ROS_DEBUG_STREAM("\ndelta_x size: " << delta_x.size());

        if (tmp_param_block_size == SIZE_POSE) {
            updatePose(&x_after[idx], &x_before[idx], &delta_x[idx]);
        } else {
            Eigen::Map<const Eigen::VectorXd> x_map(&x_before[idx], tmp_param_block_size);
            Eigen::Map<const Eigen::VectorXd> x_plus_delta_map(&x_after[idx], tmp_param_block_size);
            Eigen::Map<Eigen::VectorXd> delta_x_map(&delta_x[idx], tmp_param_block_size);
            delta_x_map = x_plus_delta_map - x_map;
//            ROS_DEBUG_STREAM("\ndelta_x_map: " << delta_x_map.transpose());
        }
    }
}

//后端非线性优化
//大作业T1.a思路 这里要添加自己的makehessian的代码AddResidualBlockSolver()//类似于marg一样管理所有的factor，只不过，这里的m是WINDOW内所有的landmark，n是所有的P，V，Tbc，td，relopose
//管理方式也是地址->idx,地址->size一样，在添加的时候指定landmark的drop_set为valid，剩下的为非valid
//在最后求解出整个delta x，在solve中用LM评估迭代效果并继续迭代
void Estimator::optimization()
{
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);//Huber损失函数
    loss_function = new ceres::CauchyLoss(1.0);//柯西损失函数

    ceres::Problem problem;

    //自己写的solver

    solver::Solver solver(strategy);
#ifdef CERES_SOLVE

    //添加ceres参数块
    //因为ceres用的是double数组，所以在下面用vector2double做类型装换
    //Ps、Rs转变成para_Pose，Vs、Bas、Bgs转变成para_SpeedBias
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);//ceres里叫参数块，g2o里是顶点和边
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }

    //ESTIMATE_EXTRINSIC!=0则camera到IMU的外参也添加到估计
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    //相机和IMU硬件不同步时估计两者的时间偏差
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }

#else
    //自己写的solver如何固定住外参呢？
//    solver::Solver solver;

#endif



    TicToc t_whole, t_prepare;
    vector2double();

    //用于check维度
    std::unordered_map<long, uint8_t> param_addr_check;//所有param维度
    std::unordered_map<long, uint8_t> landmark_addr_check;//landmark维度
    //1.添加边缘化残差（先验部分）
    size_t size_1=0;
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);//里面设置了上次先验的什么size，现在还不懂

#ifdef CERES_SOLVE
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);

//        /*用于check维度是否正确*/
        for(int i=0; i<last_marginalization_parameter_blocks.size(); ++i) {
            size_t tmp_size = last_marginalization_info->parameter_block_size[reinterpret_cast<long>(last_marginalization_parameter_blocks[i])];
            tmp_size = tmp_size==7 ? 6: tmp_size;
            //这个double*的地址代表的待优化变量的local_size，把每个地址都记录在map中，分配给待优化变量的地址都是连续的
            for(int j=0; j<tmp_size; ++j) {
                param_addr_check[reinterpret_cast<long>(last_marginalization_parameter_blocks[i]) + (double)j * (long) sizeof(long)] = 1;
            }
        }

        //打印prior的Jacobian维度
        ROS_DEBUG("\nlinearized_jacobians (rows, cols) = (%lu, %lu)",
                  last_marginalization_info->linearized_jacobians.rows(), last_marginalization_info->linearized_jacobians.cols());

        size_1 = param_addr_check.size();//76
        ROS_DEBUG("\nprior size1=%lu, param_addr_check.size() = %lu, landmark size: %lu, except landmark size = %lu",
                  size_1, param_addr_check.size(), landmark_addr_check.size(), param_addr_check.size()-landmark_addr_check.size());//landmark_addr_check中多加了个td

#else
        //dropset用于指定求解时需要Schur消元的变量，即landmark
        solver::ResidualBlockInfo *residual_block_info = new solver::ResidualBlockInfo(marginalization_factor, NULL,
                                                                       last_marginalization_parameter_blocks,
                                                                                       vector<int>{});
        solver.addResidualBlockInfo(residual_block_info);
#endif
    }

    //2.添加IMU残差
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        //两帧KF之间IMU积分时间过长的不进行优化（可能lost？）
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);//这里的factor就是残差residual，ceres里面叫factor

#ifdef CERES_SOLVE
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);

        //check维度
        long addr = reinterpret_cast<long>(para_Pose[i]);
        if(param_addr_check.find(addr) == param_addr_check.end()) {
            ROS_DEBUG("\nIMU add para_Pose[%d]", i);
            for(int k=0; k<SIZE_POSE-1; ++k) {
                param_addr_check[addr + (long)k * (long)sizeof(long)] = 1;
            }
        }

        addr = reinterpret_cast<long>(para_SpeedBias[i]);
        if(param_addr_check.find(addr) == param_addr_check.end()) {
            ROS_DEBUG("\nIMU add para_SpeedBias[%d]", i);
            for(int k=0; k<SIZE_SPEEDBIAS; ++k) {
                param_addr_check[addr + (long) k * (long) sizeof(long)] = 1;
            }
        }

        addr = reinterpret_cast<long>(para_Pose[j]);
        if(param_addr_check.find(addr) == param_addr_check.end()) {
            ROS_DEBUG("\n IMU add para_Pose[%d]", j);
            for(int k=0; k<SIZE_POSE-1; ++k) {
                param_addr_check[addr + (long) k * (long) sizeof(long)] = 1;
            }
        }

        addr = reinterpret_cast<long>(para_SpeedBias[j]);
        if(param_addr_check.find(addr) == param_addr_check.end()) {
            ROS_DEBUG("\n IMU add para_SpeedBias[%d]", j);
            for (int k = 0; k < SIZE_SPEEDBIAS; ++k) {
                param_addr_check[addr + (long) k * (long) sizeof(long)] = 1;
            }
        }
#else
        solver::ResidualBlockInfo *residual_block_info =
                new solver::ResidualBlockInfo(imu_factor, NULL,
                                              vector<double *>{para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]},
                                              vector<int>{});
        solver.addResidualBlockInfo(residual_block_info);
#endif
    }
#ifdef CERES_SOLVE
    size_t size_2 = param_addr_check.size() - size_1;//96
    ROS_DEBUG("\nIMU size2=%lu, param_addr_check.size() = %lu, landmark size: %lu, except landmark size = %lu",
              size_2, param_addr_check.size(), landmark_addr_check.size(), param_addr_check.size()-landmark_addr_check.size());//landmark_addr_check中多加了个td
#endif

    //3.添加视觉残差
    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        //!(至少两次tracking，且最新帧1st的tracking不能算（因为1st可能被marg掉），start_frame最大是[7])
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
 
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        //这个id的feature的第一帧和后面所有的帧分别构建residual block
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point;
            //是否要time offset
            if (ESTIMATE_TD)
            {
                    //对于一个feature，都跟[it_per_id.start_frame]帧进行优化
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
#ifdef CERES_SOLVE
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);

                    //check维度
                    for(int k=0; k<SIZE_POSE-1; ++k) {
                        param_addr_check[reinterpret_cast<long>(para_Pose[imu_i]) + (long)k * (long)sizeof(long)] = 1;
                    }
                    for(int k=0; k<SIZE_POSE-1; ++k) {
                        param_addr_check[reinterpret_cast<long>(para_Pose[imu_j]) + (long)k * (long)sizeof(long)] = 1;
                    }
                    for(int k=0; k<SIZE_POSE-1; ++k) {
                        param_addr_check[reinterpret_cast<long>(para_Ex_Pose[0]) + (long)k * (long)sizeof(long)] = 1;
                    }
                    param_addr_check[reinterpret_cast<long>(para_Feature[feature_index])] = 1;
                    landmark_addr_check[reinterpret_cast<long>(para_Feature[feature_index])] = 1;
                    param_addr_check[reinterpret_cast<long>(para_Td[0])] = 1;

                    /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
#else
                solver::ResidualBlockInfo *residual_block_info = new solver::ResidualBlockInfo(f_td, loss_function,
                                                                                vector<double*>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                               vector<int>{});
                solver.addResidualBlockInfo(residual_block_info);
#endif
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
#ifdef CERES_SOLVE
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
                //check维度
                for(int k=0; k<SIZE_POSE-1; ++k) {
                    param_addr_check[reinterpret_cast<long>(para_Pose[imu_i]) + (long)k * (long)sizeof(long)] = 1;
                }
                for(int k=0; k<SIZE_POSE-1; ++k) {
                    param_addr_check[reinterpret_cast<long>(para_Pose[imu_j]) + (long)k * (long)sizeof(long)] = 1;
                }
                for(int k=0; k<SIZE_POSE-1; ++k) {
                    param_addr_check[reinterpret_cast<long>(para_Ex_Pose[0]) + (long)k * (long)sizeof(long)] = 1;
                }
                param_addr_check[reinterpret_cast<long>(para_Feature[feature_index])] = 1;
                landmark_addr_check[reinterpret_cast<long>(para_Feature[feature_index])] = 1;
#else
                solver::ResidualBlockInfo *residual_block_info = new solver::ResidualBlockInfo(f, loss_function,
                                                                                vector<double*>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                               vector<int>{});
                solver.addResidualBlockInfo(residual_block_info);
#endif
            }
            f_m_cnt++;
        }
    }

#ifdef CERES_SOLVE
    size_t size_3 = param_addr_check.size() - size_1 - size_2;//应该和landmark_addr_check.size一样
    ROS_DEBUG("\nvisual size3=%lu, param_addr_check.size() = %lu, landmark size: %lu, except landmark size = %lu",
              size_3, param_addr_check.size(), landmark_addr_check.size(), param_addr_check.size()-landmark_addr_check.size());//landmark_addr_check中多加了个td
#endif
    ROS_DEBUG("visual measurement count: %d", f_m_cnt);//总的视觉观测个数，观测可能是在不同帧对同一个landmark进行观测，所以可能查过1000，注意与landmark个数进行区分
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());


    //4.添加闭环检测残差，计算滑动窗口中与每一个闭环关键帧的相对位姿，这个相对位置是为后面的图优化(pose graph)准备 或者是 快速重定位(崔华坤PDF7.2节)
    //这里注意relo_pose是Tw2_bi = Tw2_w1 * Tw1_bi
    if(relocalization_info)
    {
        ROS_DEBUG("\nhas relocation blocks");
        //printf("set relocalization factor! \n");
#ifdef CERES_SOLVE
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
#endif
        int retrive_feature_index = 0;
        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            ROS_DEBUG("\nfeature_id: %d", it_per_id.feature_id);
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            ROS_DEBUG("\nmatch_points size: %lu", match_points.size());
            if(start <= relo_frame_local_index)//必须之前看到过
            {
                //1.先在i中match的点中找到可能是现在这个feature的id的index
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)//.z()存的是i，j两帧match上的feature的id
                {
                    retrive_feature_index++;
                }
                ROS_DEBUG("\nrelo here1");
                //2.如果是，则WINDOW内的it_per_id.feature_id这个id的landmark就是被loop上的landmark,取归一化坐标，
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    //pts_j是i帧的归一化平面上的点，这里理解relo_Pose及其重要，relo_Pose实际上是Tw2_bi，视觉重投影是从WINDOW内的start帧的camera(在w2系下)，投影到i帧(在w1系下)，耦合了Tw1_w2
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;//start中的点
                    
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    //relo_Pose是Tw2_bi
#ifdef CERES_SOLVE
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);

                    //check维度
                    for(int k=0; k<SIZE_POSE-1; ++k) {
                        param_addr_check[reinterpret_cast<long>(para_Pose[start]) + (long)k * (long)sizeof(long)] = 1;
                    }
                    for(int k=0; k<SIZE_POSE-1; ++k) {
                        param_addr_check[reinterpret_cast<long>(relo_Pose) + (long)k * (long)sizeof(long)] = 1;
                    }
                    for(int k=0; k<SIZE_POSE-1; ++k) {
                        param_addr_check[reinterpret_cast<long>(para_Ex_Pose[0]) + (long)k * (long)sizeof(long)] = 1;
                    }
                    param_addr_check[reinterpret_cast<long>(para_Feature[feature_index])] = 1;
                    landmark_addr_check[reinterpret_cast<long>(para_Feature[feature_index])] = 1;
#else
                    ROS_DEBUG("\nrelo here2");
                    solver::ResidualBlockInfo *residual_block_info = new solver::ResidualBlockInfo(f, loss_function,
                                                                                                   vector<double*>{para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                                   vector<int>{});
                    solver.addResidualBlockInfo(residual_block_info);
#endif
                    retrive_feature_index++;
                    ROS_DEBUG("\nrelo here3");
                }     
            }
        }
    }
#ifdef CERES_SOLVE
    size_t size_4 = param_addr_check.size() - size_1 - size_2 - size_3;//没有loop时应该为0
    ROS_DEBUG("\nrelocation size_4=%lu, param_addr_check.size() = %lu, landmark size: %lu, except landmark size = %lu",
              size_4, param_addr_check.size(), landmark_addr_check.size(), param_addr_check.size()-landmark_addr_check.size());//landmark_addr_check中多加了个td


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
//    options.trust_region_strategy_type = ceres::DOGLEG;//狗腿算法，与LM较为接近
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;//LM
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;

/*    //获得idx和data
    solver.preMakeHessian();
    solver.makeHessian();

    ROS_DEBUG("delta1");
    int cur_x_size = 1000 + (WINDOW_SIZE + 1) * (SIZE_POSE + SIZE_SPEEDBIAS) + SIZE_POSE + 1 + 100;
    double cur_x_array[cur_x_size], cur_x_array_before[cur_x_size];
    get_cur_parameter(solver, cur_x_array);
    memcpy(cur_x_array_before, cur_x_array, sizeof(double) * cur_x_size);
    Eigen::Map<Eigen::VectorXd> cur_x(cur_x_array, solver.m + solver.n);//cur_x_array变了，cur_x才会变
    const Eigen::VectorXd cur_x_before = cur_x;*/

    ROS_DEBUG("delta2");
    ceres::Solve(options, &problem, &summary);
    ROS_DEBUG("delta3");

/*    get_cur_parameter(solver, cur_x_array);
    double delta_x_ceres[cur_x_size];
    Eigen::Map<Eigen::VectorXd> delta_x_ceres_map(delta_x_ceres, solver.m + solver.n);

    cal_delta_x(solver, cur_x_array_before, cur_x_array, delta_x_ceres);
    ROS_DEBUG_STREAM("\ncur_x before: " << cur_x_before.transpose() <<
                          "\ncur_x after: " << cur_x.transpose() <<
                          "\ndelta_x_ceres: "<< delta_x_ceres_map.transpose() <<
                          "\ndelta_x_ceres.norm(): " << delta_x_ceres_map.norm() <<
                          ",    delta_x_ceres.squaredNorm(): " << delta_x_ceres_map.squaredNorm());*/

    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("\nIterations : %d", static_cast<int>(summary.iterations.size()));

#else //手写求解器求解
    ROS_DEBUG("delta1");
    solver.preMakeHessian();
    solver.makeHessian();

    ROS_DEBUG("delta1");
    int cur_x_size = 1000 + (WINDOW_SIZE + 1) * (SIZE_POSE + SIZE_SPEEDBIAS) + SIZE_POSE + 1 + 100;
    double cur_x_array[cur_x_size], cur_x_array_before[cur_x_size];
    get_cur_parameter(solver, cur_x_array);
    memcpy(cur_x_array_before, cur_x_array, sizeof(double) * cur_x_size);
    Eigen::Map<Eigen::VectorXd> cur_x(cur_x_array, solver.m + solver.n);//cur_x_array变了，cur_x才会变
    const Eigen::VectorXd cur_x_before = cur_x;

    ROS_DEBUG("delta2");
    TicToc t_solver;
    solver.solve(NUM_ITERATIONS);
    double vins_finish_time = t_solver.toc();
    solver_time_sum_ += vins_finish_time;
    ++solve_times_;
    ROS_DEBUG("\nmy solver costs: %f ms, iter nums: %d, avg_solve_time: %f ms, solver_time_sum_: %f, solve_times_: %f",
              vins_finish_time, NUM_ITERATIONS, solver_time_sum_/solve_times_, solver_time_sum_, solve_times_);

    get_cur_parameter(solver, cur_x_array);
    double delta_x[cur_x_size];
    Eigen::Map<Eigen::VectorXd> delta_x_map(delta_x, solver.m + solver.n);

    ROS_DEBUG("delta3");

    cal_delta_x(solver, cur_x_array_before, cur_x_array, delta_x);
    TicToc t_print;
    ROS_DEBUG_STREAM(
//                          "\ncur_x before: " << cur_x_before.transpose() <<
//                          "\ncur_x after: " << cur_x.transpose() <<
                          "\ndelta_x: "<< delta_x_map.transpose() <<
                          "\ndelta_x.norm(): " << delta_x_map.norm() <<
                          ",    delta_x.squaredNorm(): " << delta_x_map.squaredNorm());
    ROS_DEBUG("\nprint costs: %f ms", t_print.toc());
#endif



    // 防止优化结果在零空间变化，通过固定第一帧的位姿(如何固定，free，gauge，fix？)
    double2vector();

    //边缘化处理
    //如果次新帧是关键帧，将边缘化最老帧，及其看到的路标点和IMU数据，将其转化为先验：
    TicToc t_whole_marginalization;//如marg掉xi_2，则需要处理跟xi_2相关的先验信息，IMU信息，视觉信息
    //1. marg 最老帧[0]
    if (marginalization_flag == MARGIN_OLD)
    {
        //new_marg_info，编译器生成默认构造函数
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();
        //1） 把上一次先验项中的残差项（尺寸为 n） 传递给当前先验项，并从中取出需要丢弃的状态量；
        // (这一步不是多此一举？第2步中的parameter_block不会保证marg掉para_Pose[0]和para_SpeedBias[0]吗？)
        //并不是，因为里面要求Jacobian，所以必须按照标准的格式传入才能求出正确的Jacobian
        if (last_marginalization_info)//如果不是第一帧（因为第一帧没有marg掉之后生成的先验matrix）
        {
            //如果上次的先验中有本次需要marg的变量，则添加到drop_set中
            vector<int> drop_set;//本次被marg的参数的idx
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            // 用上次marg的info初始化这次的marg_factor，再加到这次的info中，info管理marg的操作，
            // ceres只管调用marg_factor，不直接管info（当然factor需要info来初始化，所以是marg_factor管info，而不是ceres）
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);

            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
//            ROS_DEBUG_STREAM("\nadd MARGIN_OLD last_marginalization_info\n " <<
//                             "\ncost_function->num_residuals(): " << marginalization_factor->num_residuals() <<
//                             "\ncost_function->parameter_block_sizes().size: " << marginalization_factor->parameter_block_sizes().size());
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        //2） 将滑窗内第 0 帧和第 1 帧间的 IMU 预积分因子（ pre_integrations[1]）放到marginalization_info 中
        // （不理解为什么para_Pose[1], para_SpeedBias[1]也要marg）
        {
            if (pre_integrations[1]->sum_dt < 10.0)//两帧间时间间隔少于10s，过长时间间隔的不进行marg
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                //drop_set表示只marg掉[0][1]，即P0,V0（虽然只drop[0][1]，但是evaluate需要所有的变量来计算Jacobian，所以还是全部传进去）
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                   vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                               vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
//                ROS_DEBUG_STREAM("\nadd imu_factor\n " <<
//                                 "\ncost_function->num_residuals(): " << imu_factor->num_residuals() <<
//                                 "\ncost_function->parameter_block_sizes().size: " << imu_factor->parameter_block_sizes().size());
            }
        }

        //3） 挑 选 出 第 一 次 观 测 帧 为 第 0 帧 的 路 标 点 ， 将 对 应 的 多 组 视 觉 观 测 放 到marginalization_info 中，
        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)//只选择从[0]开始tracking的点
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;//old中的2d坐标

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});//只drop掉[0](P0)和[3](tracking始于old的landmark)
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                    vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                               vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
//                        ROS_DEBUG_STREAM("\nadd ProjectionFactor\n " <<
//                                         "\ncost_function->num_residuals(): " << f->num_residuals() <<
//                                         "\ncost_function->parameter_block_sizes().size: " << f->parameter_block_sizes().size());
                    }
                }
            }
        }

        //得到 上次的先验、IMU测量、视觉观测(都是factor)对应的参数块(parameter_blocks)、雅可比矩阵(jacobians)、残差值(residuals)，
        //与[0]有关的待优化变量存放于parameter_block_data中
        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("\npre marginalization %f ms", t_pre_margin.toc());

        //多线程计算在X0处的整个先验项的参数块，雅可比矩阵和残差值
        //5、多线程构造先验项舒尔补AX=b的结构，在X0处线性化计算Jacobian和残差
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("\nmarginalization %f ms", t_margin.toc());

        //用marg之后的待优化参数去生成新的last_marg_info和last_marg_parameter_blocks供下一次使用
        //6.调整参数块在下一次窗口中对应的位置（往前移一格），注意这里是指针，后面slideWindow中会赋新值，这里只是提前占座
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            //让指针指向
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];

            double* tmp_para_ptr = para_Pose[i-1];
            double* tmp_ptr = addr_shift[reinterpret_cast<long>(para_Pose[i])];
//            for(int j=0; j<7; ++j) {
//                ROS_DEBUG("\npara_Pose[%d] data: %f", i, *tmp_para_ptr);
//                ++tmp_para_ptr;
//                ROS_DEBUG("\naddr_shift[reinterpret_cast<long>(para_Pose[%d])] data: %f", i, *tmp_ptr);
//                ++tmp_ptr;
//            }
        }

        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;//保存此次marg info
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    //2. marg最新帧1st：仅marg掉视觉pose
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                //只drop掉2nd的视觉pose（IMU部分是在slideWindow内继承和delete的）
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);
//                ROS_DEBUG_STREAM("\nin MARGIN_SECOND_NEW add last_marginalization_info\n " <<
//                                 "\ncost_function->num_residuals(): " << marginalization_factor->num_residuals() <<
//                                 "\ncost_function->parameter_block_sizes().size: " << marginalization_factor->parameter_block_sizes().size());
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    //看不懂啥意思，后面不是还要操作slideWindow吗，这里搞地址干什么？
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    ROS_DEBUG("whole marginalization costs: %f ms", t_whole_marginalization.toc());
    
    ROS_DEBUG("whole time for ceres: %f ms", t_whole.toc());
}

//滑窗之后，WINDOW的最后两个Ps，Vs，Rs，Bas，Bgs相同，无论是old还是new，
//因为后面预积分要用最新的预积分初值，所以为了保证窗口内有11个观测，使最后两个相同
void Estimator::slideWindow()
{
    TicToc t_margin;
    //把最老的帧冒泡移到最右边，然后delete掉，在new一个新的对象出来
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0].stamp.toSec();
        back_R0 = Rs[0];//
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)//循环完成也就冒泡完成到最右侧
            {
                Rs[i].swap(Rs[i + 1]);//世界系下old冒泡

                std::swap(pre_integrations[i], pre_integrations[i + 1]);//每一帧的预积分old冒泡

                dt_buf[i].swap(dt_buf[i + 1]);//各种buf也冒泡
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];//最后一个是 Headers[WINDOW_SIZE-1] = Headers[WINDOW_SIZE]
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            //这一步是为了 new IntegrationBase时传入最新的预积分的初值acc_0, gyr_0，ba，bg，所以必须要强制等于最新的
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            //冒泡到最右边之后把对应的都delete&new或者clear掉
            delete pre_integrations[WINDOW_SIZE];//delete掉，并new新的预积分对象出来
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();
//            ROS_DEBUG("marg_flag: %d, before marg, all_image_frame.size(): %lu, WINDOW_SIZE: %d",
//                      marginalization_flag, all_image_frame.size(), WINDOW_SIZE);
            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);//t_0是最老帧的时间戳，marg_old时删掉了帧，但是marg2nd的时候没有动，但是在process时候加进来了，说明all_image_frame应该是在增长的
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;
 
                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);//erase掉从开始到被marg掉的old之间所有的帧[begin(), it_0)
                all_image_frame.erase(t_0);//erase掉old帧

            }
            slideWindowOld();//管理feature(landmark)
//            ROS_DEBUG("marg_flag: %d, after marg, all_image_frame.size(): %lu, WINDOW_SIZE: %d",
//                      marginalization_flag, all_image_frame.size(), WINDOW_SIZE);
        }
    }
    //如果2nd不是KF则直接扔掉1st的visual测量，并在2nd基础上对1st的IMU进行预积分，window前面的都不动
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
//            ROS_DEBUG("marg_flag: %d, before marg, all_image_frame.size(): %lu, WINDOW_SIZE: %d",
//                      marginalization_flag, all_image_frame.size(), WINDOW_SIZE);
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)//对最新帧的img对应的imu数据进行循环
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);//2nd对1st进行IMU预积分
                //imu数据保存，相当于一个较长的KF，eg：
                //     |-|-|-|-|-----|
                //                ↑
                //            这段img为1st时，2nd不是KF，扔掉了这个2nd的img，但buf了IMU数据，所以这段imu数据较长
                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }
            //相对世界系的预积分需要继承过来
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
//            ROS_DEBUG("marg_flag: %d, after marg, all_image_frame.size(): %lu, WINDOW_SIZE: %d",
//                      marginalization_flag, all_image_frame.size(), WINDOW_SIZE);
        }

    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        //Twb * Tbc = Twc
        //0：被marg掉的T_marg,1：新的第[0]帧的T_new
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);//为什么要转移深度？landmark不是删除了吗？
    }
    else
        f_manager.removeBack();
}

void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;//与old frame loop上的WINDOW内的帧(j帧)的时间戳
    relo_frame_index = _frame_index;//j帧的帧号
    match_points.clear();
    match_points = _match_points;//i帧中与j帧中match上的点在i帧中的归一化(x,y,id)
    //Tw1_bi=Tw1_b_old
    prev_relo_t = _relo_t;//i帧pose
    prev_relo_r = _relo_r;
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
        if(relo_frame_stamp == Headers[i].stamp.toSec())
        {
            relo_frame_local_index = i;//j帧在WINDOW中的下标
            relocalization_info = 1;
            for (int j = 0; j < SIZE_POSE; j++)
                //注意，这不是赋地址，而是new了一个新的优化变量的内存，relo_Pose虽然赋初值时为Tw2_bj，但是实际上作用是Tw2_bi
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}

