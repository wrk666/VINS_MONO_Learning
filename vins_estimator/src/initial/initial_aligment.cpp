#include "initial_alignment.h"
//估计陀螺bias：bg
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;//待求量
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);//迭代器的下一个或上一个分别用next()和prev()获取。
        MatrixXd tmp_A(3, 3);//3*3矩阵，关于bg的J^T*J
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);//qc0_bk^(-1) * qc0_bk+1
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);//Jacobian中的f25
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();//预计分量，从i积分到j，γij
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;

    }
    delta_bg = A.ldlt().solve(b);
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    //估计出bg后需要更新WINDOW中各帧对应的陀螺仪偏置，并重新propogate每一帧的预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;
    //为什么对所有帧重传，那没有更新bg的帧怎么办？
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);//预积分重新传递
    }
}

//Algorithm 1 to find b1 b2
//在半径为G的半球找到切面的一对单位正交基
MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    // 将tmp往a方向和a垂直方向分解为两个垂直的分量(tmp1,tmp2)，
    // a * (a.transpose() * tmp)求的是将tmp往a方向投影获得的分量tmp1，而使用向量减法可获得tmp2=tmp-tmp1
    // tmp2与a垂直，是a的正切空间的一个基向量，归一化之后就是单位基向量b，与a叉乘之后就得第二个单位基向量c
    b = (tmp - a * (a.transpose() * tmp)).normalized();//它是 tmp 减去在 a 方向上的投影，并进行归一化。
    Vector3d tmp_b = a.cross(tmp).normalized();
//    ROS_DEBUG_STREAM("g0: " << g0.transpose() <<
//                          "   a:" << a.transpose() <<
//                          "   b from author's code: " << b.transpose() <<
//                          "   b from my corss multiply: " << tmp_b.transpose());
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm();//求出优化出的重力的方向的重力向量
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;//这里把重力向量参数化为g=||g||g^ + w1b1 + w2b2，重力向量被转化为2DoF的变量，所以2这里指的是w=[w1 w2]^T

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for(int k = 0; k < 4; k++)//一次Refinement总共迭代4次
    {
        MatrixXd lxly(3, 2);//重力向量的正切
        lxly = TangentBasis(g0);//重力向量的正切空间的两个正交的单位基向量
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;


            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;//权值放到待优化变量中去了，正切空间需要乘进来
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;//同理
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);
            VectorXd dg = x.segment<2>(n_state - 3);//取优化出来的w(2,1)权值
            //计算refine之后的重力向量的方向向量(理解为G.norm()=9.81在3个方向上分量的权值，把9.81分解掉)
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            //double s = x(n_state - 1);
    }   
    g = g0;
}

bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1;//速度个数 * 3 + gc0 + scale尺度

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);//j > i

        MatrixXd tmp_A(6, 10);//H(6,10)
        tmp_A.setZero();
        VectorXd tmp_b(6);//b(6,1)
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();//这里的frame_i->second.R是SfM最后输出的Rw_c=Rc0_c，所以对应到公式中的Rbk_c0就要取转置
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;//这个100应该是任意取的s，转化为米制单位
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];//TIC的size为NUM_OF_CAM,单目所以直接取第一个了
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;//Rbk_c0*Rc0_bk+1
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;//βbk_bk+1
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();//信息矩阵设为Identity，不进行加权

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;//J^T*J
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;//J^Tb

        //前6维
        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();
        //后4维
        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    double s = x(n_state - 1) / 100.0;
    ROS_DEBUG("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);//取估计出来的重力向量gc0的方向向量
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)//如果估计出来的重力向量和重力的差范数大于1或者scale<0则认为标定错误
    {
        return false;
    }

    //重力细化
    RefineGravity(all_image_frame, g, x);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 )
        return false;   
    else
        return true;
}

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    //标定陀螺仪的偏置 bg
    solveGyroscopeBias(all_image_frame, Bgs);
    //估计尺度，重力向量和速度
    if(LinearAlignment(all_image_frame, g, x))
        return true;
    else 
        return false;
}
