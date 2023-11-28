#include "utility.h"

Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};//真正z轴方向
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();//这是计算从ng1->ng2的旋转Rng2_ng1=Rgw_gc0
    double yaw = Utility::R2ypr(R0).x();//计算Rgw_gc0中在yaw方向上需要的旋转
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;//-yaw即抵消在yaw方向上的旋转，只剩下pitch和roll方向的旋转，保证跟重力的z轴方向相同，不管yaw
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}
