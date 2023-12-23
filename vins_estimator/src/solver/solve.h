//
// Created by wrk on 2023/12/22.
//

#ifndef CATKIN_WS_SOLVE_H
#define CATKIN_WS_SOLVE_H

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "eigen_types.h"
#include "../utility/tic_toc.h"

namespace solver {

/*定义factor管理类
 * */
const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
            : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;//优化变量数据的地址，sizes每个优化变量块的变量大小，以IMU残差为例，为[7,9,7,9]
    std::vector<int> drop_set;//待边缘化的优化变量id

    double **raw_jacobians;//二重指针,是为了配合ceres的形参 double** jacobians
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;//这个数据结构看不懂，
    Eigen::VectorXd residuals;//残差 IMU:15X1 视觉:2X1

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

class Solver
{
public:
    //未显式定义构造函数，使用到该类时，编译器才会生成默认构造函数
    ~Solver();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);//加残差块相关信息(优化变量、待marg的变量)
    void preMakeHessian();//计算每个残差对应的Jacobian，并更新parameter_block_data
    void makeHessian();//pos为所有变量维度，m为需要marg掉的变量，n为需要保留的变量
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;//所有观测项
    int m, n;//m为要边缘化的变量个数(也就是parameter_block_idx的总localSize，以double为单位，VBias为9，PQ为6，)，n为要保留下来的变量个数
    //parameter_block_size 和 parameter_block_data分别对应block的大小和实际数据
    std::unordered_map<long, int> parameter_block_size; //global size <优化变量内存地址,localSize>
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //local size 排序前是<待边缘化的优化变量内存地址,在parameter_block_size中的id>，排序后是<marg, id>m维  <remain, id>n维
    std::unordered_map<long, double *> parameter_block_data;//<优化变量内存地址,数据>

    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;//之前看到的帖子说这是在marg过程中反解出的线性化点的参数值x0

    Eigen::MatrixXd linearized_jacobians;//线性化点处的Jacobian
    Eigen::VectorXd linearized_residuals;//线性化点处的residual
    const double eps = 1e-8;

    bool solve(int iterations);
    void solveLinearSystem();/// 解线性方程
    void updateStates();/// 更新状态变量
    void rollbackStates(); // 有时候 update 后残差会变大，需要退回去，重来
    void computeLambdaInitLM();/// 计算LM算法的初始Lambda
    void addLambdatoHessianLM();/// Hessian 对角线加上或者减去  Lambda
    void removeLambdaHessianLM();
    bool isGoodStepInLM();/// LM 算法中用于判断 Lambda 在上次迭代中是否可以，以及Lambda怎么缩放
    Eigen::MatrixXd pcgSolver(const MatXX &A, const VecX &b, int maxIter);/// PCG 迭代线性求解器


    double currentLambda_;
    double currentChi_;
    double stopThresholdLM_;    // LM 迭代退出阈值条件
    double ni_;                 //控制 Lambda 缩放大小
    std::string file_name_;
    int try_iter_;

    //求解结果
//    VecX delta_x_rr_;
//    VecX delta_x_mm_;

    /// 整个信息矩阵
    Eigen::MatrixXd Hessian_;
    Eigen::MatrixXd b_;
    Eigen::MatrixXd delta_x_;
};


//
////定义求解器类
//class Solver{
//public:
//    typedef unsigned long ulong;
//    SolverInfo *solver_info_;
//
//    enum class ProblemType {
//        SLAM_PROBLEM,
//        GENERIC_PROBLEM
//    };
//
//    //构造
//    Solver():solver_info_(new solver::SolverInfo()) {};
//
///**
//* 求解此问题
//* @param iterations
//* @return
//*/
//    bool solve(int iterations);
//
//
//
//private:
//
//    /// Solve的实现，解通用问题
//    bool SolveGenericProblem(int iterations);
//
//    /// Solve的实现，解SLAM问题
//    bool SolveSLAMProblem(int iterations);
//
//    /// 设置各顶点的ordering_index
//    void setOrdering();
//
///*    /// set ordering for new vertex in slam problem
//    void AddOrderingSLAM(std::shared_ptr<Vertex> v);*/
//
//    /// 构造大H矩阵
//    void makeHessian();
//
//    /// schur求解SBA  //TODO：先直接求逆，等跑通了再用Schur来求
//    void SchurSBA();
//
//    /// 解线性方程
//    void solveLinearSystem();
//
//    /// 更新状态变量
//    void updateStates();
//
//    void rollbackStates(); // 有时候 update 后残差会变大，需要退回去，重来
//
///*    /// 计算并更新Prior部分
//    void ComputePrior();
//
//    /// 判断一个顶点是否为Pose顶点
//    bool IsPoseVertex(std::shared_ptr<Vertex> v);
//
//    /// 判断一个顶点是否为landmark顶点
//    bool IsLandmarkVertex(std::shared_ptr<Vertex> v);
//
//    /// 在新增顶点后，需要调整几个hessian的大小
//    void ResizePoseHessiansWhenAddingPose(std::shared_ptr<Vertex> v);
//
//    /// 检查ordering是否正确
//    bool CheckOrdering();
//
//    void LogoutVectorSize();
//
//    /// 获取某个顶点连接到的边
//    std::vector<std::shared_ptr<Edge>> GetConnectedEdges(std::shared_ptr<Vertex> vertex);*/
//
//    /// Levenberg
//    /// 计算LM算法的初始Lambda
//    void computeLambdaInitLM();
//
//    /// Hessian 对角线加上或者减去  Lambda
//    void addLambdatoHessianLM();
//
//    void removeLambdaHessianLM();
//
//    /// LM 算法中用于判断 Lambda 在上次迭代中是否可以，以及Lambda怎么缩放
//    bool isGoodStepInLM();
//
//    /// PCG 迭代线性求解器
//    VecX pcgSolver(const MatXX &A, const VecX &b, int maxIter);
//
//    double currentLambda_;
//    double currentChi_;
//    double stopThresholdLM_;    // LM 迭代退出阈值条件
//    double ni_;                 //控制 Lambda 缩放大小
//    std::string file_name_;
//    int try_iter_;
//
//    ProblemType problemType_;
//
//    /// 整个信息矩阵
//    MatXX Hessian_;
//    VecX b_;
//    VecX delta_x_;
//
//    /// 先验部分信息
//    MatXX H_prior_;
//    VecX b_prior_;
//    MatXX Jt_prior_inv_;
//    VecX err_prior_;
//
//    /// SBA的Pose部分
//    MatXX H_pp_schur_;
//    VecX b_pp_schur_;
//    // Heesian 的 Landmark 和 pose 部分
//    MatXX H_pp_;
//    VecX b_pp_;
//    MatXX H_ll_;
//    VecX b_ll_;
//
///*    /// all vertices
//    HashVertex verticies_;
//
//    /// all edges
//    HashEdge edges_;
//
//    /// 由vertex id查询edge
//    HashVertexIdToEdge vertexToEdge_;*/
//
//    /// Ordering related
//    ulong ordering_poses_ = 0;
//    ulong ordering_landmarks_ = 0;
//    ulong ordering_generic_ = 0;
///*    std::map<unsigned long, std::shared_ptr<Vertex>> idx_pose_vertices_;        // 以ordering排序的pose顶点
//    std::map<unsigned long, std::shared_ptr<Vertex>> idx_landmark_vertices_;    // 以ordering排序的landmark顶点
//
//    // verticies need to marg. <Ordering_id_, Vertex>
//    HashVertex verticies_marg_;//需要边缘化的vertex*/
//
//    bool bDebug = false;
//    double t_hessian_cost_ = 0.0;
//    double t_PCGsovle_cost_ = 0.0;
//};

}




#endif //CATKIN_WS_SOLVE_H
