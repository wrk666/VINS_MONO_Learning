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
#include "../parameters.h"

namespace solve {

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
    Solver(uint8_t strategy): method_(kLM), iterations_(1), strategy_(strategy), lm_alpha_(1), mem_allocated_(false){};
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
    std::unordered_map<long, double *> parameter_block_data_backup;//<优化变量内存地址,数据>

    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;//之前看到的帖子说这是在marg过程中反解出的线性化点的参数值x0

    Eigen::MatrixXd linearized_jacobians;//线性化点处的Jacobian
    Eigen::VectorXd linearized_residuals;//线性化点处的residual
    const double eps = 1e-8;


    bool solve();
    void solveLinearSystem();/// 解线性方程
    bool updatePose(const double *x, const double *delta, double *x_plus_delta);
    bool updateStates(double weight) ;/// 更新状态变量
    bool backupStates();//回滚状态变量
    bool rollbackStates(); // 有时候 update 后残差会变大，需要退回去，重来
    double computeChi() const;
    void computeLambdaInitLM();/// 计算LM算法的初始Lambda
    void addLambdatoHessianLM();/// Hessian 对角线加上或者减去  Lambda
    void removeLambdaHessianLM();
    Eigen::MatrixXd pcgSolver(const MatXX &A, const VecX &b, int maxIter);/// PCG 迭代线性求解器
    bool isGoodStepInLM();/// LM 算法中用于判断 Lambda 在上次迭代中是否可以，以及Lambda怎么缩放

    enum SolveMethod
    {
        kLM = 0,
        kDOGLEG = 1
    };
    SolveMethod method_;
    int iterations_;//迭代轮数
    double currentChi_;

    //LM参数
    double currentLambda_;//LM中的阻尼因子，DOGLEG中的radius
    double stopThresholdLM_;    // LM 迭代退出阈值条件
    std::string file_name_;
    int try_iter_;
    int false_theshold_;//每轮迭代允许的最大失败次数
    double ni_;       //strategy3控制 Lambda 缩放大小
    double lm_alpha_; //strategy2更新使用的alpha


    //求解结果
//    VecX delta_x_rr_;
//    VecX delta_x_mm_;

    //DL参数
    double radius_;
    double epsilon_1_, epsilon_2_, epsilon_3_;
//    double dl_alpha_;

    /// 整个信息矩阵
    Eigen::MatrixXd Hessian_;
    Eigen::VectorXd b_;
    Eigen::VectorXd delta_x_;

    //多留100的余量，这个是成员变量，在程序中是局部变量，放在栈区，不需要手动释放内存，因为它会在其作用域结束时自动被销毁
    const int x_size_ = 1000 + (WINDOW_SIZE + 1) * (SIZE_POSE + SIZE_SPEEDBIAS) + SIZE_POSE + 1 + 100;
    double cur_x_array_[1000 + (WINDOW_SIZE + 1) * (SIZE_POSE + SIZE_SPEEDBIAS) + SIZE_POSE + 1 + 100];
    double delta_x_array_[1000 + (WINDOW_SIZE + 1) * (SIZE_POSE + SIZE_SPEEDBIAS) + SIZE_POSE + 1 + 100];

    //是否已调用preMakeHessian分配过内存
    bool mem_allocated_;

    uint8_t strategy_;



    double *makeHessian_time_sum_;//这个需要手撸才能统计时间，ceres无法统计
    double *makeHessian_times_;

private:
    bool get_cur_parameter(double* cur_x_array);
    double dlComputeDenom(const Eigen::VectorXd& h_dl, const Eigen::VectorXd& h_gn,
                                  const double dl_alpha, const double dl_beta) const;
    };

}




#endif //CATKIN_WS_SOLVE_H
