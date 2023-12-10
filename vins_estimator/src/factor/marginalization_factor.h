#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;//优化变量数据，sizes每个优化变量块的变量大小，以IMU残差为例，为[7,9,7,9]
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

class MarginalizationInfo
{
  public:
    //未显式定义构造函数，使用到该类时，编译器才会生成默认构造函数
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);//加残差块相关信息(优化变量、待marg的变量)
    void preMarginalize();//计算每个残差对应的Jacobian，并更新parameter_block_data
    void marginalize();//pos为所有变量维度，m为需要marg掉的变量，n为需要保留的变量
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

};

//这里不要求指定residual和输入的维度
class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
