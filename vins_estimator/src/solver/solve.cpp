//
// Created by wrk on 2023/12/22.
//
#include <iostream>
#include <fstream>

#include "solve.h"
#include "../parameters.h"

//关于变量地址管理之类的可以直接搬marg的

namespace solve{

/*solver_info相关函数*/

//计算每个残差，对应的Jacobian，并更新 parameter_block_data
void ResidualBlockInfo::Evaluate()
{
    //每个factor的残差块总维度 和 残差块具体size
    //residual总维度，先验=last n=76，IMU=15，Visual=2
    residuals.resize(cost_function->num_residuals());
    //有td时，先验factor为13(9*1+6*10+6+1)，IMU factor为4(7,9,7,9)，每个feature factor size=5(7,7,7,1)
    //无td时             12                           4                                  4
    std::vector<int> block_sizes = cost_function->parameter_block_sizes();

/*    ROS_DEBUG_STREAM("\ncost_function->num_residuals(): " << cost_function->num_residuals() <<
                      "\ncost_function->parameter_block_sizes().size: " << cost_function->parameter_block_sizes().size());
for(int i=0; i<cost_function->parameter_block_sizes().size(); ++i) {
    ROS_DEBUG("\nparameter_block_sizes()[%d]: %d", i, cost_function->parameter_block_sizes()[i]);
}*/
    raw_jacobians = new double *[block_sizes.size()];//二重指针，指针数组
    jacobians.resize(block_sizes.size());

    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
    {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();//二重指针,是为了配合ceres的形参 double** jacobians，看不懂，给data还能够操作地址？？
        //dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
    }
    //虚函数，调用的是基类自己实现的Evaluate，即分别是MarginalizationFactor、IMUFactor 和 ProjectionTdFactor(或ProjectionFactor)的Evaluate()函数
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    //std::vector<int> tmp_idx(block_sizes.size());
    //Eigen::MatrixXd tmp(dim, dim);
    //for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
    //{
    //    int size_i = localSize(block_sizes[i]);
    //    Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
    //    for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
    //    {
    //        int size_j = localSize(block_sizes[j]);
    //        Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
    //        tmp_idx[j] = sub_idx;
    //        tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
    //    }
    //}
    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
    //std::cout << saes.eigenvalues() << std::endl;
    //ROS_ASSERT(saes.eigenvalues().minCoeff() >= -1e-6);

    //这里使用的是CauchyLoss（应该是计算一个scale对residuals进行加权，先不细看，TODO：了解CauchyLoss等loss函数的意义）
    if (loss_function)
    {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();
        //loss_function 为 robust kernel function，in：sq_norm， out：rho  out[0] = rho(sq_norm),out[1] = rho'(sq_norm), out[2] = rho''(sq_norm),
        loss_function->Evaluate(sq_norm, rho);//求取鲁棒核函数关于||f(x)||^2的一二阶导数
        //printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0))
        {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else
        {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);//求根公式求方程的根
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
        {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}

Solver::~Solver()
{
    ROS_DEBUG("destractor here1");
    //new出来的是在堆上的内存，需要手动delete释放；malloc的内存使用free来释放
    if(mem_allocated_) {
        for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
            delete[] it->second;
        ROS_DEBUG("destractor here2");
        for (auto it = parameter_block_data_backup.begin(); it != parameter_block_data_backup.end(); ++it)
            delete[] it->second;
    }
    ROS_DEBUG("destractor here3");
    //这个不能在这delete放，因为ceres要用
//    for (int i = 0; i < (int)factors.size(); i++)
//    {
//
//        delete[] factors[i]->raw_jacobians;
//        ROS_DEBUG("destractor here31");
//        delete[] factors[i]->cost_function;
//        ROS_DEBUG("destractor here32");
//        delete[] factors[i];
//        ROS_DEBUG("destractor here33");
//    }
    ROS_DEBUG("destractor here4");
}

void Solver::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    factors.emplace_back(residual_block_info);

    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;//每个factor的待优化变量的地址
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();//待优化变量的维度

    //parameter_blocks.size
    //有td时，先验factor为13(9*1+6*10+6+1)，IMU factor为4(7,9,7,9)，每个feature factor size=5(7,7,7,1)
    //无td时             12                           4                                  4
    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)
    {
        double *addr = parameter_blocks[i];
        int size = parameter_block_sizes[i];//待优化变量的维度
        //map没有key时会新建key-value对
        parameter_block_size[reinterpret_cast<long>(addr)] = size;//global size <优化变量内存地址,localSize>
//        ROS_DEBUG("in addResidualBlockInfo size: %d", size);
    }

    //需要 marg 掉的变量
    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)
    {
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];//获得待marg变量的地址
        //要marg的变量先占个key的座，marg之前将m放在一起，n放在一起
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;//local size <待边缘化的优化变量内存地址,在parameter_block_size中的id>，
    }
}


void Solver::preMakeHessian()
{
//    ROS_INFO_STREAM("\nfactors.size(): " << factors.size());
    int i=0;
    ROS_DEBUG("factors size=%lu, landmark size=%lu", factors.size(), factors.size()-2); //始于[0]帧的landmark
    for (auto it : factors)
    {
//        ROS_INFO_STREAM("\nin preMarginalize i: "<< ++i);  //很大，能到900多，说明[0]观测到了很多landmark
        it->Evaluate();//计算每个factor的residual和Jacobian

        //如果完成过就只计算Jacobian和residual(里面已经耦合了sqrt_info，所以直接H=J^T*J,不用J^T*W*J)，不用再new内存,重复调用只是为了计算新的Jacobian和residual
        if(mem_allocated_) {
            continue;
        }

        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes(); //residual总维度，先验=last n=76，IMU=15，Visual=2
/*        测试地址转换之后还能否转换回来
        long tmp_addr = reinterpret_cast<long>(it->parameter_blocks[0]);
        double* tmp_pt = reinterpret_cast<double *>(tmp_addr);
        ROS_DEBUG_STREAM("\nraw double* = " << it->parameter_blocks[0] << ",   cast to long= " << tmp_addr << ",   back to double* = " << tmp_pt);*/

        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);//parameter_blocks是vector<double *>，存放的是数据的地址

            int size = block_sizes[i];
            //如果优化变量中没有这个数据就new一片内存放置
            if (parameter_block_data.find(addr) == parameter_block_data.end())
            {
                double *data = new double[size];
                //dst,src
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data[addr] = data;

                //数据备份
                double *data_backup = new double[size];
                memcpy(data_backup, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data_backup[addr] = data_backup;
            }
        }
    }
    mem_allocated_ = true;
}

int Solver::localSize(int size) const
{
    return size == 7 ? 6 : size;
}

int Solver::globalSize(int size) const
{
    return size == 6 ? 7 : size;
}

//线程函数
static void* ThreadsConstructA(void* threadsstruct)
{
    ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);

    //遍历该线程分配的所有factors，这factor可能是任何一个factor
    for (auto it : p->sub_factors)
    {
        //遍历该factor中的所有参数块,比如IMU factor传入的是vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]}
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
            if (size_i == 7) //对于pose来说，是7维的,最后一维为0，这里取左边6
                size_i = 6;
            //只提取local size部分，对于pose来说，是7维的,最后一维为0，这里取左边6维;对于其他待优化变量，size_i不变，取全部jacobian
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if (size_j == 7)
                    size_j = 6;
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                //主对角线
                if (i == j)
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    //非主对角线
                else
                {
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    return threadsstruct;
}

static bool updatePose(const double *x, const double *delta, double *x_plus_delta)
{
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    //数组转四元数
    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);//Jacobian和residual都是按照6维来处理的，但是更新rotation时需要按照四元数来更新

    p = _p + dp;
    q = (_q * dq).normalized();//四元数乘法并归一化

    return true;
}

void Solver::makeHessian()
{
    int pos = 0;//Hessian矩阵整体维度
    //it.first是要被marg掉的变量的地址,将其size累加起来就得到了所有被marg的变量的总localSize=m
    //marg的放一起，共m维，remain放一起，共n维
    for (auto &it : parameter_block_idx)
    {
        it.second = pos;//也算是排序1
        pos += localSize(parameter_block_size[it.first]);//PQ7为改为6维
    }

    m = pos;//要被marg的变量的总维度

    int tmp_n = 0;
    //与[0]相关总维度
    for (const auto &it : parameter_block_size)
    {
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end())//将不在drop_set中的剩下的维度加起来，这一步实际上算的就是n
        {
            parameter_block_idx[it.first] = pos;//排序2
            tmp_n += localSize(it.second);
            pos += localSize(it.second);
        }
    }

    n = pos - m;//remain变量的总维度，这样写建立了n和m间的关系，表意更强
    ROS_DEBUG("\nn: %d, tmp_n: %d", n, tmp_n);

    ROS_DEBUG("\nSolver, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());

    TicToc t_summing;
    Eigen::MatrixXd A(pos, pos);//总系数矩阵
    Eigen::VectorXd b(pos);//总误差项
    A.setZero();
    b.setZero();
    Hessian_.resize(pos,pos);
    b_.resize(pos);
    delta_x_.resize(pos);

//构建信息矩阵可以多线程构建
/*    //single thread
for (auto it : factors)
{
    //J^T*J
    for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
    {
        int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];//要被marg的second=0
        int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
        Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);//remain变量的初始jacobian
        for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
        {
            int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
            int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
            Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);//marg变量的初始jacobian
            //主对角线，注意这里是+=，可能之前别的变量在这个地方已经有过值了，所以要+=
            if (i == j)
                A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
            //非主对角线
            else
            {
                A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
            }
        }
        b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;//J^T*e
    }
}
ROS_INFO("summing up costs %f ms", t_summing.toc());*/


    //multi thread
    TicToc t_thread_summing;
    pthread_t tids[NUM_THREADS];//4个线程构建
    //携带每个线程的输入输出信息
    ThreadsStruct threadsstruct[NUM_THREADS];
    //将先验约束因子平均分配到4个线程中
    int i = 0;
    for (auto it : factors)
    {
        threadsstruct[i].sub_factors.push_back(it);
        i++;
        i = i % NUM_THREADS;
    }
    //将每个线程构建的A和b加起来
    for (int i = 0; i < NUM_THREADS; i++)
    {
        TicToc zero_matrix;
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos,pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        threadsstruct[i].parameter_block_size = parameter_block_size;//marg里的block_size，4个线程共享
        threadsstruct[i].parameter_block_idx = parameter_block_idx;
        int ret = pthread_create( &tids[i], NULL, ThreadsConstructA ,(void*)&(threadsstruct[i]));//参数4是arg，void*类型，取其地址并强制类型转换
        if (ret != 0)
        {
            ROS_WARN("pthread_create error");
            ROS_BREAK();
        }
    }
    //将每个线程构建的A和b加起来
    for( int i = NUM_THREADS - 1; i >= 0; i--)
    {
        pthread_join( tids[i], NULL );//阻塞等待线程完成，这里的A和b的+=操作在主线程中是阻塞的，+=的顺序是pthread_join的顺序
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }
    //ROS_DEBUG("thread summing up costs %f ms", t_thread_summing.toc());
    //ROS_INFO("A diff %f , b diff %f ", (A - tmp_A).sum(), (b - tmp_b).sum());

    Hessian_ = A;
    b_ = -b;

    //反解出J和e
    TicToc t_solve_J;
    TicToc t_SelfAdjoint;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    ROS_DEBUG("\nt_SelfAdjoint cost: %f ms", t_SelfAdjoint.toc());

    TicToc t_jacobians;
    Eigen::VectorXd S_sqrt = S.cwiseSqrt();//开根号
    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    ROS_DEBUG("\nt_jacobians cost: %f ms", t_jacobians.toc());

    TicToc t_residuals;
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
    ROS_DEBUG("\nt_residuals cost: %f ms", t_residuals.toc());

    ROS_DEBUG("\nt_solve_J cost: %f ms", t_solve_J.toc());//22ms,太久了
}

std::vector<double *> Solver::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{
    std::vector<double *> keep_block_addr;//remain的优化变量的地址
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    for (const auto &it : parameter_block_idx)
    {
        if (it.second >= m)//如果是remain部分
        {
            keep_block_size.push_back(parameter_block_size[it.first]);
            keep_block_idx.push_back(parameter_block_idx[it.first]);
            keep_block_data.push_back(parameter_block_data[it.first]);
            keep_block_addr.push_back(addr_shift[it.first]);//待优化变量的首地址需要不变，但是首地址对应的变量是P0，需要在slideWindow中被冒泡到后面delete掉
        }
    }
    ROS_DEBUG("keep_block_addr[0] long addr: %ld, [1] long addr: %ld",
              reinterpret_cast<long>(keep_block_addr[0]), reinterpret_cast<long>(keep_block_addr[1]));
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
}


//只更新状态量p，q，v，ba，bg，λ，注意prior不是状态量，虽然在待优化变量中，但是其residual是跟状态量有关，Jacobian在一轮优化中不变，
//这里更新状态的目的是因为计算chi时会用到residual，而residual和状态量有关，而先验的residual更新：f' = f + J*δxp，其中δxp=x-x0,也跟状态量x有关，
//但是因为在先验factor在Evaluate时会计算residual，所以不用手动更新，只需要更新最核心的x即可。其他的factor相同。
bool Solver::updateStates(double weight) {
    int array_size = 1000 + (WINDOW_SIZE + 1) * (SIZE_POSE + SIZE_SPEEDBIAS) + SIZE_POSE + 1 + 100;
    double used_delta_x[array_size];
    if(weight != -1.0)
        std::transform(delta_x_array_, delta_x_array_ + array_size, used_delta_x, [weight](double x) { return x * weight; });//对delta_x加权
    else
        memcpy(used_delta_x, delta_x_array_, sizeof(double) * array_size);
    //使用idx来找对应的param
    double cur_x_array[1000 + (WINDOW_SIZE + 1) * (SIZE_POSE + SIZE_SPEEDBIAS) + SIZE_POSE + 1 + 100];
    for (auto it : parameter_block_idx){
        const long addr = it.first;
        const int idx = it.second;
        const int tmp_param_block_size = parameter_block_size[addr];
/*        ROS_DEBUG_STREAM("\nidx: " << idx << ",  tmp_param_block_size: " << tmp_param_block_size);*/
        //保存一份待优化变量，和delta_x进行数量级对比
        memcpy( &cur_x_array[idx], reinterpret_cast<double *>(addr), sizeof(double) *(int)SIZE_POSE);
        if(tmp_param_block_size == SIZE_POSE) {
/*            Eigen::Map<Eigen::VectorXd> delta_pose {&delta_x_array_[idx], tmp_param_block_size};
            Eigen::Map<Eigen::VectorXd> tmp_pose {reinterpret_cast<double *>(addr), tmp_param_block_size};
            for(int i=0;i<tmp_param_block_size;++i){
                tmp_pose[i] = *(reinterpret_cast<double *>(addr + sizeof(double)*i));
                delta_pose[i] = delta_x_array_[idx+i];
            }
            ROS_DEBUG_STREAM("\npose before update: " << tmp_pose.transpose() << "\ndelta_pose: " << delta_pose.transpose());*/

            updatePose(reinterpret_cast<double *>(addr), &delta_x_array_[idx], reinterpret_cast<double *>(addr));//TODO:这个backup应该可以用parameter_block_data替代

/*            ROS_DEBUG_STREAM("\npose after update: " << tmp_pose.transpose());*/
        } else {
            Eigen::Map<const Eigen::VectorXd> x{parameter_block_data_backup[addr], tmp_param_block_size};
            Eigen::Map<const Eigen::VectorXd> delta_x{&delta_x_array_[idx], tmp_param_block_size};
            Eigen::Map<Eigen::VectorXd> x_plus_delta{reinterpret_cast<double *>(addr), tmp_param_block_size};
            /*ROS_DEBUG_STREAM("\nother parameters before update: " << x_plus_delta.transpose() << "\ndelta_x: " << delta_x.transpose());*/
            x_plus_delta = x + delta_x;
            /*ROS_DEBUG_STREAM("\nother parameters after update: " << x_plus_delta.transpose());*/
        }
    }
    // 初始化Eigen向量
    Eigen::Map<Eigen::VectorXd> cur_x(cur_x_array, m+n);
    ROS_DEBUG_STREAM("\ncur_x: " << cur_x.transpose());
    preMakeHessian();//计算更新后的Jacobian和residual
    return true;
}

//备份状态量
bool Solver::backupStates() {
    for (auto it : parameter_block_idx){
        const long addr = it.first;
        const int tmp_param_block_size = parameter_block_size[addr];
        memcpy(parameter_block_data_backup[addr], reinterpret_cast<double *>(addr), sizeof(double) * (int)tmp_param_block_size);
    }
    return true;
}

//回滚状态量
bool Solver::rollbackStates() {
    for (auto it : parameter_block_idx){
        const long addr = it.first;
        const int tmp_param_block_size = parameter_block_size[addr];
        memcpy(reinterpret_cast<double *>(addr), parameter_block_data_backup[addr], sizeof(double) * (int)tmp_param_block_size);
    }
    preMakeHessian();//计算更新后的Jacobian和residual
    return true;
}

bool Solver::get_cur_parameter(double* cur_x_array) {
    for (auto it : parameter_block_idx) {
        const long addr = it.first;
        const int idx = it.second;
        const int tmp_param_block_size = parameter_block_size[addr];
        ROS_ASSERT_MSG(tmp_param_block_size > 0, "tmp_param_block_size = %d", tmp_param_block_size);
        memcpy( &cur_x_array[idx], reinterpret_cast<double *>(addr), sizeof(double) *(int)tmp_param_block_size);
    }
    return true;
}

//在ResidualBlockInfo::Evaluate()中调用的多态Evaluate()函数后已经考虑了loss_function对Jacobian和residual的加权
//分别计算先验和其他factor的chi
double Solver::computeChi() const{
    //先验的residual维度
    size_t prior_dim = SIZE_SPEEDBIAS + (SIZE_POSE-1) * WINDOW_SIZE + (SIZE_POSE-1);
    if(ESTIMATE_TD){
        prior_dim+=1;
    }
    double tmpChi = 0;
    for (auto it : factors){
        //TODO：先验的也改成正常更新，按照参数也是PQV来理解先验的话，就应该是正常更新，而不是使用norm(),后面看看对不对
        double this_Chi = it->residuals.transpose() * it->residuals;
        tmpChi += this_Chi;

//        if(it->residuals.size()==prior_dim) {
//            double this_Chi = it->residuals.norm();
//            tmpChi += this_Chi;
///*            ROS_DEBUG_STREAM("\nprior factor, this_Chi= " << this_Chi
//                              << ",   residuals size: " << it->residuals.size()
//                              << ", residuals: " << it->residuals.transpose());*/
//        } else {
//            double this_Chi = it->residuals.transpose() * it->residuals;
//            tmpChi += this_Chi;
///*            ROS_DEBUG_STREAM("\nother factor, this_Chi= " << this_Chi
//                              << ",   residuals size: " << it->residuals.size()
//                              << ",   residuals: " << it->residuals.transpose());*/
//        }
    }
    ROS_DEBUG_STREAM("\nhere tmpChi= " << tmpChi);
    return tmpChi;
}

/// LM
void Solver::computeLambdaInitLM() {
      if(strategy_ == 1) {
        currentChi_ = computeChi();
          double maxDiagonal = 0;
          ulong size = Hessian_.cols();
          assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
          for (ulong i = 0; i < size; ++i) {
              maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal);//取H矩阵的最大值，然后*涛
          }
          double tau = 1e1;//[1e-8,1] tau越小，△x越大
          currentLambda_ = tau * maxDiagonal;
    } else if(strategy_ == 2) {
        // set a large value, so that first updates are small steps in the steepest-descent direction
        currentChi_ = computeChi();
        currentLambda_ = 1e16;
//        currentLambda_ = 1e-3;
      } else if(strategy_ == 3) {
          ni_ = 2.;
          currentLambda_ = -1.;
          currentChi_ = computeChi();

          stopThresholdLM_ = 1e-6 * currentChi_;          // 迭代条件为 误差下降 1e-6 倍

          double maxDiagonal = 0;
          ulong size = Hessian_.cols();
          assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
          for (ulong i = 0; i < size; ++i) {
              maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal);//取H矩阵的最大值，然后*涛
          }
//    double tau = 1e-5;
          double tau = 1e-15;//[1e-8,1] tau越小，△x越大//////////////////////////////////
          currentLambda_ = tau * maxDiagonal;
          ROS_DEBUG_STREAM("\nin computeLambdaInitLM currentChi_= " << currentChi_
                                                                    << ",  init currentLambda_=" << currentLambda_
                                                                    << ",  maxDiagonal=" << maxDiagonal);
      }
}

void Solver::addLambdatoHessianLM() {
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (ulong i = 0; i < size; ++i) {
        if(strategy_==1) {
            Hessian_(i, i) += currentLambda_ * Hessian_(i, i); //理解: H(k+1) = H(k) + λ H(k) = (1+λ) H(k) 策略1
        } else if(strategy_==2 || strategy_==3) {
            Hessian_(i, i) += currentLambda_;
        }
    }
}

void Solver::removeLambdaHessianLM() {
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    // TODO:: 这里不应该减去一个，数值的反复加减容易造成数值精度出问题？而应该保存叠加lambda前的值，在这里直接赋值
    for (ulong i = 0; i < size; ++i) {

        if(strategy_==1) {
            Hessian_(i, i) /= 1.0 + currentLambda_;//H回退: H(k) = 1/(1+λ) * H(k+1)，策略1
        } else if(strategy_==2 || strategy_==3) {
            Hessian_(i, i) -= currentLambda_;
        }
    }
}

//Nielsen的方法，分母直接为L，判断\rho的符号
bool Solver::isGoodStepInLM() {
    bool ret = false;
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    if(strategy_==1) {
        double tempChi = computeChi();
        ulong size = Hessian_.cols();
        MatXX diag_hessian(MatXX::Zero(size, size));
        for(ulong i = 0; i < size; ++i) {
            diag_hessian(i, i) = Hessian_(i, i);
        }
        double scale = delta_x_.transpose() * (currentLambda_ * diag_hessian * delta_x_ + b_);//scale就是rho的分母
        double rho = (currentChi_ - tempChi) / scale;//计算rho
        // update currentLambda_
        double epsilon = 0.75;
        double L_down = 9.0;
        double L_up = 11.0;
        if(rho > epsilon && isfinite(tempChi)) {
            currentLambda_ = std::max(currentLambda_ / L_down, 1e-7);
            currentChi_ = tempChi;
            ret = true;
            ROS_DEBUG("choose L_down");
        } else {
            currentLambda_ = std::min(currentLambda_ * L_up, 1e7);
            ret = false;
            ROS_DEBUG("choose L_up");
        }
        ROS_DEBUG("\nstrategy1: currentLambda_: %e,rho: %f, currentChi_: %e, tempChi: %e, scale: %e",
                  currentLambda_, rho, currentChi_, tempChi, scale);
        ROS_DEBUG_STREAM("\nret = " << ret <<",  rho>0 = " << (rho>0) <<",  rho: " << rho <<
                                    ",   delta_x_.squaredNorm(): " << delta_x_.squaredNorm() <<
                                    ",  delta_x_: " << delta_x_.transpose()<< "\nb_.norm(): " << b_.norm() );
//        ROS_DEBUG_STREAM("\nb_: " << b_.transpose());

    } else if(strategy_==2) {
        // 统计所有的残差
        double tempChi_p_h = 0.0, tempChi_p_alpha_h = 0.0;
        tempChi_p_h = computeChi();

        double alpha_up = b_.transpose() * delta_x_;
        double alpha_down = (tempChi_p_h - currentChi_) / 2. + 2. * alpha_up;
        double alpha_tmp = alpha_up / alpha_down;

        double scale = 0;
        scale = fabs((alpha_tmp * delta_x_).transpose() * (currentLambda_ * (alpha_tmp * delta_x_) + b_));
        scale += 1e-3;    // make sure it's non-zero :)

        //回滚刚才更新的x=x+△x，使用α重新更新x=x + α*△x，并重新计算残差和chi
        rollbackStates();
        updateStates(alpha_tmp);
        preMakeHessian();//更新状态之后就要更新reidual
        tempChi_p_alpha_h = computeChi();

        double rho = (currentChi_ - tempChi_p_alpha_h) / scale;

        if (rho > 0 && isfinite(tempChi_p_alpha_h)) { // last step was good, 误差在下降
            currentLambda_ = std::max(currentLambda_ / (1 + alpha_tmp), 1e-7);
            lm_alpha_ = alpha_tmp;
            currentChi_ = tempChi_p_alpha_h;
            ret = true;
        } else {
            currentLambda_ = currentLambda_ + fabs(tempChi_p_alpha_h - currentChi_) / (2 * alpha_tmp);
            ret = false;
        }
        ROS_DEBUG("\nstrategy2: currentLambda_: %e,alpha_tmp: %e, rho: %f, currentChi_: %e, tempChi_p_h: %e, tempChi_p_alpha_h: %e, scale: %e",
                  currentLambda_, alpha_tmp, rho, currentChi_, tempChi_p_h, tempChi_p_alpha_h, scale);
        ROS_DEBUG_STREAM("\nret = " << ret <<",  rho>0 = " << (rho>0) <<",  rho: " << rho <<
                                ",   delta_x_.squaredNorm(): " << delta_x_.squaredNorm() << ",  delta_x_: " << delta_x_.transpose()
                                << "\nb_.norm(): " << b_.norm() << ",  b_: " << b_.transpose());
    } else if(strategy_==3) {

        double scale = 0;
        scale = fabs(delta_x_.transpose() * (currentLambda_ * delta_x_ + b_));
        scale += 1e-3;    // make sure it's non-zero :)

        // 统计更新后的所有的chi()
        double tempChi = computeChi();

        double rho = (currentChi_ - tempChi) / scale;
        if (rho > 0 && isfinite(tempChi))   // last step was good, 误差在下降
        {
            double alpha = 1. - pow((2 * rho - 1), 3);//更新策略跟课件里面一样
            //TODO：这个ceres里面没有限制上限为2/3
            alpha = std::min(alpha, 2. / 3.);
            double scaleFactor = (std::max)(1. / 3., alpha);
            currentLambda_ *= scaleFactor;//课程里面应该是μ，需要绘制曲线
            ni_ = 2;  //v
            currentChi_ = tempChi;
            ret = true;
        } else {//如果\rho<0则增大阻尼μ，减小步长
            currentLambda_ *= ni_;
            ni_ *= 2;//2这个值越大，λ增长越快
            ret = false;
        }
        ROS_DEBUG("\nstrategy3: currentLambda_: %e, ni_: %e, rho: %f, currentChi_: %e, tempChi: %e, scale: %e",
                  currentLambda_, ni_, rho, currentChi_, tempChi, scale);
        ROS_DEBUG_STREAM("\nret = " << ret <<",  rho>0 = " << (rho>0) <<",  rho: " << rho << ",   delta_x_.squaredNorm(): " << delta_x_.squaredNorm() << ",  delta_x_: " << delta_x_.transpose()
                                    << "\nb_.norm(): " << b_.norm() << ",  b_: " << b_.transpose());
/*        FILE *fp_lambda = fopen(file_name_.data(), "a");
        fprintf(fp_lambda, "%d, %f\n", try_iter_, currentLambda_);
        fflush(fp_lambda);
        fclose(fp_lambda);*/
    }
    ROS_DEBUG("\n%d record lambda finish\n", try_iter_);

    return ret;
}

/*
* @brief conjugate gradient with perconditioning
*
*  the jacobi PCG method  共轭梯度法
*  知乎有一篇帖子是针对PCG进行改进的，能减少迭代次数：对角线预处理和不完备的Cholesky预处理
 * https://zhuanlan.zhihu.com/p/521753443
*/
Eigen::MatrixXd Solver::pcgSolver(const MatXX &A, const VecX &b, int maxIter = -1) {
    assert(A.rows() == A.cols() && "PCG solver ERROR: A is not a square matrix");
    int rows = b.rows();
    int n = maxIter < 0 ? rows : maxIter;
    VecX x(VecX::Zero(rows));
    MatXX M_inv = A.diagonal().asDiagonal().inverse();//取对角线阵的逆矩阵
    VecX r0(b);  // initial r = b - A*0 = b
    VecX z0 = M_inv * r0;
    VecX p(z0);
    VecX w = A * p;
    double r0z0 = r0.dot(z0);
    double alpha = r0z0 / p.dot(w);
    VecX r1 = r0 - alpha * w;
    int i = 0;
    double threshold = 1e-5 * r0.norm(); //比例调大可以提高阈值，放宽停止条件
    while (r1.norm() > threshold && i < n) {
        i++;
        VecX z1 = M_inv * r1;
        double r1z1 = r1.dot(z1);
        double belta = r1z1 / r0z0;
        z0 = z1;
        r0z0 = r1z1;
        r0 = r1;
        p = belta * p + z1;
        w = A * p;
        alpha = r1z1 / p.dot(w);
        x += alpha * p;
        r1 -= alpha * w;
    }
    ROS_DEBUG("\nPCG iter times: %d, n: %d, r1.norm(): %f, threshold: %f", i, n, r1.norm(), threshold);
    return x;
}

/*Solve Hx = b, we can use PCG iterative method or use sparse Cholesky*/
//TODO:使用PCG迭代而非SVD分解求解
void Solver::solveLinearSystem() {
//method1：直接求逆求解
//    delta_x_ = Hessian_.inverse() * b_;
//    delta_x_ = H.ldlt().solve(b_);

//Schur消元求解，marg掉landmark
//    //method2：schur消元求解
//    //求解Hx=b，marg不用求出△x，所以不用对方程组求解，但是优化时需要求解出整个△x
//    Eigen::MatrixXd Amm_solver = 0.5 * (Hessian_.block(0, 0, m, m) + Hessian_.block(0, 0, m, m).transpose());
//    Eigen::VectorXd bmm_solver = b_.segment(0, m);
//    Eigen::MatrixXd Amr_solver = Hessian_.block(0, m, m, n);
//    Eigen::MatrixXd Arm_solver = Hessian_.block(m, 0, n, m);
//    Eigen::MatrixXd Arr_solver = Hessian_.block(m, m, n, n);
//    Eigen::VectorXd brr_solver = b_.segment(m, n);
//
//
//
//    //求Amm_solver^(-1)
//    double scale = Amm_solver.maxCoeff();
//    Amm_solver /= scale;
//    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm_solver);
//    if (saes.info() == Eigen::Success) {
//        ROS_DEBUG("\nsaes Eigenvalue computation success.");
//    } else {
//        ROS_WARN("\nsaes Eigenvalue computation failed");
//    }
//    //这个1e-4应该是个经验值，不懂数值稳定性，暂不研究
////    ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());
//    size_t tmp_size = saes.eigenvalues().size();
//    ROS_DEBUG("\nhere saes min eigenvalue: %e, max eigenvalue: %e, saes.eigenvalues.size():%lu",
//              saes.eigenvalues().minCoeff(), saes.eigenvalues().maxCoeff(), tmp_size);
//    ROS_DEBUG_STREAM("\nsaes.eigenvalues(): " << saes.eigenvalues().transpose());
//
//    //marg的矩阵块求逆,特征值分解求逆更快
//    Eigen::MatrixXd Amm_inv_solver = saes.eigenvectors()
//                              * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal()
//                              * saes.eigenvectors().transpose();
//    Amm_inv_solver = scale * Amm_inv_solver;//恢复
//
//    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(Amm_solver);
//    if (lu_decomp.isInvertible()) {
//        ROS_DEBUG("\nAmm_solver is invertible.");
//    } else {
//        ROS_WARN("\nAmm_solver is not invertible.");
//    }
///*    Eigen::MatrixXd Amm_inv_solver = Amm_solver.inverse();*/
//    Eigen::MatrixXd tmpA_solver = Arm_solver * Amm_inv_solver;
//
//    //step1: Schur补
//    Eigen::MatrixXd Arr_schur = Arr_solver - tmpA_solver * Amr_solver;
//    Eigen::VectorXd brr_schur = brr_solver - tmpA_solver * bmm_solver;
//
//    ROS_DEBUG("here1");
//
//    // step2: solve Hpp * delta_x = bpp
//    //1 TODO：没有lambda，不知道怎么用PCG solver来求解△xrr，先用上面的SVD求逆方法，
//    //2 TODO：数值稳定性目前不懂，先不assert看看会有什么效果(可能需要rescale)
//    double scale_solver = Arr_schur.maxCoeff();
//    Arr_schur /= scale_solver;
//    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes_solver(Arr_schur);//缩放
//    if (saes_solver.info() == Eigen::Success) {
//        ROS_DEBUG("\nsaes_solver Eigenvalue computation success.");
//    } else {
//        ROS_WARN("\nsaes_solver Eigenvalue computation failed");
//    }
////    ROS_ASSERT_MSG(saes_solver.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes_solver.eigenvalues().minCoeff());
//    Eigen::MatrixXd Arr_schur_inv = saes_solver.eigenvectors()
//                                    * Eigen::VectorXd((saes_solver.eigenvalues().array() > eps).select(saes_solver.eigenvalues().array().inverse(), 0)).asDiagonal()
//                                    * saes_solver.eigenvectors().transpose();
//    Arr_schur_inv = scale_solver * Arr_schur_inv;//恢复
//    //这个可能会崩，不知道为啥
//    size_t tmp_size_solver = saes_solver.eigenvalues().size();
//    ROS_DEBUG("\nhere saes_solver min eigenvalue: %e, max eigenvalue: %e, saes_solver.eigenvalues.size():%lu",
//              saes_solver.eigenvalues().minCoeff(), saes_solver.eigenvalues().maxCoeff(), tmp_size_solver);
//    ROS_DEBUG_STREAM("\nsaes_solver.eigenvalues(): " << saes_solver.eigenvalues().transpose());
//
//    ROS_DEBUG("\nhere2");
//
//    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp_schur(Arr_schur);
//    if (lu_decomp_schur.isInvertible()) {
//        ROS_DEBUG("\nArr_schur is invertible.");
//    } else {
//        ROS_DEBUG("\nArr_schur is not invertible.");
//    }
//
///*    Eigen::MatrixXd Arr_schur_inv = Arr_schur.inverse();*/
//    ROS_DEBUG("\nhere21");
//    Eigen::VectorXd delta_x_rr = Arr_schur_inv * brr_schur;
//    ROS_DEBUG("\nhere22");
//    Eigen::VectorXd delta_x_mm = Amm_inv_solver * (bmm_solver - Amr_solver * delta_x_rr);
//    ROS_DEBUG("\nhere23");
//    delta_x_.tail(n) = delta_x_rr;
//    ROS_DEBUG("\nhere24");
//    delta_x_.head(m) = delta_x_mm;
//    ROS_DEBUG("\nhere25");


/*  判断矩阵是否可逆  20ms左右
TicToc t_judge_invertible;
Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp_Hessian(Hessian_);
if (lu_decomp_Hessian.isInvertible()) {
    ROS_DEBUG("\nHessian_ is invertible.");
} else {
    ROS_WARN("\nHessian_ is not invertible.");
}
ROS_DEBUG("\nt_judge_invertible cost %f ms", t_judge_invertible.toc());*/

    TicToc t_solve_equation;
//    delta_x_ = Hessian_.ldlt().solve(b_);
    int pcg_iter_num = Hessian_.rows()+1; // PCG迭代次数,原来给的是rows()*2
    delta_x_ = pcgSolver(Hessian_, b_, pcg_iter_num);  //0.3ms
    ROS_DEBUG("\nin solveLinearSystem solve equation cost %f ms, pcg_iter_num: %d", t_solve_equation.toc(), pcg_iter_num);
    memcpy(delta_x_array_, delta_x_.data(), sizeof(double) * (int)delta_x_.size());//转为数组，供状态更新使用(delta_x_太大)
    ROS_DEBUG_STREAM("\nhere3 solve complete, delta_x_.size()=" << delta_x_.size() << ", delta_x_.norm()= "<< delta_x_.norm()
                                                                << ",  delta_x_.squaredNorm()=" << delta_x_.squaredNorm() );
    ROS_DEBUG_STREAM("\ndelta_x_:" << delta_x_.transpose());

}

double Solver::dlComputeDenom(const Eigen::VectorXd& h_dl, const Eigen::VectorXd& h_gn,
                              const double dl_alpha, const double dl_beta) const {
    if(h_dl == h_gn)
        return currentChi_;
    else if(h_dl == b_ / b_.norm() * radius_)
        return ( radius_ * (2.0 * (dl_alpha * b_).norm() - radius_) ) / (2 * dl_alpha); //由于取norm()，所以b_符号不变
    else
        return 0.5 * dl_alpha * pow(1 - dl_beta, 2) * b_.squaredNorm() + dl_beta * (2 - dl_beta) * currentChi_;
}

//求解器相关函数
bool Solver::solve() {
    if(method_==kLM) {
        //    ROS_DEBUG("\nlinearized_jacobians (rows, cols) = (%lu, %lu)", linearized_jacobians.rows(), linearized_jacobians.cols());

        TicToc t_LM_init;
        // LM 初始化
        computeLambdaInitLM();
        ROS_DEBUG("\nsolver computeLambdaInitLM %f ms", t_LM_init.toc());
        // LM 算法迭代求解
        bool stop = false;
        int iter = 0;
        //尝试的lambda次数
        try_iter_ = 0;

        if(strategy_==1) {
            false_theshold_ = 10;
        } else if(strategy_==2) {
            false_theshold_ = 10;
        } else if (strategy_==3) {
            false_theshold_ = 6;
        }
        ROS_DEBUG("\nstrategy: %d, false_theshold_: %d", strategy_, false_theshold_);
        //保存LM阻尼阻尼系数lambda
/*    file_name_ = "./lambda.csv";
FILE *tmp_fp = fopen(file_name_.data(), "a");
fprintf(tmp_fp, "iter, lambda\n");
fflush(tmp_fp);
fclose(tmp_fp);*/

        TicToc t_LM_iter;
        while (!stop && (iter < iterations_)) {
            ROS_DEBUG_STREAM("\niter: " << iter << " , chi= " << currentChi_ << " , Lambda= " << currentLambda_ << "\n");
            bool oneStepSuccess = false;
            int false_cnt = 0;
            while (!oneStepSuccess)  // 不断尝试 Lambda, 直到成功迭代一步
            {
                ++try_iter_;
                // setLambda
                TicToc t_addLambdatoHessianLM;
                addLambdatoHessianLM();//0.01ms
                ROS_DEBUG("\naddLambdatoHessianLM cost %f ms", t_addLambdatoHessianLM.toc());

                // 第四步，解线性方程 H X = B
                TicToc t_solveLinearSystem;
                solveLinearSystem();//8ms
                ROS_DEBUG("\nsolveLinearSystem cost %f ms", t_solveLinearSystem.toc());

                TicToc t_removeLambdaHessianLM;
                removeLambdaHessianLM();//0.005ms
                ROS_DEBUG("\nremoveLambdaHessianLM cost %f ms", t_removeLambdaHessianLM.toc());


                // 优化退出条件1： delta_x_ 很小则退出 原来是1e-6

                if (delta_x_.squaredNorm() <= 1e-10 || false_cnt > false_theshold_) {
                    stop = true;
                    ROS_DEBUG("\ndelta_x too small: %e, or false_cnt=%d > 10  break", delta_x_.squaredNorm(), false_cnt);//都是在这出去的
                    break;
                } else {
                    ROS_DEBUG_STREAM("\ndelta_x_ squaredNorm matched: " << delta_x_.squaredNorm() << ",  delta_x_ size: " <<delta_x_.size()
                                                                        << ", delta_x: " << delta_x_.transpose() );
                }

                TicToc t_updateStates;
                updateStates(-1.0);//0.08ms 1.更新状态 2.preMakeHessian()计算新的residual
                ROS_DEBUG("\nupdateStates cost %f ms", t_updateStates.toc());

                // 判断当前步是否可行以及 LM 的 lambda 怎么更新
                oneStepSuccess = isGoodStepInLM();//误差是否下降
                // 后续处理
                if (oneStepSuccess) {
                    TicToc t_backupStates;
                    backupStates();//若求解成功则备份当前更新的状态量  0.03ms
                    ROS_DEBUG("\nbackupStates cost %f ms", t_backupStates.toc());

                    // 在新线性化点 构建 hessian
                    TicToc t_makeHessian;
                    makeHessian();
                    double makeHessian_finish_time = t_makeHessian.toc();
                    makeHessian_time_sum_ += makeHessian_finish_time;
                    ++makeHessian_times_;
                    ROS_DEBUG("\nmakeHessian cost: %f ms, avg_makeHessian_time: %f ms, makeHessian_time_sum_: %f, makeHessian_times_: %f",
                              makeHessian_finish_time, makeHessian_time_sum_/makeHessian_times_, makeHessian_time_sum_, makeHessian_times_);
                    // TODO:: 这个判断条件可以丢掉，条件 b_max <= 1e-12 很难达到，这里的阈值条件不应该用绝对值，而是相对值
//                double b_max = 0.0;
//                for (int i = 0; i < b_.size(); ++i) {
//                    b_max = max(fabs(b_(i)), b_max);
//                }
//                // 优化退出条件2： 如果残差 b_max 已经很小了，那就退出
//                stop = (b_max <= 1e-12);
                    false_cnt = 0;
                } else {
                    false_cnt++;
                    TicToc t_rollbackStates;
                    rollbackStates();   // 误差没下降，回滚 0.05ms
                    ROS_DEBUG("\nrollbackStates cost %f ms", t_rollbackStates.toc());
                }
                ROS_DEBUG("\nfalse_cnt: %d", false_cnt);
            }
            iter++;
            // 优化退出条件3： currentChi_ 跟第一次的chi2相比，下降了 1e6 倍则退出
            if (sqrt(currentChi_) <= stopThresholdLM_) {
                ROS_DEBUG("\ncurrentChi_ decrease matched break condition");
                stop = true;
            }
        }
        ROS_DEBUG("\nLM iterate %f ms", t_LM_iter.toc());
/*    std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
std::cout << "   makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;*/
    } else if(method_==kDOGLEG) {
        ROS_DEBUG("\nDL iter num: %d", iterations_);
        //1.初始化 radius，g=b=J^Te
        radius_ = 1;
        epsilon_1_ = 1e-10;
        //向量无穷范数：cwiseAbs："coordinate-wise"（逐元素）取绝对值，colwise().sum()计算每行的绝对值之和，maxCoeff()得最大值
        bool use_last_hessian = true;
        bool stop = (linearized_residuals.cwiseAbs().colwise().sum().maxCoeff() < epsilon_3_) ||
                    (b_.cwiseAbs().colwise().sum().maxCoeff() <= epsilon_1_);
        int iter = 0;
        double rho, tempChi;
        double rho_numerator, rho_denominator;
        bool stop_cond_1;
        currentChi_ = computeChi();
        //2.循环  stop=||e||无穷范数<=阈值3 或 ||g||无穷范数<=阈值1
        while (!stop && (iter < iterations_)) {
            ROS_DEBUG_STREAM("\niter: " << iter << " , chi= " << currentChi_ << " , radius_= " << radius_ << "\n");
//            if(!use_last_hessian)
//                makeHessian();
            double dl_alpha = b_.squaredNorm() / (linearized_jacobians * b_).squaredNorm();//都是取二范数平方，就不区分b的符号了
            //3.计算h_sd
            Eigen::VectorXd h_sd = dl_alpha * b_;
            //4.计算g_gn
            solveLinearSystem();//比较耗时
            Eigen::VectorXd h_gn = delta_x_;
            //5.计算h_dl
            Eigen::VectorXd h_dl;
            Eigen::VectorXd a = dl_alpha * h_sd;
            Eigen::VectorXd b = h_gn;
            double c = a.transpose() * (b - a);
            double dl_beta;
            if(c<=0) {
                double tmp_1 = (b-a).squaredNorm();
                dl_beta = (-c + sqrt(pow(c,2) + tmp_1 * (pow(radius_, 2) - a.squaredNorm()))) / tmp_1;
            } else {
                double tmp_1 = pow(radius_, 2) - a.squaredNorm();
                dl_beta = tmp_1 / (c + sqrt(pow(c, 2) + (b-a).squaredNorm() * tmp_1));
            }
            if(b.norm() <= radius_)
                h_dl = h_gn;
            else if(a.norm() >= radius_)
                h_dl = h_sd / h_sd.norm() * radius_;
            else {
                h_dl = a + dl_beta * (h_gn - a);
            }

            //判断是否需要继续迭代
            get_cur_parameter(cur_x_array_);
            Eigen::Map<const Eigen::VectorXd> x{cur_x_array_, x_size_};
            double tmp_1 = h_dl.norm();
            double tmp_2 = epsilon_2_*(x.norm() + epsilon_2_);
            stop_cond_1 = tmp_1 <= tmp_2;
            ROS_DEBUG("\ntmp_1: %f, tmp_2: %f, stop_cond_1: %d", tmp_1, tmp_2, stop_cond_1);
            if(stop_cond_1) {//设为1e-12
                stop = true;
            } else {
                //更新
                updateStates(-1.0);
                tempChi = computeChi();
                rho = 0.0;
                rho_numerator = currentChi_ - tempChi;
                rho_denominator = dlComputeDenom(h_dl, h_gn, dl_alpha, dl_beta);
                rho = rho_numerator / rho_denominator;
                if(rho > 0) {
                    //执行更新，即保存备份
                    backupStates();
                    preMakeHessian();
                    TicToc t_makeHessian;
                    makeHessian();
                    double makeHessian_finish_time = t_makeHessian.toc();
                    makeHessian_time_sum_ += makeHessian_finish_time;
                    ++makeHessian_times_;
                    ROS_DEBUG("\nmakeHessian cost: %f ms, avg_makeHessian_time: %f ms, makeHessian_time_sum_: %f, makeHessian_times_: %f",
                              makeHessian_finish_time, makeHessian_time_sum_/makeHessian_times_, makeHessian_time_sum_, makeHessian_times_);
                    use_last_hessian = false;
                } else {
                    rollbackStates();
                    use_last_hessian = true;
                }
                get_cur_parameter(cur_x_array_);
                if(rho > 0.75) {
                    radius_ = std::max(radius_, 3*h_dl.norm());
                } else {
                    radius_ = 0.5 * radius_;
                    stop = radius_ < epsilon_2_*(x.norm() + epsilon_2_);
                }
            }
            ROS_DEBUG("\nDL: radius_: %e, rho: %f, rho>0 = %d, currentChi_: %e, tempChi: %e, rho_numerator: %e, rho_denominator: %e",
                      radius_, rho, rho>0, currentChi_, tempChi, rho_numerator, rho_denominator);
            ROS_DEBUG_STREAM("\ndelta_x_.squaredNorm(): " << delta_x_.squaredNorm() << ",  delta_x_: " << delta_x_.transpose()
                             << "\nb_.norm(): " << b_.norm() << ",  b_: " << b_.transpose());
            ++iter;
        }
        ROS_DEBUG("\n finish DL iter, stop: %d or iter: %d >= %d", stop, iter, iterations_);
        //6.根据h_dl.norm()判断是否迭代
        // 若迭代则更新x(涉及状态更新，residual计算)，计算rho，根据rho来更新x和raidus
    }
    return true;
}

}
