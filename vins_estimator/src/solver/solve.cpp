//
// Created by wrk on 2023/12/22.
//
#include <iostream>
#include <fstream>

#include "solve.h"

//关于变量地址管理之类的可以直接搬marg的

namespace solver{

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

    //这里使用的是CauchyLoss（应该是计算一个scale对residuals进行加权，先不细看，TODO：了解CauchyLoss灯loss函数的意义）
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
    //ROS_WARN("release marginlizationinfo");

    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete[] it->second;

    for (int i = 0; i < (int)factors.size(); i++)
    {

        delete[] factors[i]->raw_jacobians;

        delete factors[i]->cost_function;

        delete factors[i];
    }
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
        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes(); //residual总维度，先验=last n=76，IMU=15，Visual=2
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
            }
        }
    }
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
void* ThreadsConstructA(void* threadsstruct)
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
    b_ = b;
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


//求解器相关函数
bool Solver::solve(int iterations) {


/*    if (edges_.size() == 0 || verticies_.size() == 0) {
        std::cerr << "\nCannot solve problem without edges or verticies" << std::endl;
        return false;
    }*/

    TicToc t_pre_margin;
    preMakeHessian();
    ROS_DEBUG("solver.solver_info_ pre marginalization %f ms", t_pre_margin.toc());

    //多线程计算在X0处的整个先验项的参数块，雅可比矩阵和残差值
    //5、多线程构造先验项舒尔补AX=b的结构，在X0处线性化计算Jacobian和残差
    TicToc t_margin;
    makeHessian();
    ROS_DEBUG("\nsolver_info_ marginalization %f ms", t_margin.toc());
    ROS_DEBUG("\nlinearized_jacobians (rows, cols) = (%lu, %lu)", linearized_jacobians.rows(), linearized_jacobians.cols());

    TicToc t_solve;

    // LM 初始化
    computeLambdaInitLM();
    // LM 算法迭代求解
    bool stop = false;
    int iter = 0;
    //尝试的lambda次数
    try_iter_ = 0;
    //保存LM阻尼阻尼系数lambda
    file_name_ = "./lambda.csv";
    FILE *tmp_fp = fopen(file_name_.data(), "a");
    fprintf(tmp_fp, "iter, lambda\n");
    fflush(tmp_fp);
    fclose(tmp_fp);

    while (!stop && (iter < iterations)) {
        std::cout << "iter: " << iter << " , chi= " << currentChi_ << " , Lambda= " << currentLambda_
                  << std::endl;
        bool oneStepSuccess = false;
        int false_cnt = 0;
        while (!oneStepSuccess)  // 不断尝试 Lambda, 直到成功迭代一步
        {
            ++try_iter_;
            // setLambda
            addLambdatoHessianLM();
            // 第四步，解线性方程 H X = B
            solveLinearSystem();
            //
            removeLambdaHessianLM();

            // 优化退出条件1： delta_x_ 很小则退出
            if (delta_x_.squaredNorm() <= 1e-6 || false_cnt > 10) {
                stop = true;
                printf("delta_x too small: %f, or false_cnt=%d > 10  break", delta_x_.squaredNorm(), false_cnt);
                break;
            }

            // 更新状态量 X = X+ delta_x
            updateStates();
            // 判断当前步是否可行以及 LM 的 lambda 怎么更新
            oneStepSuccess = isGoodStepInLM();//误差是否下降
            // 后续处理，
            if (oneStepSuccess) {
                // 在新线性化点 构建 hessian
                makeHessian();
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
                rollbackStates();   // 误差没下降，回滚
            }
        }
        iter++;
        // 优化退出条件3： currentChi_ 跟第一次的chi2相比，下降了 1e6 倍则退出
        if (sqrt(currentChi_) <= stopThresholdLM_) {
            printf("currentChi_ decrease matched break condition");
            stop = true;
        }


    }
/*    std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
    std::cout << "   makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;*/
    return true;
}


/*Solve Hx = b, we can use PCG iterative method or use sparse Cholesky*/
void Solver::solveLinearSystem() {
//直接求逆
//    delta_x_ = Hessian_.inverse() * b_;
//    delta_x_ = H.ldlt().solve(b_);

        /*这里进行schur消元求解*/
    /*求Amm的逆矩阵时，为了保证数值稳定性，做了Amm=1/2*(Amm+Amm^T)的运算，Amm本身是一个对称矩阵，所以等式成立。接着对Amm进行了特征值分解,再求逆，更加的快速*/
    //数值计算中，由于计算机浮点数精度的限制，有时候数值误差可能导致 Amm 不精确地保持对称性。
    //通过将 Amm 更新为其本身与其转置的平均值，可以强制保持对称性，提高数值稳定性。这种对称性维护的策略在数值计算中比较常见。
    Eigen::MatrixXd Amm = 0.5 * (Hessian_.block(0, 0, m, m) + Hessian_.block(0, 0, m, m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

    //这个1e-4应该是个经验值，不懂数值稳定性，暂不研究
    ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());
    size_t tmp_size = saes.eigenvalues().size();
    ROS_DEBUG("\nhere saes min eigenvalue: %f, saes.eigenvalues.size():%lu, sigma3: %f, sigma4: %f,  sigma4/sigma3=%f",
              saes.eigenvalues().minCoeff(), tmp_size, saes.eigenvalues()(tmp_size-2), saes.eigenvalues()(tmp_size-1),
              saes.eigenvalues()(tmp_size-1)/saes.eigenvalues()(tmp_size-2));

    //marg的矩阵块求逆,特征值分解求逆更快
    Eigen::MatrixXd Amm_inv = saes.eigenvectors()
                              * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal()
                              * saes.eigenvectors().transpose();
    //printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());

    //求解Hx=b，marg不用求出△x，所以不用对方程组求解，但是优化时需要求解出整个△x
    Eigen::MatrixXd Amm_solver = 0.5 * (Hessian_.block(0, 0, m, m) + Hessian_.block(0, 0, m, m).transpose());
    Eigen::VectorXd bmm_solver = b_.segment(0, m);
    Eigen::MatrixXd Amr_solver = Hessian_.block(0, m, m, n);
    Eigen::MatrixXd Arm_solver = Hessian_.block(m, 0, n, m);
    Eigen::MatrixXd Arr_solver = Hessian_.block(m, m, n, n);
    Eigen::VectorXd brr_solver = b_.segment(m, n);
    Eigen::MatrixXd Amm_inv_solver = Amm_inv;
    Eigen::MatrixXd tmpA_solver = Arm_solver * Amm_inv_solver;

    //step1: Schur补
    Eigen::MatrixXd Arr_schur = Arr_solver - tmpA_solver * Amr_solver;
    Eigen::VectorXd brr_schur = brr_solver - tmpA_solver * bmm_solver;

    ROS_DEBUG("here1");

    // step2: solve Hpp * delta_x = bpp
    //1 TODO：没有lambda，不知道怎么用PCG solver来求解△xrr，先用上面的SVD求逆方法，
    //2 TODO：数值稳定性目前不懂，先不assert看看会有什么效果
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes_solver(Arr_schur);
//    ROS_ASSERT_MSG(saes_solver.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes_solver.eigenvalues().minCoeff());
    Eigen::MatrixXd Arr_schur_inv = saes_solver.eigenvectors()
                                    * Eigen::VectorXd((saes_solver.eigenvalues().array() > eps).select(saes_solver.eigenvalues().array().inverse(), 0)).asDiagonal()
                                    * saes_solver.eigenvectors().transpose();
    //这个可能会崩，不知道为啥
//    tmp_size = saes_solver.eigenvalues().size();
//    ROS_DEBUG("\nhere saes_solver min eigenvalue: %f, saes.eigenvalues.size():%lu, sigma3: %f, sigma4: %f,  sigma4/sigma3=%f",
//              saes_solver.eigenvalues().minCoeff(), tmp_size, saes.eigenvalues()(tmp_size-2), saes_solver.eigenvalues()(tmp_size-1),
//              saes_solver.eigenvalues()(tmp_size-1)/saes_solver.eigenvalues()(tmp_size-2));

    ROS_DEBUG("here2");

    Eigen::VectorXd delta_x_rr = Arr_schur_inv * brr_schur;
    Eigen::VectorXd delta_x_mm = Amm_inv_solver * (bmm_solver - Amr_solver * delta_x_rr);
    delta_x_.tail(n) = delta_x_rr;
    delta_x_.head(m) = delta_x_mm;
    ROS_DEBUG("here3 solve complete");

}

void Solver::updateStates() {
    //要使用idx来找对应的param
    for (auto vertex: verticies_) {
        ulong idx = vertex.second->OrderingId();
        ulong dim = vertex.second->LocalDimension();
        VecX delta = delta_x_.segment(idx, dim);

        // 所有的参数 x 叠加一个增量  x_{k+1} = x_{k} + delta_x
        vertex.second->Plus(delta);
    }
}

void Solver::rollbackStates() {
    for (auto vertex: verticies_) {
        ulong idx = vertex.second->OrderingId();
        ulong dim = vertex.second->LocalDimension();
        VecX delta = delta_x_.segment(idx, dim);

        // 之前的增量加了后使得损失函数增加了，我们应该不要这次迭代结果，所以把之前加上的量减去。
        vertex.second->Plus(-delta);
    }
}

/// LM
void Solver::computeLambdaInitLM() {
    ni_ = 2.;
    currentLambda_ = -1.;
    currentChi_ = 0.0;
    // TODO:: robust cost chi2
    for (auto edge: edges_) {
        currentChi_ += edge.second->Chi2();
    }
    if (err_prior_.rows() > 0)
        currentChi_ += err_prior_.norm();

    stopThresholdLM_ = 1e-6 * currentChi_;          // 迭代条件为 误差下降 1e-6 倍

    double maxDiagonal = 0;
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (ulong i = 0; i < size; ++i) {
        maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal);//取H矩阵的最大值，然后*涛
    }
    double tau = 1e-5;
    currentLambda_ = tau * maxDiagonal;
}

void Solver::addLambdatoHessianLM() {
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (ulong i = 0; i < size; ++i) {
        Hessian_(i, i) += currentLambda_;
    }
}

void Solver::removeLambdaHessianLM() {
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    // TODO:: 这里不应该减去一个，数值的反复加减容易造成数值精度出问题？而应该保存叠加lambda前的值，在这里直接赋值
    for (ulong i = 0; i < size; ++i) {
        Hessian_(i, i) -= currentLambda_;
    }
}

//Nielsen的方法，分母直接为L，判断\rho的符号
bool Solver::isGoodStepInLM() {
    bool ret = false;
    double scale = 0;
    scale = delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
    scale += 1e-3;    // make sure it's non-zero :)

    // recompute residuals after update state
    // 统计所有的残差
    double tempChi = 0.0;
    for (auto edge: edges_) {
        edge.second->ComputeResidual();
        tempChi += edge.second->Chi2();//计算cost
    }

    double rho = (currentChi_ - tempChi) / scale;
    if (rho > 0 && isfinite(tempChi))   // last step was good, 误差在下降
    {
        double alpha = 1. - pow((2 * rho - 1), 3);//更新策略跟课件里面一样
        alpha = std::min(alpha, 2. / 3.);
        double scaleFactor = (std::max)(1. / 3., alpha);
        currentLambda_ *= scaleFactor;//课程里面应该是μ，需要绘制曲线
        ni_ = 2;  //v
        currentChi_ = tempChi;
        ret = true;
    } else {//如果\rho<0则增大阻尼μ，减小步长
        currentLambda_ *= ni_;
        ni_ *= 2;
        ret = false;
    }
    FILE *fp_lambda = fopen(file_name_.data(), "a");
    fprintf(fp_lambda, "%d, %f\n", try_iter_, currentLambda_);
    fflush(fp_lambda);
    fclose(fp_lambda);

    printf("%d record lambda finish\n", try_iter_);

    return ret;
}

/*
* @brief conjugate gradient with perconditioning
*
*  the jacobi PCG method
*
*/
Eigen::MatrixXd Solver::pcgSolver(const MatXX &A, const VecX &b, int maxIter = -1) {
    assert(A.rows() == A.cols() && "PCG solver ERROR: A is not a square matrix");
    int rows = b.rows();
    int n = maxIter < 0 ? rows : maxIter;
    VecX x(VecX::Zero(rows));
    MatXX M_inv = A.diagonal().asDiagonal().inverse();
    VecX r0(b);  // initial r = b - A*0 = b
    VecX z0 = M_inv * r0;
    VecX p(z0);
    VecX w = A * p;
    double r0z0 = r0.dot(z0);
    double alpha = r0z0 / p.dot(w);
    VecX r1 = r0 - alpha * w;
    int i = 0;
    double threshold = 1e-6 * r0.norm();
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
    return x;
}

}
