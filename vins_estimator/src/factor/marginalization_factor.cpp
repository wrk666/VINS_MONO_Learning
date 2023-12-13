#include "marginalization_factor.h"

//计算每个残差，对应的Jacobian，并更新 parameter_block_data
void ResidualBlockInfo::Evaluate()
{
    //每个factor的残差块总维度 和 残差块具体size
    //residual总维度，先验=last n=76，IMU=15，Visual=2
    residuals.resize(cost_function->num_residuals());
    //有td时，先验factor为13(9*1+6*10+6+1)，IMU factor为4(7,9,7,9)，每个feature factor size=5(7,7,7,1)
    //无td时             12                           4                                  4
    std::vector<int> block_sizes = cost_function->parameter_block_sizes();

//    ROS_DEBUG_STREAM("\ncost_function->num_residuals(): " << cost_function->num_residuals() <<
//                          "\ncost_function->parameter_block_sizes().size: " << cost_function->parameter_block_sizes().size());
//    for(int i=0; i<cost_function->parameter_block_sizes().size(); ++i) {
//        ROS_DEBUG("\nparameter_block_sizes()[%d]: %d", i, cost_function->parameter_block_sizes()[i]);
//    }
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

MarginalizationInfo::~MarginalizationInfo()
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

void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
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


void MarginalizationInfo::preMarginalize()
{
//    ROS_INFO_STREAM("\nfactors.size(): " << factors.size());
    int i=0;
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

int MarginalizationInfo::localSize(int size) const
{
    return size == 7 ? 6 : size;
}

int MarginalizationInfo::globalSize(int size) const
{
    return size == 6 ? 7 : size;
}

//线程函数
void* ThreadsConstructA(void* threadsstruct)
{
    ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);

    //遍历该线程分配的所有factors，所有观测项
    for (auto it : p->sub_factors)
    {
        //遍历该factor中的所有参数块P0，V0等
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
            if (size_i == 7) //对于pose来说，是7维的,最后一维为0，这里取左边6
                size_i = 6;
            //只提取local size部分，对于pose来说，是7维的,最后一维为0，这里取左边6维
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


void MarginalizationInfo::marginalize()
{
    int pos = 0;
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

    //ROS_DEBUG("marginalization, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());

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
    //
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
        threadsstruct[i].parameter_block_size = parameter_block_size;
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


    /*求Amm的逆矩阵时，为了保证数值稳定性，做了Amm=1/2*(Amm+Amm^T)的运算，Amm本身是一个对称矩阵，所以  等式成立。接着对Amm进行了特征值分解,再求逆，更加的快速*/
    //数值计算中，由于计算机浮点数精度的限制，有时候数值误差可能导致 Amm 不精确地保持对称性。
    //通过将 Amm 更新为其本身与其转置的平均值，可以强制保持对称性，提高数值稳定性。这种对称性维护的策略在数值计算中比较常见。
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

    //ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());

    //marg的矩阵块求逆,特征值分解求逆更快
    Eigen::MatrixXd Amm_inv = saes.eigenvectors()
                            * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal()
                            * saes.eigenvectors().transpose();
    //printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());

    Eigen::VectorXd bmm = b.segment(0, m);
    Eigen::MatrixXd Amr = A.block(0, m, m, n);
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);
    //Shur compliment marginalization，求取边际概率
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;

    //由于Ceres里面不能直接操作信息矩阵，所以这里从信息矩阵中反解出来了Jacobian和residual，而g2o是可以的，ceres里只需要维护H和b
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    //对称阵
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    //对称阵的倒数阵
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();//开根号
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    //从H和b中反解出Jacobian和residual
    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
    //std::cout << A << std::endl
    //          << std::endl;
    //std::cout << linearized_jacobians << std::endl;
    //printf("error2: %f %f\n", (linearized_jacobians.transpose() * linearized_jacobians - A).sum(),
    //      (linearized_jacobians.transpose() * linearized_residuals - b).sum());
}

std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
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

MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* _marginalization_info):marginalization_info(_marginalization_info)
{
    //把marginalization_info中的
    int cnt = 0;
    for (auto it : marginalization_info->keep_block_size)
    {
        mutable_parameter_block_sizes()->push_back(it);//设置输入维度(参数块维度) parameter_block_sizes_
        cnt += it;
    }
    //printf("residual size: %d, %d\n", cnt, n);
    set_num_residuals(marginalization_info->n);//设置输出维度，上一个先验中的remain的size(我猜除了landmark，减掉marg掉的pi,qi,vi共减去15)
//    ROS_DEBUG("\nin MarginalizationFactor last n is: %d", marginalization_info->n);
};

//先验部分的factor，求Jacobian
bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    //printf("internal addr,%d, %d\n", (int)parameter_block_sizes().size(), num_residuals());
    //for (int i = 0; i < static_cast<int>(keep_block_size.size()); i++)
    //{
    //    //printf("unsigned %x\n", reinterpret_cast<unsigned long>(parameters[i]));
    //    //printf("signed %x\n", reinterpret_cast<long>(parameters[i]));
    //printf("jacobian %x\n", reinterpret_cast<long>(jacobians));
    //printf("residual %x\n", reinterpret_cast<long>(residuals));
    //}
    int n = marginalization_info->n;
    int m = marginalization_info->m;
    Eigen::VectorXd dx(n);
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m;//因为当时存的是marg时的idx，是在m后面的，现在单看先验块的话就需要减去m
        //优化后，本次marg前的待优化变量
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        //优化前，上次maerg后的变量，即Jacobian的线性化点x0
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);
        //求优化后的变量与优化前的差，dx即公式中的δxp。
        //IMU block、landmark depth bolck直接相减，而camera pose block的rotation部分需使用四元数计算δxp
        if (size != 7)
            dx.segment(idx, size) = x - x0;
        else
        {
            //translation直接相减
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            //rotation部分：δq=[1, 1/2 delta theta]^T(为何非要取正的？)
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    //更新误差：f' = f + J*δxp
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
    //Jacobian保持不变（FEJ要解决这样做带来的解的零空间变化的问题）
    if (jacobians)
    {

        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
        {
            if (jacobians[i])
            {
                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m;
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                jacobian.setZero();
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
            }
        }
    }
    return true;
}
