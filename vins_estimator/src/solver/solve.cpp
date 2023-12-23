//
// Created by wrk on 2023/12/22.
//
#include <iostream>
#include <fstream>

#include "solve.h"

//关于变量地址管理之类的可以直接搬marg的

namespace lm_strategy{

    bool Solve::solve(int iterations) {


        if (edges_.size() == 0 || verticies_.size() == 0) {
            std::cerr << "\nCannot solve problem without edges or verticies" << std::endl;
            return false;
        }

        TicToc t_solve;
        // 统计优化变量的维数，为构建 H 矩阵做准备
        setOrdering();
        // 遍历edge, 构建 H = J^T * J 矩阵
        makeHessian();
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
        std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
        std::cout << "   makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;
        return true;
    }


    void Solve::setOrdering() {

        // 每次重新计数
        ordering_poses_ = 0;
        ordering_generic_ = 0;
        ordering_landmarks_ = 0;

        // Note:: verticies_ 是 map 类型的, 顺序是按照 id 号排序的
        // 统计带估计的所有变量的总维度
        for (auto vertex: verticies_) {
            ordering_generic_ += vertex.second->LocalDimension();  // 所有的优化变量总维数
        }
    }

//可以暂时不看，后面会再讲
    void Solve::makeHessian() {
        TicToc t_h;
        // 直接构造大的 H 矩阵
        ulong size = ordering_generic_;
        MatXX H(MatXX::Zero(size, size));
        VecX b(VecX::Zero(size));

        // TODO:: accelate, accelate, accelate
//#ifdef USE_OPENMP
//#pragma omp parallel for
//#endif

        // 遍历每个残差，并计算他们的雅克比，得到最后的 H = J^T * J
        for (auto &edge: edges_) {

            edge.second->ComputeResidual();
            edge.second->ComputeJacobians();

            auto jacobians = edge.second->Jacobians();
            auto verticies = edge.second->Verticies();
            assert(jacobians.size() == verticies.size());
            for (size_t i = 0; i < verticies.size(); ++i) {
                auto v_i = verticies[i];
                if (v_i->IsFixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

                auto jacobian_i = jacobians[i];
                ulong index_i = v_i->OrderingId();
                ulong dim_i = v_i->LocalDimension();

                MatXX JtW = jacobian_i.transpose() * edge.second->Information();
                for (size_t j = i; j < verticies.size(); ++j) {
                    auto v_j = verticies[j];

                    if (v_j->IsFixed()) continue;

                    auto jacobian_j = jacobians[j];
                    ulong index_j = v_j->OrderingId();
                    ulong dim_j = v_j->LocalDimension();

                    assert(v_j->OrderingId() != -1);
                    MatXX hessian = JtW * jacobian_j;
                    // 所有的信息矩阵叠加起来
                    H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                    if (j != i) {
                        // 对称的下三角
                        H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                    }
                }
                b.segment(index_i, dim_i).noalias() -= JtW * edge.second->Residual();
            }

        }
        Hessian_ = H;
        b_ = b;
        t_hessian_cost_ += t_h.toc();

        delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;

    }

/*
* Solve Hx = b, we can use PCG iterative method or use sparse Cholesky
*/
    void Solve::solveLinearSystem() {

        delta_x_ = Hessian_.inverse() * b_;
//        delta_x_ = H.ldlt().solve(b_);

    }

    void Solve::updateStates() {
        for (auto vertex: verticies_) {
            ulong idx = vertex.second->OrderingId();
            ulong dim = vertex.second->LocalDimension();
            VecX delta = delta_x_.segment(idx, dim);

            // 所有的参数 x 叠加一个增量  x_{k+1} = x_{k} + delta_x
            vertex.second->Plus(delta);
        }
    }

    void Solve::rollbackStates() {
        for (auto vertex: verticies_) {
            ulong idx = vertex.second->OrderingId();
            ulong dim = vertex.second->LocalDimension();
            VecX delta = delta_x_.segment(idx, dim);

            // 之前的增量加了后使得损失函数增加了，我们应该不要这次迭代结果，所以把之前加上的量减去。
            vertex.second->Plus(-delta);
        }
    }

/// LM
    void Solve::computeLambdaInitLM() {
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

    void Solve::addLambdatoHessianLM() {
        ulong size = Hessian_.cols();
        assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
        for (ulong i = 0; i < size; ++i) {
            Hessian_(i, i) += currentLambda_;
        }
    }

    void Solve::removeLambdaHessianLM() {
        ulong size = Hessian_.cols();
        assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
        // TODO:: 这里不应该减去一个，数值的反复加减容易造成数值精度出问题？而应该保存叠加lambda前的值，在这里直接赋值
        for (ulong i = 0; i < size; ++i) {
            Hessian_(i, i) -= currentLambda_;
        }
    }

//Nielsen的方法，分母直接为L，判断\rho的符号
    bool Solve::isGoodStepInLM() {
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

/** @brief conjugate gradient with perconditioning
*
*  the jacobi PCG method
*
*//*

VecX Solve::pcgSolver(const MatXX &A, const VecX &b, int maxIter = -1) {
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
}*/

}
