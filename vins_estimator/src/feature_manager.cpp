#include "feature_manager.h"

//由于buf window的所有数据的时候会多buf一帧(如Ps定义为 Vector3d Ps[(WINDOW_SIZE + 1)];)，
//用于判断要如何marg，所以endframe是可能取到最后一帧的，即edFrame()==WINDOW_SIZE
int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)//遍历WINDOW内的所有feature_id，查看可用的feature种类
    {

        it.used_num = it.feature_per_frame.size();
        //如果feature数量不少于2且start_frame小于倒数第二帧（为啥是倒数第二帧？应该是为了保证tracking上以能够三角化出深度）
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}

/**
 * 把当前帧图像（frame_count）的特征点添加到feature容器中
 * 计算第2最新帧与第3最新帧之间的平均视差（当前帧是第1最新帧）
 * 也就是说当前帧图像特征点存入feature中后，并不会立即判断是否将当前帧添加为新的关键帧，而是去判断当前帧的前一帧（第2最新帧）。
 * 当前帧图像要在下一次接收到图像时进行判断（那个时候，当前帧已经变成了第2最新帧）
 */
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0; // 第2最新帧和第3最新帧之间跟踪到的特征点的总视差
    int parallax_num = 0; // 第2最新帧和第3最新帧之间跟踪到的特征点的数量
    last_track_num = 0;// 当前帧（第1最新帧）图像跟踪到的特征点的数量
    for (auto &id_pts : image)//遍历map中的每个元素(键值对)，每个元素都是feature_id --映射--> 该帧图像内此id对应的具体feature(单目，所以vector内只有一个pair对象)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);//取第1个camera(即[0])的的特征，这里只取0是因为只有一个camera, camera_id只有0

        int feature_id = id_pts.first;//feature的id
        //成员变量feature是按照id来关联sliding window内的所有feature的，list
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;//查找该帧图象上的这个id在已有的window内的feature里面有没有
                          });

        //如果没找到就说明是新feature，需要重新根据id和frame号来new一个feature的list元素并push_back
        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        //如果找到了则说明是老feature，就往相应的feature_id对应的vector中push_back
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;// 当前帧（第1最新帧）图像跟踪到的特征点的数量
        }
    }

    // 1. 当前帧的帧号小于2，即为0或1，为0，则没有第2最新帧，为1，则第2最新帧是滑动窗口中的第1帧，则没有第3新帧；故<2时均无法计算第2和第3新帧见的视差
    // 2. 当前帧（第1最新帧）跟踪到的特征点数量小于20
    //（因为如果跟踪到的点数很少，则之前的feature都出现了lost(运动快或者纹理较弱)，所以需要是KF，论文IV.A节）
    // 出现以上2种情况的任意一种，则认为第2最新帧是关键帧
    if (frame_count < 2 || last_track_num < 20)
        return true;

    for (auto &it_per_id : feature)
    {
        // it_per_id.feature_per_frame.size()就是sliding window内该id的feature被tracking的次数，
        // 这两个判断条件意思是：要有至少3帧，且第3新帧和第2新帧之间要tracking上，保证有共视点来计算视差 TODO:可以debug看看，Done：已经debug过了，确实是这个意思
        // 这里img一定是it_per_id.feature_per_frame.size()>=1，即至少tracking了1次，即使在1st lost了但是在3rd和2nd时分别都tracking上了，那么3rd和2nd之间就有视差
        int condition_2 = it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1;
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1) //后一个判断条件是什么意思？
        {
//            ROS_DEBUG("condition matched: condition_1: %d, condition_2: %d, frame_count: %d", it_per_id.start_frame, condition_2, frame_count);
            // 对于给定id的特征点，计算第2最新帧和第3最新帧之间该特征点的视差（当前帧frame_count是第1最新帧）
            //（需要使用IMU数据补偿由于旋转造成的视差）
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        } else {
//            ROS_WARN("condition not matched: condition_1: %d, condition_2: %d, frame_count: %d", it_per_id.start_frame, condition_2, frame_count);
        }
    }

    //如果没有共视点则肯定是new KF
    if (parallax_num == 0)
    {
        return true;
    }
    //若有共视点则判断平均共视点是否大于阈值MIN_PARALLAX，若大于，则视为new KF
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;//MIN_PARALLAX这里配置为10
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);//取逆深度
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;//深度<0则set to solve fail
        }
        else
            it_per_id.solve_flag = 1;//set to solve success
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);//重新赋深度
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());//获取观测到的landmark的数量，并按照此数量定义一个Vector，用于存放逆深度
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))//跳过无效feature(不可能Triangulate出深度的点)
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

//松耦合三角化，不假设观测是在归一化平面上
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature)//WINDOW内的所有特征点
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        //至少被测到2次 && 且观测到该特征点的第一帧图像应该早于或等于滑动窗口第4最新关键帧
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //只三角化刚才标记为负的点
        if (it_per_id.estimated_depth > 0)
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        //P0设为Identity()的T，但是后面没用
        Eigen::Matrix<double, 3, 4> P0;
        //这里Tc0_b[i]*Tbc = Tc0_c[i] = [R0|t0]，求出来的深度都是在c0系下，注意c0是WINDOW[l]，而不是WINDOW[0]，不要被下标搞混乱
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];//Rc0_b[k]*tbc + tc0_bk即Tc0_bk * Tbc = Tc0_ck = Tc0_c[i]，从IMU系再转回camera系
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();
//        ROS_DEBUG_STREAM("P0:\n" << P0 <<"\nR0:\n" << R0 << "\nt0:\n" << t0.transpose());
        //构建Dy=0矩阵，SVD求解
        for (auto &it_per_frame : it_per_id.feature_per_frame)//遍历该id对应的所有不同帧上对此landmark的观测
        {
            imu_j++;

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];//Tc0_cj
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;//Tc0_ci^T * Tc0_cj = Tci_cj
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();//Tci_cj^T = Tcj_ci
            P.rightCols<1>() = -R.transpose() * t;//Tcj_ci
            //λ*[u v w] = Tcj_ci * [x y z 1]这里的[x y z 1]应该没有实际的意义，可能是在ci系下的什么意思，左边整体是cj系下的3d坐标
            //normalized之后不是归一化坐标了，但是模长为1，f为imu_j帧的观测[u',v',1]^T，标准化为[u,v,w]^T
            Eigen::Vector3d f = it_per_frame.point.normalized();
//            ROS_DEBUG_STREAM("before normalized f: " << it_per_frame.point.transpose() <<"   after normalized f: " << f.transpose());
            //D找那个一个block的两行方程，当观测不为归一化坐标时，下面的构建是更一般的形式
            //设观测为[u,v,w]^T，则D.block为：
            //D.row(0) = u * (Tk,3)^T - w * (Tk,1)^T
            //D.row(1) = v * (Tk,3)^T - w * (Tk,2)^T
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        //y取σ4对应的取特征向量，即最后一个 rightCols<1> size=(4,1)
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        //齐次化得camera系下的深度
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;//如果估计的不对，就设为默认的5.0
        }

    }
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

//由于三角化出的camera系下的深度都绑定在start_frame上，所以当marg掉start_frame时，要将深度传递给后面的帧，这里绑定在了start_frame下一帧
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        //不始于第[0]帧的landmark的start_frame前移，
        //始于第[0]帧的landmark，1.如果只在[0]tracking，则直接删掉（因为仅1帧算不出深度），2.如果tracking多于1帧，则将深度传递给start_frame+1帧
        //管理marg之后的start_frame，要往前移1
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;//归一化->camera_marg
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;//Twc_marg * camera = world
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);//Twc_new^(-1) * world=camera_new
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
//            ROS_DEBUG("feature id: %d, start_frame: %d, tracking_size: %lu",it->feature_id, it->start_frame, it->feature_per_frame.size());
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
    ROS_DEBUG("this removeBackShiftDepth end");
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)//如果只在[0]tracking上，则marg掉上个观测之后就没有观测了，删除这个id的feature
                feature.erase(it);
        }
    }
}

//删除2nd
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

/**
 * 对于给定id的特征点
 * 计算第2最新帧和第3最新帧之间该特征点的视差（当前帧frame_count是第1最新帧）
 * （需要使用IMU数据补偿由于旋转造成的视差）
 */
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    //计算3rd和2nd在it_per_id.feature_per_frame下的index，size是该id的feature已经被tracking的次数，
    // 如frame_count=4时，从第0帧开始被tracking，start_frame=0，则3rd和2nd时，it_per_id.feature_per_frame.size()分别为3[index=2],4[index=3]，则他们的index分别为4-2-0=2，4-1-0=3
    // start_frame=1时，size()=2[index=1],3[index=2], index分别为4-2-1=1，4-1-1=2，图示见博客4.2节：https://blog.csdn.net/qq_37746927/article/details/134436475
    int third_lst_idx = frame_count - 2 - it_per_id.start_frame;
    int second_lst_idx = frame_count - 1 - it_per_id.start_frame;
    //这里的window size是10
    /*ROS_INFO("======here1: frame_count: %d, third_lst_idx: %d, second_lst_idx: %d", frame_count, third_lst_idx, second_lst_idx);*/
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[third_lst_idx];//third last frame
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[second_lst_idx];//seconde last frame


    // -------------   3rd     2nd         1st
    //  other_frame     i       j     frame_count
    double ans = 0;
    Vector3d p_j = frame_j.point;//2nd

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;//3rd
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;//这是将i重投影到j？计算rpj error？搞错了吧
    p_i_comp = p_i;
    double dep_i = p_i(2);//深度
    double u_i = p_i(0) / dep_i;//i归一化
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;//计算i,j帧间的视差(为啥j不归一化？)

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;
//    ROS_INFO("====u_equal: %d, v_equal: %d, u_i: %f, u_i_comp: %f, v_i: %f, v_i_comp: %f", u_i==u_i_comp, v_i==v_i_comp, u_i, u_i_comp, v_i, v_i_comp);
    //这俩货是一样的，有啥作用？也没看到IMUN啥补偿啊？
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));//勾股定理计算视差的欧氏距离

    return ans;
}