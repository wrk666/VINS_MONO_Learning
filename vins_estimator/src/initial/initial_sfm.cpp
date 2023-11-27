#include "initial_sfm.h"

GlobalSFM::GlobalSFM(){}

//手动三角化(底层实现是对构建的H矩阵进行SVD分解，取最后一个特征值对应的特征向量)
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
    //构建Dy=0的系数矩阵D，维度是(2n,4)，n为观测次数，这里是2次观测，一次观测有2个方程，所以就是(4,4)
    //详见博客3.2节式(12)~(14)：https://blog.csdn.net/qq_37746927/article/details/133693726#t4
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	//SVD分解求出V的最右边一列即为最小特征值对应的特征向量
	//return the singular value decomposition of \c *this computed by two-sided Jacobi transformations.
	triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);//三角化出来的是非齐次的，/第4维变为齐次的3D landmark
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

//PnP求解第i帧与第l帧的pose:Tl_i
//输出: R_initial，P_initial
bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state != true)//如果该feature没有被三角化过则直接跳过
			continue;
		Vector2d point2d;
        //这里都是已经被三角化过的landmark_id
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
		    //如果这个id被第[i]帧观测到了（则也被cur观测到过，因为所有的Triangulation都是跟cur做的）
			if (sfm_f[j].observation[k].first == i)
			{
				Vector2d img_pts = sfm_f[j].observation[k].second;
                //第i帧里面的feature_num第j个id的landmark在sfm_f[j].observation的第[k]次观测里面的2D坐标
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
                //之前Triangulation出来的3D landmark坐标
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}
	//如果两帧之间用于PnP的(即tracking上的点太少，则tracking不稳定，不能用于PnP，因为解出来的pose可能不准)
	if (int(pts_2_vector.size()) < 15)//<15则报warning
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)//<10直接报求解失败
			return false;
	}
	cv::Mat r, rvec, t, D, tmp_r;//D畸变系数，设为0
	cv::eigen2cv(R_initial, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);//内参设为Identity()
	bool pnp_succ;
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;

}

void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
    //TODO：感觉这个功能也可以用getCorresponding()来实现啊，找出来的是两帧之间的所有的corres，便利corres，对每个pair进行三角化，后面尝试一下
	for (int j = 0; j < feature_num; j++)//遍历window内feature_id，即window内feature对应的3D landmark的个数
	{
		if (sfm_f[j].state == true)//如果该feature已经被Triangulate那就continue
			continue;
		bool has_0 = false, has_1 = false;//该landmark是否被frame0和frame1观测到
		Vector2d point0;
		Vector2d point1;
		//遍历观测到该landmark的所有帧的frame_id，确定该landmark是否被frame0和frame1观测到
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == frame0)//vector<pair<int,Vector2d>> observation,pair<观测到该landmark的frame_id, feature 2D坐标>
			{
				point0 = sfm_f[j].observation[k].second;
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}
		if (has_0 && has_1)
		{
			Vector3d point_3d;
			//手动三角化
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_f[j].state = true;//已三角化过
			sfm_f[j].position[0] = point_3d(0);//读取3D landmark值
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w 
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
//frame_num-1表示当前帧下标,relative_R,relative_T是Rl_[WINDOW_SIZE],是从最新帧到第l帧
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	feature_num = sfm_f.size();//window内的feature_id的数量
	//cout << "set 0 and " << l << " as known " << endl;
	// have relative_r relative_t
	// intial two view
	q[l].w() = 1;//将第l帧设为参考帧？ Tw_lcam=Tll=Identity()
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();
	q[frame_num - 1] = q[l] * Quaterniond(relative_R);//Tw_cur
	T[frame_num - 1] = relative_T;
	//cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
	//cout << "init t_l " << T[l].transpose() << endl;

	//rotate to cam frame
	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];
	Quaterniond c_Quat[frame_num];
	double c_rotation[frame_num][4];//二维数组，存的都是window内的旋转四元数
	double c_translation[frame_num][3];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];//Matrix<double, 3, 4>的数组，即Tc_w的数组

	//求Tl_l
	c_Quat[l] = q[l].inverse();//转为Tlcam_w
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];//这个Pose存的是Tl_w=Tl_l
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    //求Tcur_l
	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];//Tcur_w=Tcur_l
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];




	//1: trangulate between l ----- frame_num - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1;
    //跟cur帧三角化(l,cur)->跟l帧PnP(l,l+1)->跟cur帧三角化(l+1,cur)->跟l帧PnP(l,l+2)...
    //前面的l的筛选机制，保证了[l~cur]这中间的帧都是有tracking上的点的，至于在帧后(如第l+1帧)才被insert进来的新点，
    //在解出了l+1帧的pose之后就能使用pose进行三角化了，如此循环就能三角化解出[l~cur]中所有的landmark
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp
		if (i > l)
		{
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))//PnP求Tl_i=Ti_w
				return false;
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		// triangulate point based on the solve pnp result手动三角化(SVD分解)：找[i]和[frame_num - 1]中都tracking上的点(两次观测)，构建Dy=0，SVD求解，结果齐次化
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}
	//3: triangulate l-----l+1 l+2 ... frame_num -2
	//三角化cur帧中没有的，但是在l帧和其他帧中tracking上的landmark
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);

	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
	//跟l帧PnP得Tl-1_l->跟l帧三角化得3D landmark->
	//         Tl-2_l->           3D landmark->...
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}
	//5: triangulate all other points
	//在求解完window内所有frame的pose(Ti_l)后，可以对window内的任一个被观测超过1次的landmark进行Triangulation，
	//比如只在(l-2,l-1)之间tracking上的landmark，在与l做Triangulation时是求不出来的
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
        //当某个id的landmark观测帧数大于1时，则可以对该landmark进行triangulation(取第一次和最后一次的观测量，尽量增大t，降低triangulation的不确定性，《14讲》P179~180)
		if ((int)sfm_f[j].observation.size() >= 2)
		{
			Vector2d point0, point1;
			int frame_0 = sfm_f[j].observation[0].first;//第一次观测
			point0 = sfm_f[j].observation[0].second;
			int frame_1 = sfm_f[j].observation.back().first;//最后一次观测
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}		
	}

/*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/
	//full BA  构建BA问题
	ceres::Problem problem;
    //当四元数为优化的对象时，需要调用ceres::QuaternionParameterization来消除自由度冗余
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	//cout << " begin full BA " << endl;
	for (int i = 0; i < frame_num; i++)
	{
		//double array for ceres
		c_translation[i][0] = c_Translation[i].x();
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w();
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i], 3);
		//fix住world(l)系的pose和cur的t
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_rotation[i]);//world
		}
		if (i == l || i == frame_num - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}

    //用window内的frame pose和landmark构建残差块(4,3,3)，并依次加入到problem中
	for (int i = 0; i < feature_num; i++)
	{
		if (sfm_f[i].state != true)
			continue;
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
			int l = sfm_f[i].observation[j].first;
			//自定义cost function，这里使用的是自动求导(AutoDiffCostFunction)，也可以自定义解析Jacobian的计算方式
			//解析求导比自动求导计算更快
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());

    		problem.AddResidualBlock(cost_function, //cost_function
                                    NULL,           //loss_function,一般不用
                                    c_rotation[l],  //后面这3个都是优化的初值，维度分别为4，3，3
                                    c_translation[l],
    								sfm_f[i].position);	 
		}

	}
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		//cout << "vision only BA converge" << endl;
	}
	else
	{
		//cout << "vision only BA not converge " << endl;
		return false;
	}
	//保存Twc
	//旋转qwc
	for (int i = 0; i < frame_num; i++)
	{
		q[i].w() = c_rotation[i][0]; //到cam系
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse();//转为qc_w转为qw_c
		//cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	//平移twc
	for (int i = 0; i < frame_num; i++)
	{

		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
		//cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
	}
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;

}

