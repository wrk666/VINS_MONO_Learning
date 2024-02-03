This project is based on VINS_MONO, with some added annotations and extended research, such as manually implementing LM (Levenberg-Marquardt), DogLeg, manual construction of the Hessian matrix, etc.

For more details, please refer to my series of blog posts:
1. [Installing ROS-Noetic from Source on Ubuntu 22.04 and Compiling & Running VINS-MONO](https://blog.csdn.net/qq_37746927/article/details/134392787)
2. [VINS-MONO Code Analysis 1 -- Configuration Files, Data Structures, Frontend Feature Tracker](https://blog.csdn.net/qq_37746927/article/details/134436252)
3. [VINS-MONO Code Analysis 2 -- VINS Estimator (Overall Pipeline and Keyframe Selection Part)](https://blog.csdn.net/qq_37746927/article/details/134436475)
4. [VINS-MONO Code Analysis 3 -- VINS Estimator (Robust Initialization Part)](https://blog.csdn.net/qq_37746927/article/details/134601107)
5. [VINS-MONO Code Analysis 4 -- VINS Estimator (Backend Solving Part)](https://blog.csdn.net/qq_37746927/article/details/134800523)
6. [VINS-MONO Code Analysis 5 -- VINS Estimator (Marginalization Part)](https://blog.csdn.net/qq_37746927/article/details/134880726)
7. [VINS-MONO Code Analysis 6 -- Pose Graph (Final Chapter)](https://blog.csdn.net/qq_37746927/article/details/134952695)
8. [VINS-MONO Extension 1 -- Handwriting Backend Solver, Three Damping Factor Strategies of LM, DogLeg, Constructing the Hessian Matrix](https://blog.csdn.net/qq_37746927/article/details/135150070)
9. [VINS-MONO Extension 2 -- Making Hessian Matrix Faster (p_thread, OpenMP, CUDA, TBB)](https://blog.csdn.net/qq_37746927/article/details/135150104)