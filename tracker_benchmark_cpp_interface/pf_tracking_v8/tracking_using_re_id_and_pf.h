/// @file tracking_using_re_id_and_pf.h
/// @brief 基于粒子滤波的行人跟踪
/// @author 王文
/// @version 8.0
/// @date 2017-9-27

#ifndef TRACKING_USING_RE_ID_AND_PF_H
#define TRACKING_USING_RE_ID_AND_PF_H

//#define SAVE_SCORES_		///< 去掉注释则会将中间结果保存到文件中

#include "particle_filter.h"
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "boost/random.hpp"
#include "boost/smart_ptr.hpp"
#include "boost/function.hpp"
#include "boost/thread.hpp"
#include "boost/bind.hpp"
#include <time.h>
#include <fstream> 
#include "acf_feature_extractor.h"

/// @brief 基于粒子滤波的行人跟踪主类
class PTUsingReIdandPF : public ParticleFilter		//pedestrian tracking using particle filter
{
	typedef std::vector<double> vec_d;
public:
	PTUsingReIdandPF(const std::string& des, int states = 5, int particles = 52);

	/// @brief 构造函数，各参数含义见主函数，粒子状态的5维分别是 bounding box的中心x、中心y、宽度、x方向上速度、y方向上速度
	PTUsingReIdandPF(int states = 5, int particles = 52, bool uofl = true, bool uo = true, double lr = 0.5, double sx = 20, double sy = 20, 
		double sw = 10, double ecri = 0.6, double riw = 0.8, double hpx = 1.2, int tn = 4, double ar = 0.43, double rwr = 0.5, 
		int binL = 16, int binU = 16, int binV = 4, int binHOG = 6, int init_probe = 5, int cur_probe = 3);
	~PTUsingReIdandPF();

	/// @brief 父类系统方程接口的实现，没用到
	void sys(vec_d& xkm1, vec_d& uk, vec_d& xk);

	/// @brief 父类观测方程接口的实现，没用到
	void obs(vec_d& xk, vec_d& vk, vec_d& yk);

	/// @brief 单个粒子初始状态的生成
	/// @param [out] x0 初始状态
	void gen_x0(vec_d& x0);

	/// @brief 所有粒子初始状态和权重的生成
	/// @param [out] x0 所有粒子的初始状态
	/// @param [out] w0 粒子权重
	void gen_x0(std::vector<vec_d >& x0, vec_d& w0);

	/// @brief 运动模型，没用到
	/// @param [in] xkm1 粒子的前一时刻状态
	/// @param [out] xk 运动模型推算出的粒子当前状态
	/// @return p(xk|xk_1)
	double p_xk_given_xkm1(vec_d& xkm1, vec_d& xk);

	/// @brief 观测模型（target-specific + class-specific），将所有粒子分成particles/thread_num_组，并行计算
	/// @param [in] xk 所有粒子
	/// @param [in] yk 观测
	/// @param [out] pro p(yk|xk)
	void p_yk_given_xk(/*input*/std::vector<vec_d>& xk, vec_d& yk, /*output*/std::vector<double>& pro);	//normalized observation model

	/// @brief 由p_yk_given_xk调用
	/// @param [in] s_xk 本组粒子状态的起始
	/// @param [out] s_item0 本组粒子target-specific概率的起始
	/// @param [out] s_item1 本组粒子class-specific概率的起始
	/// @param [in] 本线程处理的粒子数
	void p_yk_given_xk_multi_thread(std::vector<vec_d>::iterator& s_xk, 
		std::vector<double>::iterator& s_item0, 
		std::vector<double>::iterator& s_item1, 
		int particles_per_thread);

	/// @brief 建议分布，没用到
	double q_xk_given_xkm1_yk(vec_d& xkm1, vec_d& yk, vec_d& xk);

	/// @brief 生成系统噪声
	/// @param [out] uk 系统噪声
	void gen_sys_noise(vec_d& uk);

	/// @brief 生成观测噪声
	/// @param [out] vk 观测噪声
	void gen_obs_noise(vec_d& vk);

	/// @brief 从建议分布中采样，x ~ p(xk|xk_1,yk)，没用到
	void sample_from_q(vec_d& xkm1, vec_d& yk, int n, vec_d& x);

	/// @brief 权重更新
	void update_weight(/*input*/vec_d& wkm1, std::vector<vec_d>& xkm1, std::vector<vec_d>& xk, vec_d& yk, /*output*/vec_d& wk);

	/// @brief 跟踪器初始化
	/// @param [in] first_img 应该在第一帧图像处初始化
	/// @param [in] first_loc 第一帧图像中目标的bounding box
	void init(cv::Mat& first_img, cv::Rect& first_loc);

	/// @brief 跟踪接口
	/// @param [in] frame 图像
	/// @return 返回跟踪到的bounding box
	cv::Rect track(cv::Mat& frame);
private:
	AcfFeatureExtractor acf_extractor;		///< 特征提取实例，用于target-specific
	std::string des_;
	cv::HOGDescriptor hog_;					///< 用于class-specific
	int dx;				//velocity of x
	int dy;				//velocity of y
	int dw;				//velocity of width
	//int dh;			//height = 2*width
	cv::Rect init_loc;
	cv::Rect previous_loc;
	cv::Mat* cur_frame;
	cv::Mat pre_frame;

	/// @brief 状态到Rect的转换
	/// @param [in] x 状态向量
	/// @param [out] r Rect
	void states2rect(vec_d& x, cv::Rect& r);

	/// @brief Rect到状态的转换
	/// @param [in] r Rect
	/// @param [out] x 状态向量
	void rect2states(cv::Rect& r, vec_d& x);

	/// @brief 两个矩形框的相似度，交/并
	inline double rects_sim(cv::Rect& r1, cv::Rect& r2){return 1.*(r1&r2).area()/(r1|r2).area();};
	boost::mt19937 rng;
	double sigma_[3];			///< 状态向量中 x, y, width的方差，不会大于sigma_max_
	double sigma_max_[3];		///< 状态向量中 x, y, width的方差的最大值，由构造函数传入
	bool no_matched_orb;

	//new added lk
	/// @defgroup feature_points_track 用于特征点跟踪
	/// @{
	cv::TermCriteria termcrit_;
	cv:: Size sub_pix_winSize, win_size;
	const int MAX_COUNT;
	std::vector<cv::Point2f> lk_points_[2];

	cv::Rect proposal_loc_;		///< 由特征点跟踪得到的期望，论文中图3

	/// @brief 用于生成提取特征点的区域mask
	/// @param [out] msk mask
	/// @param [in] rows mask的高度
	/// @param [in] cols mask的宽度
	/// @param [in] region region内的置1
	void generate_mask(cv::Mat& msk, int rows, int cols, cv::Rect& region);

	/// @brief 计算特征点的中心，论文图6
	/// @param [in] kp 特征点
	/// @param [out] ctr 中心
	void key_points_center(std::vector<cv::Point2f>& kp, cv::Point2f& ctr);

	/// @brief 论文图3
	/// @param [in] pre_kp 前一帧的特征点
	/// @param [in] cur_kp 当前帧配对的特征点
	/// @param [in] pre_loc 前一帧的跟踪结果
	/// @param [out] cur_proposal_loc 由特征点跟踪得到的期望
	void get_proposal_location(std::vector<cv::Point2f>& pre_kp, std::vector<cv::Point2f>& cur_kp, cv::Rect& pre_loc, cv::Rect& cur_proposal_loc);
	/// @}

	/// @brief 调试用，用来显示特征点配对结果
	void drawMatches(cv::Mat& img1, std::vector<cv::Point2f>& points1, cv::Mat& img2, std::vector<cv::Point2f>& points2, cv::Mat& res);

	/// @brief class-specific detector
	double hog_predict(const std::vector<float>& v);
	std::vector<float> hogDetector;		///<  = cv::HOGDescriptor::getDefaultPeopleDetector();

	//parameters
	bool use_optical_flow_lk_;
	bool update_online_;
	double learning_rate_;
	double sigmax_;
	double sigmay_;
	double sigmaw_;
	double exp_coeff_re_id_;
	double re_id_weight_;
	double hog_particle_expand_;
	int thread_num_;
	double aspect_ratio_;

	boost::mutex mut_;

	/// @brief 归一化，使其和为1
	void normalize_(vec_d& data);

	/// @brief class-specific detector得分的映射，论文3.3章节
	void hog_scores_mapping_(vec_d& scores);
	/// @brief target-specific detector得分的映射，论文3.3章节
	void sdalf_scores_mapping_(vec_d& scores);

#ifdef SAVE_SCORES_
	std::ofstream hog_scores_, match_scores_;
#endif

	double width_star_, vx_star_, vy_star_;
	double random_walk_ratio_;

	/// @brief 生成均值m、标准差s的高斯噪声
	double gen_gaussian_noise(double m, double s){boost::normal_distribution<double> nd(m,s);return nd(rng);}
	/// @brief 生成[lb,ub)间的均匀分布噪声
	double gen_uniform_noise(double lb, double ub){boost::uniform_real<double> ur(lb,ub);return ur(rng);}
	cv::Rect get_particles_bounding_box(std::vector<vec_d>::iterator& b, std::vector<vec_d>::iterator& e);

	/// @brief 由随机游走传递粒子
	/// @param [in] particle_pre 前一帧粒子
	/// @param [out] particle_cur 传递后的粒子
	/// @param [in] w_star 前一刻系统状态的宽度，据此生成x 和 y的噪声
	/// @param [in] vx_star 前一时刻系统状态的x - 前前一时刻系统状态的x，目前没用到
	/// @param [in] vy_star 前一时刻系统状态的y - 前前一时刻系统状态的y，目前没用到
	/// @param [in] frame_w 图像宽度，用于判断粒子是否越界
	/// @param [in] frame_h 图像高度，用于判断粒子是否越界
	void rw_motion_update(vec_d& particle_pre, vec_d& particle_cur, double w_star, double vx_star, double vy_star, int frame_w, int frame_h);
	double filter_width(double d);
	double filter_vx(double d);
	double filter_vy(double d);

	/// @brief 从建议分布中根据前一时刻粒子生成当前时刻粒子，论文中 建议分布=运动模型，即直接根据运动模型传递粒子
	/// @param [in] pold 前一时刻所有粒子
	/// @param [out] pnew 生成的当前时刻所有粒子
	void sample_from_q(std::vector<vec_d >& pold, std::vector<vec_d >& pnew/*, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2*/);

	/// @brief 调试用，用来显示特征点配对结果
	void drawMatches(cv::Mat& img1, std::vector<cv::Point2f>& points1, std::vector<vec_d>& ptc1, cv::Mat& img2, std::vector<cv::Point2f>& points2,std::vector<vec_d>& ptc2, cv::Mat& res);

	int probe_needs_;
	double get_weight_(int i, int n);

	//intermediate result
	cv::Mat toshow;
};

#endif