/// @file particle_filter.h
/// @brief 粒子滤波的接口类定义
/// @author 王文
/// @version 8.0
/// @date 2017-9-27

#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <vector>
#include "boost/function.hpp"
#include "boost/bind.hpp"
#include "boost/random.hpp"
#include <numeric>
#include "opencv2/opencv.hpp"
#include <time.h>
#include <fstream>

/// @brief 粒子滤波接口类
class ParticleFilter
{
	typedef std::vector<double> vec_d;
public:
	/// @brief 系统方程，没用到
	virtual void sys(/*input*/vec_d& xkm1, vec_d& uk, /*output*/vec_d& xk) = 0;

	/// @brief 观测方程，没用到
	virtual void obs(/*input*/vec_d& xk, vec_d& vk, /*output*/vec_d& yk) = 0;

	/// @brief 单个粒子初始状态的生成
	/// @param [out] x0 初始状态
	virtual void gen_x0(vec_d& x0) = 0;
	
	/// @brief 所有粒子初始状态和权重的生成
	/// @param [out] x0 所有粒子的初始状态
	/// @param [out] w0 粒子权重
	virtual void gen_x0(std::vector<vec_d >& x0, vec_d& w0) = 0;
	
	/// @brief 运动模型，没用到
	/// @param [in] xkm1 粒子的前一时刻状态
	/// @param [out] xk 运动模型推算出的粒子当前状态
	/// @return p(xk|xk_1)
	virtual double p_xk_given_xkm1(/*input*/vec_d& xkm1, /*output*/vec_d& xk) = 0;	//kinematic model

	/// @brief 观测模型
	/// @param [in] xk 所有粒子
	/// @param [in] yk 观测
	/// @param [out] pro p(yk|xk)
	virtual void p_yk_given_xk(/*input*/std::vector<vec_d>& xk, vec_d& yk, /*output*/std::vector<double>& pro) = 0;	//normalized observation model

	/// @brief 建议分布，没用到
	virtual double q_xk_given_xkm1_yk(/*input*/vec_d& xkm1, vec_d& yk, vec_d& xk) = 0;	//proposal distribution
	
	/// @brief 生成系统噪声
	/// @param [out] uk 系统噪声
	virtual void gen_sys_noise(vec_d& uk) = 0;

	/// @brief 生成观测噪声
	/// @param [out] vk 观测噪声
	virtual void gen_obs_noise(vec_d& vk) = 0;

	/// @brief 从建议分布中采样，x ~ p(xk|xk_1,yk)，没用到
	virtual void sample_from_q(/*input*/vec_d& xkm1, vec_d& yk, int n, /*output*/vec_d& x) = 0;

	/// @brief 从建议分布中根据前一时刻粒子生成当前时刻粒子，论文中 建议分布=运动模型，即直接根据运动模型传递粒子
	/// @param [in] pold 前一时刻所有粒子
	/// @param [out] pnew 生成的当前时刻所有粒子
	virtual void sample_from_q(std::vector<vec_d >& pold, std::vector<vec_d >& pnew) = 0;

	/// @brief 权重更新
	virtual void update_weight(/*input*/vec_d& wkm1, std::vector<vec_d>& xkm1, std::vector<vec_d>& xk, vec_d& yk, /*output*/vec_d& wk) = 0;

	/// @brief 粒子滤波的实现，根据当前观测得到系统状态，内部将更新所有粒子状态和权重
	/// @param [in] yk 观测
	/// @param [out] xk 状态
	/// @param [in] resample_strategy 重采样策略，目前仅实现了  SYSTEMATIC_RESAMPLE
	void filter(vec_d& yk, vec_d& xk, int resample_strategy = SYSTEMATIC_RESAMPLE);

	/// @brief 粒子滤波的实现，根据当前观测得到系统状态，内部将更新所有粒子状态和权重，调试用
	/// @param [in] yk 观测
	/// @param [out] xk 状态
	/// @param [out] particles 每个粒子
	/// @param [in] resample_strategy 重采样策略，目前仅实现了  SYSTEMATIC_RESAMPLE
	void filter(vec_d& yk, vec_d& xk, std::vector<vec_d>& particles, int resample_strategy = SYSTEMATIC_RESAMPLE);

	/// @brief mcmc采样，没用到
	void mcmc(/*input*/boost::function<double (vec_d&)>& target_dis, 
		boost::function<double (vec_d&, vec_d&)>& proposal_dis, boost::function<void (vec_d&, vec_d&)>& proposal_sam, 
		/*output*/std::vector<vec_d>& x, /*input*/vec_d& x0, int ite_num = 2000, int n = 1);

	/// @brief 构造函数
	/// @param [in] sn 状态空间维度
	/// @param [in] pn 粒子个数
	ParticleFilter(int sn, int pn);
	enum{MULTINOMIAL_RESAMPLE, SYSTEMATIC_RESAMPLE};
	inline double normal_distribution(double x,double p_mean = 0., double p_sigma = 1.){return 0.3989422804/p_sigma*exp(-(x-p_mean)*(x-p_mean)/(2*p_sigma*p_sigma));};
	void save_weights(std::ofstream& ofs);

	bool whether_resample;
	int states_num_;
	int particles_num_;
	//WWMatrix<double> xkm1_;
	std::vector<vec_d> xkm1_;
	vec_d wkm1_;
	std::vector<vec_d> sorted_particles;

	/// @brief 重采样
	/// @param [in,out] xk 所有粒子状态
	/// @param [in] wk 粒子权重
	/// @param [in] strategy 重采样策略，目前仅实现了  SYSTEMATIC_RESAMPLE
	void resample_(std::vector<vec_d>& xk, vec_d& wk, int strategy = SYSTEMATIC_RESAMPLE);
	void bbNms(std::vector<vec_d>& bbs, vec_d& w, vec_d& res, double overlap = 0.65);
	void sort_particles(std::vector<vec_d>& p, vec_d& w, std::vector<vec_d>& sp);
	void calc_average(std::vector<vec_d>& p, vec_d& w, vec_d& avr);
};

#endif