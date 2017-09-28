/// @file particle_filter.h
/// @brief �����˲��Ľӿ��ඨ��
/// @author ����
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

/// @brief �����˲��ӿ���
class ParticleFilter
{
	typedef std::vector<double> vec_d;
public:
	/// @brief ϵͳ���̣�û�õ�
	virtual void sys(/*input*/vec_d& xkm1, vec_d& uk, /*output*/vec_d& xk) = 0;

	/// @brief �۲ⷽ�̣�û�õ�
	virtual void obs(/*input*/vec_d& xk, vec_d& vk, /*output*/vec_d& yk) = 0;

	/// @brief �������ӳ�ʼ״̬������
	/// @param [out] x0 ��ʼ״̬
	virtual void gen_x0(vec_d& x0) = 0;
	
	/// @brief �������ӳ�ʼ״̬��Ȩ�ص�����
	/// @param [out] x0 �������ӵĳ�ʼ״̬
	/// @param [out] w0 ����Ȩ��
	virtual void gen_x0(std::vector<vec_d >& x0, vec_d& w0) = 0;
	
	/// @brief �˶�ģ�ͣ�û�õ�
	/// @param [in] xkm1 ���ӵ�ǰһʱ��״̬
	/// @param [out] xk �˶�ģ������������ӵ�ǰ״̬
	/// @return p(xk|xk_1)
	virtual double p_xk_given_xkm1(/*input*/vec_d& xkm1, /*output*/vec_d& xk) = 0;	//kinematic model

	/// @brief �۲�ģ��
	/// @param [in] xk ��������
	/// @param [in] yk �۲�
	/// @param [out] pro p(yk|xk)
	virtual void p_yk_given_xk(/*input*/std::vector<vec_d>& xk, vec_d& yk, /*output*/std::vector<double>& pro) = 0;	//normalized observation model

	/// @brief ����ֲ���û�õ�
	virtual double q_xk_given_xkm1_yk(/*input*/vec_d& xkm1, vec_d& yk, vec_d& xk) = 0;	//proposal distribution
	
	/// @brief ����ϵͳ����
	/// @param [out] uk ϵͳ����
	virtual void gen_sys_noise(vec_d& uk) = 0;

	/// @brief ���ɹ۲�����
	/// @param [out] vk �۲�����
	virtual void gen_obs_noise(vec_d& vk) = 0;

	/// @brief �ӽ���ֲ��в�����x ~ p(xk|xk_1,yk)��û�õ�
	virtual void sample_from_q(/*input*/vec_d& xkm1, vec_d& yk, int n, /*output*/vec_d& x) = 0;

	/// @brief �ӽ���ֲ��и���ǰһʱ���������ɵ�ǰʱ�����ӣ������� ����ֲ�=�˶�ģ�ͣ���ֱ�Ӹ����˶�ģ�ʹ�������
	/// @param [in] pold ǰһʱ����������
	/// @param [out] pnew ���ɵĵ�ǰʱ����������
	virtual void sample_from_q(std::vector<vec_d >& pold, std::vector<vec_d >& pnew) = 0;

	/// @brief Ȩ�ظ���
	virtual void update_weight(/*input*/vec_d& wkm1, std::vector<vec_d>& xkm1, std::vector<vec_d>& xk, vec_d& yk, /*output*/vec_d& wk) = 0;

	/// @brief �����˲���ʵ�֣����ݵ�ǰ�۲�õ�ϵͳ״̬���ڲ���������������״̬��Ȩ��
	/// @param [in] yk �۲�
	/// @param [out] xk ״̬
	/// @param [in] resample_strategy �ز������ԣ�Ŀǰ��ʵ����  SYSTEMATIC_RESAMPLE
	void filter(vec_d& yk, vec_d& xk, int resample_strategy = SYSTEMATIC_RESAMPLE);

	/// @brief �����˲���ʵ�֣����ݵ�ǰ�۲�õ�ϵͳ״̬���ڲ���������������״̬��Ȩ�أ�������
	/// @param [in] yk �۲�
	/// @param [out] xk ״̬
	/// @param [out] particles ÿ������
	/// @param [in] resample_strategy �ز������ԣ�Ŀǰ��ʵ����  SYSTEMATIC_RESAMPLE
	void filter(vec_d& yk, vec_d& xk, std::vector<vec_d>& particles, int resample_strategy = SYSTEMATIC_RESAMPLE);

	/// @brief mcmc������û�õ�
	void mcmc(/*input*/boost::function<double (vec_d&)>& target_dis, 
		boost::function<double (vec_d&, vec_d&)>& proposal_dis, boost::function<void (vec_d&, vec_d&)>& proposal_sam, 
		/*output*/std::vector<vec_d>& x, /*input*/vec_d& x0, int ite_num = 2000, int n = 1);

	/// @brief ���캯��
	/// @param [in] sn ״̬�ռ�ά��
	/// @param [in] pn ���Ӹ���
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

	/// @brief �ز���
	/// @param [in,out] xk ��������״̬
	/// @param [in] wk ����Ȩ��
	/// @param [in] strategy �ز������ԣ�Ŀǰ��ʵ����  SYSTEMATIC_RESAMPLE
	void resample_(std::vector<vec_d>& xk, vec_d& wk, int strategy = SYSTEMATIC_RESAMPLE);
	void bbNms(std::vector<vec_d>& bbs, vec_d& w, vec_d& res, double overlap = 0.65);
	void sort_particles(std::vector<vec_d>& p, vec_d& w, std::vector<vec_d>& sp);
	void calc_average(std::vector<vec_d>& p, vec_d& w, vec_d& avr);
};

#endif