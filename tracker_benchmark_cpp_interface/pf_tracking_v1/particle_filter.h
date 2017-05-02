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

class ParticleFilter
{
	typedef std::vector<double> vec_d;
public:
	virtual void sys(/*input*/vec_d& xkm1, vec_d& uk, /*output*/vec_d& xk) = 0;
	virtual void obs(/*input*/vec_d& xk, vec_d& vk, /*output*/vec_d& yk) = 0;
	virtual void gen_x0(vec_d& x0) = 0;
	virtual double p_xk_given_xkm1(/*input*/vec_d& xkm1, vec_d& xk) = 0;	//kinematic model
	virtual double p_yk_given_xk(/*input*/vec_d& xk, vec_d& yk) = 0;		//observation model
	virtual void p_yk_given_xk(/*input*/std::vector<vec_d>& xk, vec_d& yk, /*output*/std::vector<double>& pro) = 0;	//normalized observation model
	virtual double q_xk_given_xkm1_yk(/*input*/vec_d& xkm1, vec_d& yk, vec_d& xk) = 0;	//proposal distribution
	virtual void gen_sys_noise(vec_d& uk) = 0;
	virtual void gen_obs_noise(vec_d& vk) = 0;
	virtual void sample_from_q(/*input*/vec_d& xkm1, vec_d& yk, int n, /*output*/vec_d& x) = 0;
	virtual double update_weight(/*input*/double wkm1, vec_d& xkm1, vec_d& xk, vec_d& yk) = 0;
	virtual void update_weight(/*input*/vec_d& wkm1, std::vector<vec_d>& xkm1, std::vector<vec_d>& xk, vec_d& yk, /*output*/vec_d& wk) = 0;
	void filter(vec_d& yk, vec_d& xk, int resample_strategy = SYSTEMATIC_RESAMPLE);
	void filter(vec_d& yk, vec_d& xk, std::vector<vec_d>& particles, int resample_strategy = SYSTEMATIC_RESAMPLE);
	void mcmc(/*input*/boost::function<double (vec_d&)>& target_dis, 
		boost::function<double (vec_d&, vec_d&)>& proposal_dis, boost::function<void (vec_d&, vec_d&)>& proposal_sam, 
		/*output*/std::vector<vec_d>& x, /*input*/vec_d& x0, int ite_num = 2000, int n = 1);
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
	void resample_(std::vector<vec_d>& xk, vec_d& wk, int strategy = SYSTEMATIC_RESAMPLE);
	void bbNms(std::vector<vec_d>& bbs, vec_d& w, vec_d& res, double overlap = 0.65);
	void sort_particles(std::vector<vec_d>& p, vec_d& w, std::vector<vec_d>& sp);
};

#endif