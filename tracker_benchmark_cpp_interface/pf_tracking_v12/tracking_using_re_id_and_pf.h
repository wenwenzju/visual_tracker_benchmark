#ifndef TRACKING_USING_RE_ID_AND_PF_H
#define TRACKING_USING_RE_ID_AND_PF_H

#define SAVE_SCORES_

#include "sdalf_re_id.h"
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
#include <fstream> //////////////////////////////////////
#include "acf_feature_extractor.h"

class PTUsingReIdandPF : public ParticleFilter		//pedestrian tracking using person re-identification and particle filter
{
	typedef std::vector<double> vec_d;
public:
	PTUsingReIdandPF(const std::string& des, int states = 5, int particles = 52);
	PTUsingReIdandPF(int states = 5, int particles = 52, bool uofl = true, bool uo = true, double lr = 0.5, double sx = 20, double sy = 20, 
		double sw = 10, double ecri = 0.6, double riw = 0.8, double hpx = 1.2, int tn = 4, double ar = 0.43, double rwr = 0.5, 
		int binL = 16, int binU = 16, int binV = 4, int binHOG = 6, int init_probe = 5, int cur_probe = 3);
	~PTUsingReIdandPF();
	void sys(vec_d& xkm1, vec_d& uk, vec_d& xk);
	void obs(vec_d& xk, vec_d& vk, vec_d& yk);
	void gen_x0(vec_d& x0);
	void gen_x0(std::vector<vec_d >& x0, vec_d& w0);
	double p_xk_given_xkm1(vec_d& xkm1, vec_d& xk);
	void p_yk_given_xk(/*input*/std::vector<vec_d>& xk, vec_d& yk, /*output*/std::vector<double>& pro);	//normalized observation model
	void p_yk_given_xk_multi_thread(std::vector<vec_d>::iterator s_xk, 
		std::vector<double>::iterator s_pro, 
		std::vector<double>::iterator s_item0,  
		double* si0, int particles_per_thread);
	double q_xk_given_xkm1_yk(vec_d& xkm1, vec_d& yk, vec_d& xk);
	void gen_sys_noise(vec_d& uk);
	void gen_obs_noise(vec_d& vk);
	void sample_from_q(vec_d& xkm1, vec_d& yk, int n, vec_d& x);
	void update_weight(/*input*/vec_d& wkm1, std::vector<vec_d>& xkm1, std::vector<vec_d>& xk, vec_d& yk, /*output*/vec_d& wk);
	void init(cv::Mat& first_img, cv::Rect& first_loc);
	cv::Rect track(cv::Mat& frame);
private:
	//pe_re_id::SdalfPeReId sdalf;
	AcfFeatureExtractor acf_extractor;
	std::string des_;
	cv::HOGDescriptor hog_;
	int dx;				//velocity of x
	int dy;				//velocity of y
	int dw;				//velocity of width
	int dh;				//velocity of height
	cv::Rect init_loc;
	cv::Rect previous_loc;
	cv::Mat* cur_frame;
	cv::Mat pre_frame;
	void states2rect(vec_d& x, cv::Rect& r);
	void rect2states(cv::Rect& r, vec_d& x);
	inline double rects_sim(cv::Rect& r1, cv::Rect& r2){return 1.*(r1&r2).area()/(r1|r2).area();};
	boost::mt19937 rng;
	double sigma_[4];
	double sigma_max_[3];
	bool no_matched_orb;

	//new added lk
	cv::TermCriteria termcrit_;
	cv:: Size sub_pix_winSize, win_size;
	const int MAX_COUNT;
	std::vector<cv::Point2f> lk_points_[2];
	cv::Rect proposal_loc_;
	void generate_mask(cv::Mat& msk, int rows, int cols, cv::Rect& region);
	void key_points_center(std::vector<cv::KeyPoint>& kp, cv::Point2f& ctr);
	void get_proposal_location(std::vector<cv::Point2f>& pre_kp, std::vector<cv::Point2f>& cur_kp, cv::Rect& pre_loc, cv::Rect& cur_proposal_loc);
	void drawMatches(cv::Mat& img1, std::vector<cv::Point2f>& points1, cv::Mat& img2, std::vector<cv::Point2f>& points2, cv::Mat& res);
	double hog_predict(const std::vector<float>& v);
	void re_id_model_update(pe_re_id::SdalfFeature& sf1, pe_re_id::SdalfFeature& sf2, pe_re_id::SdalfFeature& sf3/*output*/, double alpha);
	std::vector<float> hogDetector;
	pe_re_id::SdalfFeature init_probe;

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


	////////////////////////////////////
	boost::mutex mut_;

	void normalize_(vec_d& data);
	void hog_scores_mapping_(vec_d& scores);
	void sdalf_scores_mapping_(vec_d& scores);

#ifdef SAVE_SCORES_
	std::ofstream match_scores_;
#endif

	/************************************************************************/
	/*                       version 4.0 add                                        */
	double width_star_, height_star_, vx_star_, vy_star_;
	double random_walk_ratio_;
	double gen_gaussian_noise(double m, double s){boost::normal_distribution<double> nd(m,s);return nd(rng);}
	double gen_uniform_noise(double lb, double ub){boost::uniform_real<double> ur(lb,ub);return ur(rng);}
	cv::Rect get_particles_bounding_box(std::vector<vec_d>::iterator& b, std::vector<vec_d>::iterator& e);
	void fpt_motion_update(vec_d& particle_pre, vec_d& particle_cur, double meanx, double meany);
	void rw_motion_update(vec_d& particle_pre, vec_d& particle_cur, double w_star, double h_star, double vx_star, double vy_star, int frame_w, int frame_h);
	double filter_width(double d);
	double filter_height(double d);
	double filter_vx(double d);
	double filter_vy(double d);
	void sample_from_q(std::vector<vec_d >& pold, std::vector<vec_d >& pnew/*, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2*/);
	void drawMatches(cv::Mat& img1, std::vector<cv::Point2f>& points1, std::vector<vec_d>& ptc1, cv::Mat& img2, std::vector<cv::Point2f>& points2,std::vector<vec_d>& ptc2, cv::Mat& res);
	/************************************************************************/

	/************************************************************************/
	/* version 5.0 add                                                      */
	void re_id_model_update(pe_re_id::SdalfFeature& sf, std::vector<pe_re_id::SdalfFeature>& p);
	int probe_needs_;
	double get_weight_(int i, int n);
	/************************************************************************/

	/************************************************************************/
	/* version 9.0 add                                                      */
	bool is_fast_motion(std::vector<cv::Point2f>& p1, std::vector<cv::Point2f>& p2);
	/************************************************************************/
};

#endif