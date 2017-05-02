#ifndef TRACKING_USING_RE_ID_AND_PF_H
#define TRACKING_USING_RE_ID_AND_PF_H

#include "sdalf_re_id.h"
#include "particle_filter.h"
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
#include "boost/random.hpp"
#include "boost/smart_ptr.hpp"
#include "boost/function.hpp"
#include "boost/thread.hpp"
#include "boost/bind.hpp"
#include "Eigen/Core"
#include "eigen.hpp"
#include "nca.h"
#include <time.h>
#include <fstream> //////////////////////////////////////

class PTUsingReIdandPF : public ParticleFilter		//pedestrian tracking using person re-identification and particle filter
{
	typedef std::vector<double> vec_d;
public:
	PTUsingReIdandPF(const std::string& des, int states = 3, int particles = 52);
	PTUsingReIdandPF(int states = 3, int particles = 52, bool uofl = true, bool uo = true, double lr = 0.5, double sx = 20, double sy = 20, 
		double sw = 10, double ecri = 0.6, double riw = 0.8, double hpx = 1.2, int tn = 4, double ar = 0.43);
	~PTUsingReIdandPF();
	void sys(vec_d& xkm1, vec_d& uk, vec_d& xk);
	void obs(vec_d& xk, vec_d& vk, vec_d& yk);
	void gen_x0(vec_d& x0);
	double p_xk_given_xkm1(vec_d& xkm1, vec_d& xk);
	double p_yk_given_xk(vec_d& xk, vec_d& yk);
	void p_yk_given_xk(/*input*/std::vector<vec_d>& xk, vec_d& yk, /*output*/std::vector<double>& pro);	//normalized observation model
	void p_yk_given_xk_multi_thread(std::vector<vec_d>::iterator s_xk, 
		std::vector<double>::iterator s_pro, 
		std::vector<double>::iterator s_item0, 
		std::vector<double>::iterator s_item1, 
		double* si0, double* si1, 
		int particles_per_thread);
	double q_xk_given_xkm1_yk(vec_d& xkm1, vec_d& yk, vec_d& xk);
	void gen_sys_noise(vec_d& uk);
	void gen_obs_noise(vec_d& vk);
	void sample_from_q(vec_d& xkm1, vec_d& yk, int n, vec_d& x);
	double update_weight(double wkm1, vec_d& xkm1, vec_d& xk, vec_d& yk);
	void update_weight(/*input*/vec_d& wkm1, std::vector<vec_d>& xkm1, std::vector<vec_d>& xk, vec_d& yk, /*output*/vec_d& wk);
	void init(cv::Mat& first_img, cv::Rect& first_loc);
	cv::Rect track(cv::Mat& frame);
private:
	pe_re_id::SdalfPeReId sdalf;
	std::string des_;
	cv::HOGDescriptor hog_;
	int dx;				//velocity of x
	int dy;				//velocity of y
	int dw;				//velocity of width
	//int dh;			//height = 2*width
	cv::Rect init_loc;
	cv::Rect previous_loc;
	cv::Mat* cur_frame;
	cv::Mat pre_frame;
	int probe_needs;
	void states2rect(vec_d& x, cv::Rect& r);
	void rect2states(cv::Rect& r, vec_d& x);
	inline double rects_sim(cv::Rect& r1, cv::Rect& r2){return 1.*(r1&r2).area()/(r1|r2).area();};
	boost::mt19937 rng;
	double sigma_[3];
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
	std::ofstream hog_scores_;
	std::ofstream particle_features_;
	boost::mutex mut_;

	//distance metric learning about
	Eigen::MatrixXd input_;
	Eigen::VectorXi labels_;
	int valid_inputs_;
	int first_frame_valid_inputs_;
	cv::Mat M_;
	inline double overlap_(const cv::Rect& r1, const cv::Rect& r2){return 1.*(r1&r2).area()/(r1|r2).area();}
	std::vector<pe_re_id::SdalfFeature> par_fea_;
	std::vector<vec_d > ptcs_;
	bool template_pool_full_;
	std::ofstream sdalf_scores_;
};

#endif