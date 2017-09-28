/// @file tracking_using_re_id_and_pf.h
/// @brief ���������˲������˸���
/// @author ����
/// @version 8.0
/// @date 2017-9-27

#ifndef TRACKING_USING_RE_ID_AND_PF_H
#define TRACKING_USING_RE_ID_AND_PF_H

//#define SAVE_SCORES_		///< ȥ��ע����Ὣ�м������浽�ļ���

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

/// @brief ���������˲������˸�������
class PTUsingReIdandPF : public ParticleFilter		//pedestrian tracking using particle filter
{
	typedef std::vector<double> vec_d;
public:
	PTUsingReIdandPF(const std::string& des, int states = 5, int particles = 52);

	/// @brief ���캯���������������������������״̬��5ά�ֱ��� bounding box������x������y����ȡ�x�������ٶȡ�y�������ٶ�
	PTUsingReIdandPF(int states = 5, int particles = 52, bool uofl = true, bool uo = true, double lr = 0.5, double sx = 20, double sy = 20, 
		double sw = 10, double ecri = 0.6, double riw = 0.8, double hpx = 1.2, int tn = 4, double ar = 0.43, double rwr = 0.5, 
		int binL = 16, int binU = 16, int binV = 4, int binHOG = 6, int init_probe = 5, int cur_probe = 3);
	~PTUsingReIdandPF();

	/// @brief ����ϵͳ���̽ӿڵ�ʵ�֣�û�õ�
	void sys(vec_d& xkm1, vec_d& uk, vec_d& xk);

	/// @brief ����۲ⷽ�̽ӿڵ�ʵ�֣�û�õ�
	void obs(vec_d& xk, vec_d& vk, vec_d& yk);

	/// @brief �������ӳ�ʼ״̬������
	/// @param [out] x0 ��ʼ״̬
	void gen_x0(vec_d& x0);

	/// @brief �������ӳ�ʼ״̬��Ȩ�ص�����
	/// @param [out] x0 �������ӵĳ�ʼ״̬
	/// @param [out] w0 ����Ȩ��
	void gen_x0(std::vector<vec_d >& x0, vec_d& w0);

	/// @brief �˶�ģ�ͣ�û�õ�
	/// @param [in] xkm1 ���ӵ�ǰһʱ��״̬
	/// @param [out] xk �˶�ģ������������ӵ�ǰ״̬
	/// @return p(xk|xk_1)
	double p_xk_given_xkm1(vec_d& xkm1, vec_d& xk);

	/// @brief �۲�ģ�ͣ�target-specific + class-specific�������������ӷֳ�particles/thread_num_�飬���м���
	/// @param [in] xk ��������
	/// @param [in] yk �۲�
	/// @param [out] pro p(yk|xk)
	void p_yk_given_xk(/*input*/std::vector<vec_d>& xk, vec_d& yk, /*output*/std::vector<double>& pro);	//normalized observation model

	/// @brief ��p_yk_given_xk����
	/// @param [in] s_xk ��������״̬����ʼ
	/// @param [out] s_item0 ��������target-specific���ʵ���ʼ
	/// @param [out] s_item1 ��������class-specific���ʵ���ʼ
	/// @param [in] ���̴߳����������
	void p_yk_given_xk_multi_thread(std::vector<vec_d>::iterator& s_xk, 
		std::vector<double>::iterator& s_item0, 
		std::vector<double>::iterator& s_item1, 
		int particles_per_thread);

	/// @brief ����ֲ���û�õ�
	double q_xk_given_xkm1_yk(vec_d& xkm1, vec_d& yk, vec_d& xk);

	/// @brief ����ϵͳ����
	/// @param [out] uk ϵͳ����
	void gen_sys_noise(vec_d& uk);

	/// @brief ���ɹ۲�����
	/// @param [out] vk �۲�����
	void gen_obs_noise(vec_d& vk);

	/// @brief �ӽ���ֲ��в�����x ~ p(xk|xk_1,yk)��û�õ�
	void sample_from_q(vec_d& xkm1, vec_d& yk, int n, vec_d& x);

	/// @brief Ȩ�ظ���
	void update_weight(/*input*/vec_d& wkm1, std::vector<vec_d>& xkm1, std::vector<vec_d>& xk, vec_d& yk, /*output*/vec_d& wk);

	/// @brief ��������ʼ��
	/// @param [in] first_img Ӧ���ڵ�һ֡ͼ�񴦳�ʼ��
	/// @param [in] first_loc ��һ֡ͼ����Ŀ���bounding box
	void init(cv::Mat& first_img, cv::Rect& first_loc);

	/// @brief ���ٽӿ�
	/// @param [in] frame ͼ��
	/// @return ���ظ��ٵ���bounding box
	cv::Rect track(cv::Mat& frame);
private:
	AcfFeatureExtractor acf_extractor;		///< ������ȡʵ��������target-specific
	std::string des_;
	cv::HOGDescriptor hog_;					///< ����class-specific
	int dx;				//velocity of x
	int dy;				//velocity of y
	int dw;				//velocity of width
	//int dh;			//height = 2*width
	cv::Rect init_loc;
	cv::Rect previous_loc;
	cv::Mat* cur_frame;
	cv::Mat pre_frame;

	/// @brief ״̬��Rect��ת��
	/// @param [in] x ״̬����
	/// @param [out] r Rect
	void states2rect(vec_d& x, cv::Rect& r);

	/// @brief Rect��״̬��ת��
	/// @param [in] r Rect
	/// @param [out] x ״̬����
	void rect2states(cv::Rect& r, vec_d& x);

	/// @brief �������ο�����ƶȣ���/��
	inline double rects_sim(cv::Rect& r1, cv::Rect& r2){return 1.*(r1&r2).area()/(r1|r2).area();};
	boost::mt19937 rng;
	double sigma_[3];			///< ״̬������ x, y, width�ķ���������sigma_max_
	double sigma_max_[3];		///< ״̬������ x, y, width�ķ�������ֵ���ɹ��캯������
	bool no_matched_orb;

	//new added lk
	/// @defgroup feature_points_track �������������
	/// @{
	cv::TermCriteria termcrit_;
	cv:: Size sub_pix_winSize, win_size;
	const int MAX_COUNT;
	std::vector<cv::Point2f> lk_points_[2];

	cv::Rect proposal_loc_;		///< ����������ٵõ���������������ͼ3

	/// @brief ����������ȡ�����������mask
	/// @param [out] msk mask
	/// @param [in] rows mask�ĸ߶�
	/// @param [in] cols mask�Ŀ��
	/// @param [in] region region�ڵ���1
	void generate_mask(cv::Mat& msk, int rows, int cols, cv::Rect& region);

	/// @brief ��������������ģ�����ͼ6
	/// @param [in] kp ������
	/// @param [out] ctr ����
	void key_points_center(std::vector<cv::Point2f>& kp, cv::Point2f& ctr);

	/// @brief ����ͼ3
	/// @param [in] pre_kp ǰһ֡��������
	/// @param [in] cur_kp ��ǰ֡��Ե�������
	/// @param [in] pre_loc ǰһ֡�ĸ��ٽ��
	/// @param [out] cur_proposal_loc ����������ٵõ�������
	void get_proposal_location(std::vector<cv::Point2f>& pre_kp, std::vector<cv::Point2f>& cur_kp, cv::Rect& pre_loc, cv::Rect& cur_proposal_loc);
	/// @}

	/// @brief �����ã�������ʾ��������Խ��
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

	/// @brief ��һ����ʹ���Ϊ1
	void normalize_(vec_d& data);

	/// @brief class-specific detector�÷ֵ�ӳ�䣬����3.3�½�
	void hog_scores_mapping_(vec_d& scores);
	/// @brief target-specific detector�÷ֵ�ӳ�䣬����3.3�½�
	void sdalf_scores_mapping_(vec_d& scores);

#ifdef SAVE_SCORES_
	std::ofstream hog_scores_, match_scores_;
#endif

	double width_star_, vx_star_, vy_star_;
	double random_walk_ratio_;

	/// @brief ���ɾ�ֵm����׼��s�ĸ�˹����
	double gen_gaussian_noise(double m, double s){boost::normal_distribution<double> nd(m,s);return nd(rng);}
	/// @brief ����[lb,ub)��ľ��ȷֲ�����
	double gen_uniform_noise(double lb, double ub){boost::uniform_real<double> ur(lb,ub);return ur(rng);}
	cv::Rect get_particles_bounding_box(std::vector<vec_d>::iterator& b, std::vector<vec_d>::iterator& e);

	/// @brief ��������ߴ�������
	/// @param [in] particle_pre ǰһ֡����
	/// @param [out] particle_cur ���ݺ������
	/// @param [in] w_star ǰһ��ϵͳ״̬�Ŀ�ȣ��ݴ�����x �� y������
	/// @param [in] vx_star ǰһʱ��ϵͳ״̬��x - ǰǰһʱ��ϵͳ״̬��x��Ŀǰû�õ�
	/// @param [in] vy_star ǰһʱ��ϵͳ״̬��y - ǰǰһʱ��ϵͳ״̬��y��Ŀǰû�õ�
	/// @param [in] frame_w ͼ���ȣ������ж������Ƿ�Խ��
	/// @param [in] frame_h ͼ��߶ȣ������ж������Ƿ�Խ��
	void rw_motion_update(vec_d& particle_pre, vec_d& particle_cur, double w_star, double vx_star, double vy_star, int frame_w, int frame_h);
	double filter_width(double d);
	double filter_vx(double d);
	double filter_vy(double d);

	/// @brief �ӽ���ֲ��и���ǰһʱ���������ɵ�ǰʱ�����ӣ������� ����ֲ�=�˶�ģ�ͣ���ֱ�Ӹ����˶�ģ�ʹ�������
	/// @param [in] pold ǰһʱ����������
	/// @param [out] pnew ���ɵĵ�ǰʱ����������
	void sample_from_q(std::vector<vec_d >& pold, std::vector<vec_d >& pnew/*, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2*/);

	/// @brief �����ã�������ʾ��������Խ��
	void drawMatches(cv::Mat& img1, std::vector<cv::Point2f>& points1, std::vector<vec_d>& ptc1, cv::Mat& img2, std::vector<cv::Point2f>& points2,std::vector<vec_d>& ptc2, cv::Mat& res);

	int probe_needs_;
	double get_weight_(int i, int n);

	//intermediate result
	cv::Mat toshow;
};

#endif