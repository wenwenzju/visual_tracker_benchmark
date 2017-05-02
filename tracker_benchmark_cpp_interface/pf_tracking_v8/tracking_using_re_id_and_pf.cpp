#include "tracking_using_re_id_and_pf.h"
#include "imwrite.h"

#define USE_WEIGHTED_AVERAGE_TEMPLATE
//#define SHOW_EACH_PARTICLE
#define USE_SPATIAL

PTUsingReIdandPF::PTUsingReIdandPF(const std::string& des, int states, int particles) : ParticleFilter(states, particles), acf_extractor(), /*rng(time(0)),*/ no_matched_orb(false),
	termcrit_(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03), sub_pix_winSize(10,10), win_size(31,31), MAX_COUNT(500)
{
	dx=dy=0;dw = 0;probe_needs_ = 5;
	sigma_[0] = 10.;
	sigma_[1] = 10.;
	sigma_[2] = 2.;
	hogDetector = cv::HOGDescriptor::getDefaultPeopleDetector();
}

PTUsingReIdandPF::PTUsingReIdandPF(int states /* = 3 */, int particles /* = 52 */, bool uofl /* = true */, bool uo /* = true */, double lr /* = 0.4 */, double sx /* = 20 */, 
	double sy /* = 10 */, double sw /* = 8 */, double ecri /* = 0.5 */, double riw /* = 0.8 */, double hpx /* = 1.2 */, int tn /* = 4 */, double ar /* = 0.43*/, double rwr /* = 0.5*/, 
	int binL /* = 16*/, int binU /* = 16*/, int binV /* = 4*/, int binHOG /* = 6*/, int init_probe /* = 5*/, int cur_probe /* = 3*/)
	: ParticleFilter(states, particles), acf_extractor(binL, binU, binV, binHOG, 
#ifdef USE_SPATIAL
	4, 2, 
#else
	0, 0, 
#endif
	init_probe, cur_probe), /*rng(time(0)),*/ no_matched_orb(false),
	termcrit_(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03), sub_pix_winSize(10,10), win_size(31,31), MAX_COUNT(500),
	use_optical_flow_lk_(uofl), update_online_(uo), learning_rate_(lr), sigmax_(sx), sigmay_(sy), sigmaw_(sw), exp_coeff_re_id_(ecri), re_id_weight_(riw), hog_particle_expand_(hpx), thread_num_(tn), aspect_ratio_(ar)
	,random_walk_ratio_(rwr)
#ifdef SAVE_SCORES_
	,hog_scores_("hog_scores.txt"), match_scores_("match_scores.txt")
#endif
{
	sigma_max_[0] = sigma_[0] = sx;
	sigma_max_[1] = sigma_[1] = sy;
	sigma_max_[2] = sigma_[2] = sw;
	hogDetector = cv::HOGDescriptor::getDefaultPeopleDetector();
}

PTUsingReIdandPF::~PTUsingReIdandPF()
{

}

void PTUsingReIdandPF::init(cv::Mat& first_img, cv::Rect& first_loc)
{
	init_loc = first_loc;
//	if (first_loc.width<first_loc.height*aspect_ratio_)
//	{
		init_loc.width = cvRound(first_loc.height*aspect_ratio_);
		int x = first_loc.x-(init_loc.width-first_loc.width)/2;
		init_loc.x = x<0?0:x;
//	}
// 	else
// 	{
// 		init_loc.height = cvRound(first_loc.width/aspect_ratio_);
// 		int y = first_loc.y - (init_loc.height - first_loc.height)/2;
// 		init_loc.y = y<0?0:y;
// 	}
	previous_loc = init_loc;
	pre_frame = first_img.clone();

	acf_extractor.feature_extract(first_img);
	acf_extractor.update_match_model(first_loc);

	vx_star_ = 0.;
	vy_star_ = 0.;
	width_star_ = init_loc.width;

// 	//new add lk optical flow
// 	if (use_optical_flow_lk_)
// 	{
// 		cv::Mat msk;
// 		generate_mask(msk, first_img.rows, first_img.cols, first_loc);
// 		cv::Mat gray;
// 		cv::cvtColor(first_img, gray, CV_BGR2GRAY);
// 		cv::goodFeaturesToTrack(gray, lk_points_[0], MAX_COUNT, 0.01, 10, msk, 3, 0, 0.04);
// 		cornerSubPix(gray, lk_points_[0], sub_pix_winSize, cv::Size(-1,-1), termcrit_);
// 	}

	////////////////////////////////////new added self-adaptive sigma
	//sigma_[0] = 1.*previous_loc.width/3.;
	sigma_[0] = 1.*previous_loc.width/6. < sigma_max_[0] ? 1.*previous_loc.width/3. : sigma_max_[0];
	//sigma_[1] = 1.*previous_loc.width/6.;
	sigma_[1] = 1.*previous_loc.width/6. < sigma_max_[1] ? 1.*previous_loc.width/6. : sigma_max_[1];
	//sigma_[2] = 1.*previous_loc.width/6.;
	sigma_[2] = 1.*previous_loc.width/10. < sigma_max_[2] ? 1.*previous_loc.width/10. : sigma_max_[2];
}

void PTUsingReIdandPF::sys(vec_d& xkm1, vec_d& uk, vec_d& xk)
{
	if (xkm1.size() == 0) return;
	xk = vec_d(states_num_, 0);
	int uks = uk.size();

	//xk[3] = xkm1[3] + (uks == 0 ? 0 : uk[0]);			//dx
	//xk[4] = xkm1[4] + (uks == 0 ? 0 : uk[1]);			//dy
	//xk[5] = xkm1[5] + (uks == 0 ? 0 : uk[2]);			//dw
	dx = (uks == 0 ? 0 : uk[0]);			//dx
	dy = (uks == 0 ? 0 : uk[1]);			//dy
	dw = (uks == 0 ? 0 : uk[2]);			//dw

	xk[0] = xkm1[0] + dx;							//center x
	xk[1] = xkm1[1] + dy;							//center y
	xk[2] = xkm1[2] + dw;							//width
}

void PTUsingReIdandPF::obs(vec_d& xk, vec_d& vk, vec_d& yk)
{

}

void PTUsingReIdandPF::gen_x0(vec_d& x0)
{
	//x0.clear();
	//x0.push_back(init_loc.x);
	//x0.push_back(init_loc.y);
	//x0.push_back(init_loc.width);
	rect2states(init_loc, x0);
	//x0.push_back(0.);			//dx
	//x0.push_back(0.);			//dy
	//x0.push_back(0.);			//dw
	x0.clear();
	vec_d x;
	rect2states(init_loc, x);
	x0.push_back(x[0] + gen_gaussian_noise(0, sigma_[0]));//x
	x0.push_back(x[1] + gen_gaussian_noise(0, sigma_[1]));//y
	x0.push_back(x[2] + gen_gaussian_noise(0, sigma_[2]));//width
	x0.push_back(0. + gen_gaussian_noise(0, 5));		//vx
	x0.push_back(0. + gen_gaussian_noise(0, 5));		//vy

	if (x0[0] < 0) x0[0] = 0;
	else if (x0[0] > pre_frame.cols) x0[0] = pre_frame.cols;
	if (x0[1] < 0) x0[1] = 0;
	else if (x0[1] > pre_frame.rows) x0[1] = pre_frame.rows;
	if (x0[2] < 10) x0[2] = 10;
	else if (x0[2]/aspect_ratio_ > pre_frame.rows) x0[2] = pre_frame.rows*aspect_ratio_;
}

void PTUsingReIdandPF::gen_x0(std::vector<vec_d >& x0, vec_d& w0)
{
	for (int i = 0; i < particles_num_; ++i)
	{
		vec_d x;
		gen_x0(x);
		x0.push_back(x);
	}
	w0 = vec_d(particles_num_, 1./particles_num_);

	 	if (use_optical_flow_lk_)
	 	{
	 		cv::Mat msk;
	 		generate_mask(msk, pre_frame.rows, pre_frame.cols, init_loc);
	 		cv::Mat gray;
	 		cv::cvtColor(pre_frame, gray, CV_BGR2GRAY);
	 		cv::goodFeaturesToTrack(gray, lk_points_[0], MAX_COUNT, 0.01, 10, msk, 3, 0, 0.04);
			if (lk_points_[0].size())
	 			cornerSubPix(gray, lk_points_[0], sub_pix_winSize, cv::Size(-1,-1), termcrit_);
	 	}
}

double PTUsingReIdandPF::p_xk_given_xkm1(vec_d& xkm1, vec_d& xk)
{
	//ternary normal distribution; indenpendent identically distributed
	vec_d xk_mean;
	sys(xkm1, vec_d(), xk_mean);
	return	normal_distribution(xk[3], xk_mean[3], sigmax_) * 
			normal_distribution(xk[4], xk_mean[4], sigmay_) * 
			normal_distribution(xk[5], xk_mean[5], sigmaw_);
}

void PTUsingReIdandPF::p_yk_given_xk_multi_thread(std::vector<vec_d>::iterator& s_xk, 
	std::vector<double>::iterator& s_item0, 
	std::vector<double>::iterator& s_item1, 
	double* si0, double* si1, 
	int particles_per_thread)
{
	for (int i = 0; i < particles_per_thread; s_xk++, s_item0++, s_item1++, i++)
	{
		cv::Rect r;
		states2rect(*s_xk, r);
		
		std::vector<cv::Mat> g(1);
		cv::Rect tmp;
		tmp = (r&cv::Rect(0,0,cur_frame->cols, cur_frame->rows));

		double d = acf_extractor.calc_roi_scores(tmp, learning_rate_);
		//item0[i] = normal_distribution(d, 0., 0.35);

		//*s_item0 = exp(exp_coeff_re_id_*1./(d+DBL_MIN));
		*s_item0 = d;

		std::vector<float> descrip;
		cv::Mat hog_p;
		cv::Rect hog_r,hog_roi;
		vec_d hog_xk(states_num_);
		hog_xk[0] = (*s_xk)[0];
		hog_xk[1] = (*s_xk)[1];		//same center
		hog_xk[2] = hog_particle_expand_*(*s_xk)[2];		//expand width
		states2rect(hog_xk, hog_r);

		hog_roi = hog_r&cv::Rect(0,0,cur_frame->cols, cur_frame->rows);
		cur_frame->operator()(hog_roi).copyTo(hog_p);
		copyMakeBorder(hog_p, hog_p, hog_roi.y-hog_r.y, hog_r.br().y-hog_roi.br().y, hog_roi.x-hog_r.x, hog_r.br().x-hog_roi.br().x, cv::BORDER_REPLICATE);
		resize(hog_p, hog_p, cv::Size(64, 128));
		hog_.compute(hog_p,descrip);
		double det_coeff = hog_predict(descrip);

		//*s_item1 = exp(2*det_coeff);
		*s_item1 = det_coeff;

#ifdef SAVE_SCORES_
		{
			boost::mutex::scoped_lock scop(mut_);
			hog_scores_ << det_coeff << " " << (*s_xk)[0] << " " << (*s_xk)[1] << " " << (*s_xk)[2] << std::endl;
			match_scores_ << d << " " << (*s_xk)[0] << " " << (*s_xk)[1] << " " << (*s_xk)[2] << std::endl;
		}
#endif
	}
}

void PTUsingReIdandPF::p_yk_given_xk(std::vector<vec_d>& xk, vec_d& yk, std::vector<double>& pro)
{
//#define MIN_MAX
	int pn = xk.size();
	std::vector<double> item0(pn, 0), item1(pn, 0);
	pro = std::vector<double>(pn, 0.);
	double si0 = 0., si1 = 0.;
	std::vector<double> ds;

	int particles_per_thread = particles_num_ / thread_num_;
	std::vector<boost::thread> threads(thread_num_);

	for (int i = 0; i < thread_num_; ++i)
	{
		
		std::vector<vec_d>::iterator s_xk = xk.begin()+i*particles_per_thread;
		//std::vector<double>::iterator s_pro = pro.begin()+i*particles_per_thread;
		std::vector<double>::iterator s_item0 = item0.begin()+i*particles_per_thread;
		std::vector<double>::iterator s_item1 = item1.begin()+i*particles_per_thread;

		if (i != thread_num_-1)
			threads[i] = boost::thread(boost::bind(&PTUsingReIdandPF::p_yk_given_xk_multi_thread, this, 
			s_xk, s_item0, s_item1, &si0, &si1, particles_per_thread));
		else
		{
			threads[i] = boost::thread(boost::bind(&PTUsingReIdandPF::p_yk_given_xk_multi_thread, this, 
				s_xk, s_item0, s_item1, &si0, &si1, particles_per_thread+particles_num_%thread_num_));
		}
	}

	for (int i = 0; i < thread_num_; ++i)
	{
		threads[i].join();
	}

	hog_scores_mapping_(item1);
	sdalf_scores_mapping_(item0);

	for (int i = 0; i < pn; ++i)
		pro[i] = re_id_weight_*item0[i] + (1-re_id_weight_)*item1[i];
}

double PTUsingReIdandPF::q_xk_given_xkm1_yk(vec_d& xkm1, vec_d& yk, vec_d& xk)
{
	//Version 3.0 use ORB
	vec_d xk_mean;
	rect2states(proposal_loc_, xk_mean);
	return normal_distribution(xk[0], xk_mean[0], sigmax_) * 
		normal_distribution(xk[1], xk_mean[1], sigmay_) * 
		normal_distribution(xk[2], xk_mean[2], sigmaw_);
}

void PTUsingReIdandPF::gen_sys_noise(vec_d& uk)
{
	uk.clear();

	for (int i = 0; i < 3; ++i)
	{
		boost::normal_distribution<double> nd(0, sigma_[i]);
		
		uk.push_back(nd(rng));
	}
}

void PTUsingReIdandPF::gen_obs_noise(vec_d& vk)
{

}

void PTUsingReIdandPF::sample_from_q(vec_d& xkm1, vec_d& yk, int n, vec_d& x)
{
	//version 3.0 use orb to propose proposal locations
	vec_d uk;
	do
	{
		rect2states(proposal_loc_, x);
		gen_sys_noise(uk);
		x[0] += uk[0];
		x[1] += uk[1];
		x[2] += uk[2];

	}while(x[0] - x[2]/2 < 0 || x[1] - x[2]/aspect_ratio_/2 < 0 || x[2] < 10);
}

#define NSIGMA 2
void PTUsingReIdandPF::sample_from_q(std::vector<vec_d >& pold, std::vector<vec_d >& pnew)
{
	if (lk_points_[0].size())
	{
		cv::Mat pre_gray, cur_gray;
		std::vector<uchar> status;
		std::vector<float> err;
		
		cv::cvtColor(pre_frame, pre_gray, CV_BGR2GRAY);
		cv::cvtColor(*cur_frame, cur_gray, CV_BGR2GRAY);
		cv::calcOpticalFlowPyrLK(pre_gray, cur_gray, lk_points_[0], lk_points_[1], status, err, win_size,
			3, termcrit_, 0, 0.001);
		size_t i, k;
		for( i = k = 0; i < lk_points_[1].size(); i++ )
		{
			if( !status[i] )
				continue;

			lk_points_[1][k] = lk_points_[1][i];
			lk_points_[0][k++] = lk_points_[0][i];
		}
		lk_points_[1].resize(k);
		lk_points_[0].resize(k);
	}
	else lk_points_[1].clear();

	cv::Rect cur_proposal;
	get_proposal_location(lk_points_[0], lk_points_[1], previous_loc, cur_proposal);
	vec_d cp; rect2states(cur_proposal, cp);

	//pnew = pold;
	pnew.resize(pold.size(), vec_d(states_num_));
	std::vector<vec_d>::iterator fpt_ite_new = pnew.begin()+pnew.size()*random_walk_ratio_;
	for (; fpt_ite_new != pnew.end(); fpt_ite_new++)
	{
		(*fpt_ite_new)[0] = cp[0]+gen_gaussian_noise(0, sigma_[0]);
		(*fpt_ite_new)[1] = cp[1]+gen_gaussian_noise(0, sigma_[1]);
		(*fpt_ite_new)[2] = cp[2]+gen_gaussian_noise(0, sigma_[2]);
		(*fpt_ite_new)[3] = vx_star_+gen_gaussian_noise(0,2);
		(*fpt_ite_new)[4] = vy_star_+gen_gaussian_noise(0,2);
		if ((*fpt_ite_new)[0] < 0) (*fpt_ite_new)[0] = 0;
		else if ((*fpt_ite_new)[0] > pre_frame.cols) (*fpt_ite_new)[0] = pre_frame.cols;
		if ((*fpt_ite_new)[1] < 0) (*fpt_ite_new)[1] = 0;
		else if ((*fpt_ite_new)[1] > pre_frame.rows) (*fpt_ite_new)[1] = pre_frame.rows;
		if ((*fpt_ite_new)[2] < 10) (*fpt_ite_new)[2] = 10;
		else if ((*fpt_ite_new)[2]/aspect_ratio_ > pre_frame.rows) (*fpt_ite_new)[2] = pre_frame.rows*aspect_ratio_;
	}
	///////////////////////////////////////////////////////////////////////
	
	/////////////////////////////////////////////////////////////////////

	std::vector<vec_d>::iterator rw_ite_old = pold.begin(),rw_ite_old_end = pold.begin()+pold.size()*random_walk_ratio_,
		rw_ite_new = pnew.begin();
	for (; rw_ite_old != rw_ite_old_end; rw_ite_old++, rw_ite_new++)
	{
		rw_motion_update(*rw_ite_old, *rw_ite_new, width_star_, vx_star_, vy_star_, pre_frame.cols, pre_frame.rows);
	}

#ifdef SHOW_EACH_PARTICLE
 	//cv::Mat toshow;
 	drawMatches(pre_frame, lk_points_[0], pold, *cur_frame, lk_points_[1], pnew, toshow);
//	cv::Point2f pc1; key_points_center(lk_points_[1], pc1);
//	cv::circle(toshow, pc1+cv::Point2f(pre_frame.cols, 0), 3, cv::Scalar(0,0,255), -1);
//	cv::Point2f pc0; key_points_center(lk_points_[0], pc0);
//	cv::circle(toshow, pc0, 3, cv::Scalar(0,0,255), -1);
 	//imshow("match",toshow);
 	//cv::waitKey(2);
#endif

	//for paper
	//cv::Mat pre_frm_tmp = pre_frame.clone();
	//cv::rectangle(pre_frm_tmp, previous_loc, cv::Scalar(0,0,255), 2);
	//for (int i = 0; i < lk_points_[0].size(); ++i) cv::circle(pre_frm_tmp, lk_points_[0][i], 3, cv::Scalar(0,255,0));
	//cv::Point2f pc; key_points_center(lk_points_[0], pc);
	//cv::circle(pre_frm_tmp, pc, 3, cv::Scalar(0,0,255), -1);
	//imshow("feature points", pre_frm_tmp);
}

void PTUsingReIdandPF::update_weight(vec_d& wkm1, std::vector<vec_d>& xkm1, std::vector<vec_d>& xk, vec_d& yk, vec_d& wk)
{
	std::vector<double> pro;		//normalized probability of p_yk_given_xk
	p_yk_given_xk(xk, yk, pro);

	wk = vec_d(xk.size(), 0.);
	for (int i = 0; i < xk.size(); ++i)
		wk[i] = wkm1[i]*pro[i];
		//wk[i] = 1./particles_num_*pro[i];
}

cv::Rect PTUsingReIdandPF::track(cv::Mat& frame)
{
	static int frame_num = 1;
	cur_frame = &frame;

	acf_extractor.feature_extract(frame);

	vec_d xk;			//states of time t
	std::vector<vec_d> p;
	filter(vec_d(), xk, p);

	////////////////////////////////////
// 	if (whether_resample) {learning_rate_ = 0.8;}
// 	else {learning_rate_ = 0.4;}
	//std::cout << "**********learning rate = " << learning_rate_ << "*************" << std::endl; 

	width_star_ = filter_width(xk[2]);
	vec_d xk_1;			//states of time t-1
	rect2states(previous_loc, xk_1);
	vx_star_ = filter_vx(xk[0] - xk_1[0]);
	//vx_star_ = xk[0] - xk_1[0];
	vy_star_ = filter_vy(xk[1] - xk_1[1]);

	//std::cout << "vx_star=" << vx_star_ << "; vy_star=" << vy_star_ << std::endl;

#ifdef SHOW_EACH_PARTICLE
	//cv::rectangle(toshow, previous_loc, cv::Scalar(0,0,255), 2);
#endif

	states2rect(xk, previous_loc);

#ifdef SHOW_EACH_PARTICLE
	//cv::rectangle(toshow, cv::Rect(previous_loc.x+frame.cols, previous_loc.y, previous_loc.width, previous_loc.height), cv::Scalar(0,0,255), 2);
	//cv::Mat toshowtmp = toshow(cv::Rect(frame.cols, 0, frame.cols, frame.rows));
	//cv::imshow("p", toshowtmp);
	cv::imshow("match", toshow);
#endif

	//static double rwr = random_walk_ratio_;
	//random_walk_ratio_ = rwr;
	if (whether_resample)
	{acf_extractor.update_match_model(previous_loc&cv::Rect(0,0,frame.cols, frame.rows));}
	//else random_walk_ratio_ = 0.8;
	 	//if (whether_resample) {learning_rate_ = 0.8;}
	 	//else {learning_rate_ = 0.2;}

	pre_frame = frame.clone();

	////////////////////////////////////new added self-adaptive sigma
	//sigma_[0] = 1.*previous_loc.width/3.;
	sigma_[0] = 1.*previous_loc.width/6. < sigma_max_[0] ? 1.*previous_loc.width/6. : sigma_max_[0];
	//sigma_[1] = 1.*previous_loc.width/6.;
	sigma_[1] = 1.*previous_loc.width/6. < sigma_max_[1] ? 1.*previous_loc.width/6. : sigma_max_[1];
	//sigma_[2] = 1.*previous_loc.width/6.;
	sigma_[2] = 1.*previous_loc.width/10. < sigma_max_[2] ? 1.*previous_loc.width/10. : sigma_max_[2];

	if (use_optical_flow_lk_)
	{
		cv::Mat msk;
		generate_mask(msk, pre_frame.rows, pre_frame.cols, previous_loc);
		cv::Mat gray;
		cv::cvtColor(pre_frame, gray, CV_BGR2GRAY);
		cv::goodFeaturesToTrack(gray, lk_points_[0], MAX_COUNT, 0.01, 10, msk, 3, 0, 0.04);
		if (lk_points_[0].size())
			cornerSubPix(gray, lk_points_[0], sub_pix_winSize, cv::Size(-1,-1), termcrit_);
	}

	return previous_loc;
}


void PTUsingReIdandPF::states2rect(vec_d& x, cv::Rect& r)
{
	int xx = x[0] - x[2]/2;
	int yy = x[1] - cvRound(x[2]/aspect_ratio_)/2;
	r = cv::Rect(xx,yy,(int)x[2],cvRound(x[2]/aspect_ratio_));
}

void PTUsingReIdandPF::rect2states(cv::Rect& r, vec_d& x)
{
	x.clear();
	x.push_back(r.x+r.width/2);
	x.push_back(r.y+r.height/2);
	x.push_back(r.width);
}

void PTUsingReIdandPF::generate_mask(cv::Mat& msk,int rows, int cols, cv::Rect& region)
{
	msk = cv::Mat::zeros(rows, cols, CV_8UC1);
	cv::Rect r = region & cv::Rect(0,0,cols, rows);
	msk(r) = cv::Mat::ones(r.height, r.width, msk.type());
}

void PTUsingReIdandPF::key_points_center(std::vector<cv::Point2f>& kp, cv::Point2f& ctr)
{
	if (kp.size() == 0) return;
	std::vector<cv::Point2f>::iterator b = kp.begin(), e = kp.end();
	float x = 0., y = 0.;
	for (;b != e; ++b)
	{
		x += b->x;
		y += b->y;
	}
	ctr.x = x/kp.size();
	ctr.y = y/kp.size();
}

void PTUsingReIdandPF::get_proposal_location(std::vector<cv::Point2f>& pre_kp, std::vector<cv::Point2f>& cur_kp, cv::Rect& pre_loc, cv::Rect& cur_proposal_loc)
{
	CV_Assert(pre_kp.size() == cur_kp.size());
	if (pre_kp.size() == 0) {cur_proposal_loc = pre_loc;return;}

	float prex = 0., prey = 0., curx = 0., cury = 0.;
	for (int i = 0;i < pre_kp.size(); ++i)
	{
		prex += pre_kp[i].x;
		prey += pre_kp[i].y;
		curx += cur_kp[i].x;
		cury += cur_kp[i].y;
	}
	prex /= pre_kp.size();
	prey /= pre_kp.size();
	curx /= pre_kp.size();
	cury /= pre_kp.size();		//center of matched keypoints

	cur_proposal_loc.width = pre_loc.width;
	cur_proposal_loc.height = pre_loc.height;
	cur_proposal_loc.x = pre_loc.x + curx - prex;
	cur_proposal_loc.y = pre_loc.y + cury - prey;	//relative center of matched keypoints in cur_proposal_loc is the same in pre_loc
}

double PTUsingReIdandPF::hog_predict(const std::vector<float>& v)
{
		CV_Assert(v.size()<=hogDetector.size());
		float s = hogDetector[hogDetector.size()-1];
		int k = 0;
		std::for_each(v.begin(),v.end(),[&](const float d){s+=d*hogDetector[k];k++;});
		return s;
}

void PTUsingReIdandPF::drawMatches(cv::Mat& img1, std::vector<cv::Point2f>& points1, cv::Mat& img2, std::vector<cv::Point2f>& points2, cv::Mat& res)
{
	CV_Assert(points1.size() == points2.size());
	CV_Assert(img1.size() == img2.size());
	using namespace cv;
	res = Mat(img1.rows, img1.cols*2, img1.type());
	Point offset(img1.cols, 0);
	Rect roi_left(0,0,img1.cols, img1.rows);
	Rect roi_right(img1.cols,0,img1.cols, img1.rows);
	img1.copyTo(res(roi_left));
	img2.copyTo(res(roi_right));
	for (int i = 0; i < points1.size(); ++i)
	{
		circle(res, points1[i], 3, Scalar(0,255,0), 1, 8);
		circle(res, Point(points2[i].x+offset.x, points2[i].y+offset.y), 3, Scalar(0,255,0), 1, 8);
		line(res, points1[i], Point(points2[i].x+offset.x, points2[i].y+offset.y), Scalar(0, 255, 0), 1, 8);
	}
}

void PTUsingReIdandPF::re_id_model_update(pe_re_id::SdalfFeature& sf1, pe_re_id::SdalfFeature& sf2/*new*/, pe_re_id::SdalfFeature& sf3/*output updated*/, double alpha)
{
	CV_Assert(alpha <= 1. && alpha >= 0.);
	sf3.mapkrnl_div3.BUsim = (1-alpha)*sf1.mapkrnl_div3.BUsim+alpha*sf2.mapkrnl_div3.BUsim;
	sf3.mapkrnl_div3.HDanti = (1-alpha)*sf1.mapkrnl_div3.HDanti+alpha*sf2.mapkrnl_div3.HDanti;
	sf3.mapkrnl_div3.head_det = sf1.mapkrnl_div3.head_det;
	sf3.mapkrnl_div3.head_det_flag = sf1.mapkrnl_div3.head_det_flag;
	sf3.mapkrnl_div3.is_ready = sf1.mapkrnl_div3.is_ready;
	sf3.mapkrnl_div3.LEGsim = (1-alpha)*sf1.mapkrnl_div3.LEGsim+alpha*sf2.mapkrnl_div3.LEGsim;
	sf3.mapkrnl_div3.TLanti = (1-alpha)*sf1.mapkrnl_div3.TLanti+alpha*sf2.mapkrnl_div3.TLanti;
	
	sf3.whisto2.is_ready = sf1.whisto2.is_ready;
	sf3.whisto2.whisto = (1-alpha)*sf1.whisto2.whisto+alpha*sf2.whisto2.whisto;
}

void PTUsingReIdandPF::re_id_model_update(pe_re_id::SdalfFeature& sf, std::vector<pe_re_id::SdalfFeature>& p)
{
#ifdef USE_WEIGHTED_AVERAGE_TEMPLATE
	static std::vector<pe_re_id::SdalfFeature> history_probe(probe_needs_+1);
	static int cnt = 1;
	if (cnt == 1) {history_probe[0] = p[0];history_probe[1] = sf; cnt++;}
	else if (cnt <= probe_needs_) {history_probe[cnt] = sf; cnt++;}
	else {history_probe.erase(history_probe.begin()+1); history_probe.push_back(sf);}
	if (p.size() == 1) p.push_back(history_probe[0]);
	p[1].whisto2.whisto = cv::Mat::zeros(p[0].whisto2.whisto.size(), p[0].whisto2.whisto.type());
	for (int i = 0; i < cnt; ++i)
	{
		p[1].whisto2.whisto = p[1].whisto2.whisto + get_weight_(i,cnt)*history_probe[i].whisto2.whisto;
		std::cout << get_weight_(i, cnt) << " ";
	}
	std::cout << std::endl;
#else
	if (p.size() < probe_needs_) p.push_back(sf);
	else {p.erase(p.begin()+1); p.push_back(sf);}
#endif
}
double PTUsingReIdandPF::get_weight_(int i, int n)
{
	if (i == 0) return 1-learning_rate_;
	const static double w1[] = {1.};
	const static double w2[] = {0.2689,0.7311};
	const static double w3[] = {0.1483,0.2889,0.5627};
	const static double w4[] = {0.1015,0.1674,0.2760,0.4551};
	const static double w5[] = {0.0770,0.1148,0.1713,0.2556,0.3813};
	const static double w6[] = {0.0619,0.0864,0.1206,0.1683,0.2349,0.3278};
	const static double w7[] = {0.0518,0.0689,0.0917,0.1220,0.1623,0.2160,0.2874};
	const static double w8[] = {0.0445,0.0571,0.0733,0.0941,0.1208,0.1552,0.1992,0.2558};
	const static double w9[] = {0.0389,0.0486,0.0607,0.0759,0.0947,0.1183,0.1478,0.1845,0.2305};
	const static double w10[] = {0.0347,0.0423,0.0517,0.0631,0.0771,0.0942,0.1151,0.1405,0.1716,0.2096};
	const static double* w[] = {w1,w2,w3,w4,w5,w6,w7,w8,w9,w10};

	return w[n-2][i-1]-(1-learning_rate_)/(n-1);
}

void PTUsingReIdandPF::normalize_(vec_d& data)
{
	double s = 0.;
	std::for_each(data.begin(), data.end(), [&s](double d){s += d;});
	std::for_each(data.begin(), data.end(), [s](double& d){d /= s;});
}

void PTUsingReIdandPF::hog_scores_mapping_(vec_d& scores_)
{
	//#define EXP_COEFF_HOG 2
	//	int l = scores_.size();
	//	double maxs = -DBL_MAX, mins = DBL_MAX;
	//	for (int i = 0; i < l; ++i)
	//	{
	//		if (maxs < scores_[i]) maxs = scores_[i];
	//		if (mins > scores_[i]) mins = scores_[i];
	//	}
	//	std::for_each(scores_.begin(), scores_.end(), [maxs, mins](double& d){d = exp(EXP_COEFF_HOG*(d-mins)/(maxs-mins));});
	std::for_each(scores_.begin(), scores_.end(), [](double& d){d = exp(2*d);});
	normalize_(scores_);
}

void PTUsingReIdandPF::sdalf_scores_mapping_(vec_d& scores)
{
//#define EXP_COEFF_MATCH 2
//	int l = scores.size();
//	double maxs = -DBL_MAX, mins = DBL_MAX;
//	for (int i = 0; i < l; ++i)
//	{
//		if (maxs < scores[i]) maxs = scores[i];
//		if (mins > scores[i]) mins = scores[i];
//	}
//	std::for_each(scores.begin(), scores.end(), [maxs,mins](double& d) {d = exp(EXP_COEFF_MATCH*(d-maxs)/(mins-maxs));});
//	normalize_(scores);

	double s = 0; 
	double coeff = exp_coeff_re_id_;
	std::for_each(scores.begin(), scores.end(), [&s, coeff](double& d){if (d > 0.01) d = exp(coeff*1./(d+DBL_MIN));else d = 0.; s+=d;});
	std::for_each(scores.begin(), scores.end(), [s](double& d){d /= s;});
}

cv::Rect PTUsingReIdandPF::get_particles_bounding_box(std::vector<vec_d>::iterator& b, std::vector<vec_d>::iterator& e)
{
	int minx = INT_MAX, miny = INT_MAX, maxx = 0, maxy = 0;
	for (;b!=e;b++)
	{
		cv::Rect r;
		states2rect(*b, r);
		if (minx > r.x) minx = r.x;
		if (miny > r.y) miny = r.y;
		if (maxx < r.br().x-1) maxx = r.br().x-1;
		if (maxy < r.br().y-1) maxy = r.br().y-1;
	}
	return cv::Rect(minx, miny, maxx-minx+1, maxy-miny+1);
}

#define SCALE_GAUSSIAN_NOISE_VARIANCE 5
void PTUsingReIdandPF::fpt_motion_update(vec_d& particle_pre, vec_d& particle_cur, double meanx, double meany)
{
	double xtmp = particle_pre[0] + meanx;
	if (xtmp < 0)
		particle_cur[0] = xtmp+gen_uniform_noise(0,width_star_/2);
	else if (xtmp > pre_frame.cols)
		particle_cur[0] = xtmp-gen_uniform_noise(0, width_star_/2);
	else
		particle_cur[0] = xtmp+gen_gaussian_noise(0,2);
	if (particle_cur[0] < 0) particle_cur[0] = 0;
	else if (particle_cur[0] > pre_frame.cols) particle_cur[0] = pre_frame.cols;

	double ytmp = particle_pre[1] + meany;
	if (ytmp < 0)
		particle_cur[1] = ytmp+gen_uniform_noise(0,width_star_/2/aspect_ratio_);
	else if (ytmp > pre_frame.rows)
		particle_cur[1] = ytmp-gen_uniform_noise(0,width_star_/2/aspect_ratio_);
	else
		particle_cur[1] = ytmp+gen_gaussian_noise(0,2);
	if (particle_cur[1] < 0) particle_cur[1] = 0;
	else if (particle_cur[1] > pre_frame.rows) particle_cur[1] = pre_frame.rows;

	particle_cur[2] = particle_pre[2] + gen_gaussian_noise(0,width_star_/6);
	if (particle_cur[2] < 10) particle_cur[2] = 10;
	else if (particle_cur[2]/aspect_ratio_ > pre_frame.rows) particle_cur[2] = pre_frame.rows*aspect_ratio_;
	//particle_cur[3] = particle_cur[0]-particle_pre[0]+gen_gaussian_noise(0,width_star_/4);
	//particle_cur[4] = particle_cur[1] - particle_pre[1]+gen_gaussian_noise(0, width_star_/4);
	//particle_cur[3] = meanx + gen_gaussian_noise(0, width_star_/4);
	particle_cur[3] = gen_gaussian_noise(0, 1);
	//particle_cur[4] = meany + gen_gaussian_noise(0, width_star_/4);
	particle_cur[4] = gen_gaussian_noise(0, 1);
}
void PTUsingReIdandPF::rw_motion_update(vec_d& particle_pre, vec_d& particle_cur, double w_star, double vx_star, double vy_star, int frame_w, int frame_h)
{
	double xtmp = particle_pre[0]+particle_pre[3];
	if (xtmp < 0)
		particle_cur[0] = xtmp+gen_uniform_noise(0,w_star/4);
	else if (xtmp > frame_w)
		particle_cur[0] = xtmp-gen_uniform_noise(0, w_star/4);
	else
		particle_cur[0] = xtmp+gen_gaussian_noise(0,w_star/4);
	if (particle_cur[0] < 0) particle_cur[0] = 0;
	else if (particle_cur[0] > frame_w) particle_cur[0] = frame_w;

	double ytmp = particle_pre[1] + particle_pre[4];
	if (ytmp < 0)
		particle_cur[1] = ytmp+gen_uniform_noise(0,w_star/4/aspect_ratio_);
	else if (ytmp > frame_h)
		particle_cur[1] = ytmp-gen_uniform_noise(0,w_star/4/aspect_ratio_);
	else
		particle_cur[1] = ytmp+gen_gaussian_noise(0,w_star/4);
	if (particle_cur[1] < 0) particle_cur[1] = 0;
	else if (particle_cur[1] > frame_h) particle_cur[1] = frame_h;

	/*particle_cur[2] = particle_pre[2] + gen_gaussian_noise(0,3);*/
	particle_cur[2] = w_star + gen_gaussian_noise(0,width_star_/6);
	if (particle_cur[2] < 10) particle_cur[2] = 10;
	else if (particle_cur[2]/aspect_ratio_ > frame_h) particle_cur[2] = frame_h*aspect_ratio_;

	if (particle_pre[0]< w_star/2) particle_cur[3] = particle_pre[3]+gen_uniform_noise(0,w_star/2);
	else if(particle_pre[0] > frame_w-w_star/2) particle_cur[3] = particle_pre[3]-gen_uniform_noise(0,w_star/2);
	//else particle_cur[3] = particle_pre[3]+gen_uniform_noise(std::min(particle_pre[3], vx_star), std::max(particle_pre[3], vx_star));
	//else particle_cur[3] = vx_star+gen_uniform_noise(std::min(particle_pre[3], vx_star), std::max(particle_pre[3], vx_star));
	//else particle_cur[3] = particle_pre[3] + gen_gaussian_noise(0,1);
	else particle_cur[3] = gen_gaussian_noise(0, 3);
	//else particle_cur[3] = gen_gaussian_noise(0, w_star/3);

	if (particle_pre[1]< w_star) particle_cur[4] = particle_pre[4]+gen_uniform_noise(0,w_star/2);
	else if(particle_pre[1] > frame_h-w_star) particle_cur[4] = particle_pre[4]-gen_uniform_noise(0,w_star/2);
	//else particle_cur[4] = particle_pre[4]+gen_uniform_noise(std::min(particle_pre[4], vy_star), std::max(particle_pre[4], vy_star));
	//else particle_cur[4] = vy_star+gen_uniform_noise(std::min(particle_pre[4], vy_star), std::max(particle_pre[4], vy_star));
	//else particle_cur[4] = particle_pre[4] + gen_gaussian_noise(0,1);
	else particle_cur[4] = gen_gaussian_noise(0, 3);
	//else particle_cur[4] = gen_gaussian_noise(0, w_star/3);
}

double PTUsingReIdandPF::filter_width(double d)
{
	static int t = 1;
	static double pt_1 = 0., et = 0, sumet = 0., et_1 = 0.;
	static double Kp=0.2, Ki=0.003, Kd=0.03;
	if (t == 1)		//if the first frame
	{
		pt_1 = d;
		t++;
		return d;
	}
	if (t == 2)
	{
		et = d - pt_1;
		sumet += et;
		pt_1 = pt_1 + (Kp*et+Ki*sumet);
		et_1 = et;
		t++;
		return pt_1;
	}
	et = d - pt_1;
	sumet += et;
	pt_1 = pt_1 + (Kp*et+Ki*sumet+Kd*(et-et_1));
	et_1 = et;
	t++;
	return pt_1;
}

double PTUsingReIdandPF::filter_vx(double d)
{
	static int t = 1;
	static double pt_1 = 0., et = 0, sumet = 0., et_1 = 0.;
	static double Kp=0.2, Ki=0.003, Kd=0.03;
	if (t == 1)		//if the first frame
	{
		pt_1 = d;
		t++;
		return d;
	}
	if (t == 2)
	{
		et = d - pt_1;
		sumet += et;
		pt_1 = pt_1 + (Kp*et+Ki*sumet);
		et_1 = et;
		t++;
		return pt_1;
	}
	et = d - pt_1;
	sumet += et;
	pt_1 = pt_1 + (Kp*et+Ki*sumet+Kd*(et-et_1));
	et_1 = et;
	t++;
	return pt_1;
}

double PTUsingReIdandPF::filter_vy(double d)
{
	static int t = 1;
	static double pt_1 = 0., et = 0, sumet = 0., et_1 = 0.;
	static double Kp=0.2, Ki=0.003, Kd=0.03;
	if (t == 1)		//if the first frame
	{
		pt_1 = d;
		t++;
		return d;
	}
	if (t == 2)
	{
		et = d - pt_1;
		sumet += et;
		pt_1 = pt_1 + (Kp*et+Ki*sumet);
		et_1 = et;
		t++;
		return pt_1;
	}
	et = d - pt_1;
	sumet += et;
	pt_1 = pt_1 + (Kp*et+Ki*sumet+Kd*(et-et_1));
	et_1 = et;
	t++;
	return pt_1;
}

void PTUsingReIdandPF::drawMatches(cv::Mat& img1, std::vector<cv::Point2f>& points1, std::vector<vec_d>& ptc1, cv::Mat& img2, std::vector<cv::Point2f>& points2,std::vector<vec_d>& ptc2, cv::Mat& res)
{
	CV_Assert(points1.size() == points2.size());
	CV_Assert(img1.size() == img2.size());
	using namespace cv;
	res = Mat(img1.rows, img1.cols*2, img1.type());
	Point offset(img1.cols, 0);
	Rect roi_left(0,0,img1.cols, img1.rows);
	Rect roi_right(img1.cols,0,img1.cols, img1.rows);
	img1.copyTo(res(roi_left));
	img2.copyTo(res(roi_right));
	for (int i = 0; i < points1.size(); ++i)
	{
		circle(res, points1[i], 3, Scalar(0,255,0), 1, 8);
		circle(res, Point(points2[i].x+offset.x, points2[i].y+offset.y), 3, Scalar(0,255,0), 1, 8);
		//line(res, points1[i], Point(points2[i].x+offset.x, points2[i].y+offset.y), Scalar(0, 255, 0), 1, 8);
	}

	vector<vec_d>::iterator iteb1 = ptc1.begin(), itee1 = ptc1.begin()+10,//+ptc1.size()*random_walk_ratio_,
		iteb2 = ptc2.begin(), itee2 = ptc2.begin()+10;//+ptc2.size()*random_walk_ratio_;
	for (;iteb1 != itee1; iteb1++,iteb2++)
	{
		//Rect r1, r2;
		//states2rect(*iteb1, r1);
		//states2rect(*iteb2, r2);
		//rectangle(res, r1, Scalar(255,0,0), 2);
		//rectangle(res, r2.tl()+offset, r2.br()+offset-Point(1,1), Scalar(255,0,0), 2);
	}
	iteb1 = itee1;iteb2 = itee2;itee1 += 5;//= ptc1.end();
	for (;iteb1 != itee1; iteb1++,iteb2++)
	{
		//Rect r1, r2;
		//states2rect(*iteb1, r1);
		//states2rect(*iteb2, r2);
		//rectangle(res, r1, Scalar(0,255,255),2);
		//rectangle(res, r2.tl()+offset, r2.br()+offset-Point(1,1), Scalar(0,255,255),2);
	}
}