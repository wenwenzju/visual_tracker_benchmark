#include "tracking_using_re_id_and_pf.h"
#include "imwrite.h"

PTUsingReIdandPF::PTUsingReIdandPF(const std::string& des, int states, int particles) : ParticleFilter(states, particles), sdalf(des), rng(time(0)), no_matched_orb(false),
	termcrit_(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03), sub_pix_winSize(10,10), win_size(31,31), MAX_COUNT(500)
{
	dx=dy=0;dw = 0;probe_needs = 5;
	sigma_[0] = 10.;
	sigma_[1] = 10.;
	sigma_[2] = 2.;
	hogDetector = cv::HOGDescriptor::getDefaultPeopleDetector();
}

PTUsingReIdandPF::PTUsingReIdandPF(int states /* = 3 */, int particles /* = 52 */, bool uofl /* = true */, bool uo /* = true */, double lr /* = 0.4 */, double sx /* = 20 */, 
	double sy /* = 10 */, double sw /* = 8 */, double ecri /* = 0.5 */, double riw /* = 0.8 */, double hpx /* = 1.2 */, int tn /* = 4 */, double ar /* = 0.43*/)
	: ParticleFilter(states, particles), sdalf(), rng(time(0)), no_matched_orb(false),
	termcrit_(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03), sub_pix_winSize(10,10), win_size(31,31), MAX_COUNT(500),
	use_optical_flow_lk_(uofl), update_online_(uo), learning_rate_(lr), sigmax_(sx), sigmay_(sy), sigmaw_(sw), exp_coeff_re_id_(ecri), re_id_weight_(riw), hog_particle_expand_(hpx), thread_num_(tn), aspect_ratio_(ar)
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
	if (first_loc.width<first_loc.height*aspect_ratio_)
	{
		init_loc.width = cvRound(first_loc.height*aspect_ratio_);
		int x = first_loc.x-(init_loc.width-first_loc.width)/2;
		init_loc.x = x<0?0:x;
	}
	else
	{
		init_loc.height = cvRound(first_loc.width/aspect_ratio_);
		int y = first_loc.y - (init_loc.height - first_loc.height)/2;
		init_loc.y = y<0?0:y;
	}
	previous_loc = init_loc;
	pre_frame = first_img.clone();
	if (!sdalf.probe_ready)
	{
		cv::Mat tmp;
		cv::resize(pre_frame(init_loc), tmp, cv::Size(64,128));
		sdalf.probe_feature_extract(tmp);
	}
	init_probe = sdalf.probe[0];

	//new add lk optical flow
	if (use_optical_flow_lk_)
	{
		cv::Mat msk;
		generate_mask(msk, first_img.rows, first_img.cols, first_loc);
		cv::Mat gray;
		cv::cvtColor(first_img, gray, CV_BGR2GRAY);
		cv::goodFeaturesToTrack(gray, lk_points_[0], MAX_COUNT, 0.01, 10, msk, 3, 0, 0.04);
		if (lk_points_[0].size())
			cornerSubPix(gray, lk_points_[0], sub_pix_winSize, cv::Size(-1,-1), termcrit_);
	}

	////////////////////////////////////new added self-adaptive sigma
	//sigma_[0] = 1.*previous_loc.width/3.;
	sigma_[0] = 1.*previous_loc.width/3. < sigma_max_[0] ? 1.*previous_loc.width/3. : sigma_max_[0];
	//sigma_[1] = 1.*previous_loc.width/6.;
	sigma_[1] = 1.*previous_loc.width/6. < sigma_max_[1] ? 1.*previous_loc.width/6. : sigma_max_[1];
	//sigma_[2] = 1.*previous_loc.width/6.;
	sigma_[2] = 1.*previous_loc.width/6. < sigma_max_[2] ? 1.*previous_loc.width/6. : sigma_max_[2];
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

double PTUsingReIdandPF::p_yk_given_xk(vec_d& xk, vec_d& yk)
{
	cv::Rect r;
	states2rect(xk, r);
	std::vector<cv::Mat> g(1);
	cv::Rect tmp;
	tmp = r&cv::Rect(0,0,cur_frame->cols, cur_frame->rows);
	double coeff = 1.*tmp.area()/previous_loc.area();
	coeff = exp(-abs(coeff-1));
	coeff = 1.0;
	if (tmp.area() <= 0)
		return DBL_MIN;
	cv::resize(cur_frame->operator()(tmp), g[0], cv::Size(64, 128));
	sdalf.gallery_feautre_extract(g);
	double d;
	sdalf.feature_match(&d);

	std::vector<float> descrip;
	hog_.compute(g[0],descrip);
	double det_coeff = hog_predict(descrip);
	std::cout << "ith particle score in acf detector: " << det_coeff << std::endl;
	double nd = normal_distribution(det_coeff, 5, 2);
	if (det_coeff >= 5.) nd = normal_distribution(5., 5, 2);


	//return coeff*(1-DETWEIGHT)*exp(-EXPCOEFF*d)+DETWEIGHT*exp(EXPCOEFF*(max_sim-1));
	//return coeff*DETWEIGHT*normal_distribution(d, 0., 0.35) + (1-coeff*DETWEIGHT)*normal_distribution(max_sim, 1, 0.35);
	return coeff*re_id_weight_*normal_distribution(d, 0., 0.35) + (1-coeff*re_id_weight_)*nd;
}

void PTUsingReIdandPF::p_yk_given_xk_multi_thread(std::vector<vec_d>::iterator s_xk, 
	std::vector<double>::iterator s_pro, 
	std::vector<double>::iterator s_item0, 
	std::vector<double>::iterator s_item1, 
	double* si0, double* si1, 
	int particles_per_thread)
{
	for (int i = 0; i < particles_per_thread; s_xk++, s_pro++, s_item0++, s_item1++, i++)
	{
		cv::Rect r;
		states2rect(*s_xk, r);
		
		std::vector<cv::Mat> g(1);
		cv::Rect tmp;
		tmp = r&cv::Rect(0,0,cur_frame->cols, cur_frame->rows);

		cv::resize(cur_frame->operator()(tmp), g[0], cv::Size(64, 128));
		std::vector<pe_re_id::SdalfFeature> gallery_feature;
		sdalf.gallery_feautre_extract(g, gallery_feature);
		double d;
		sdalf.feature_match(gallery_feature, &d);
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
		copyMakeBorder(hog_p, hog_p, hog_roi.x-hog_r.x, hog_r.br().y-hog_roi.br().y, hog_roi.x-hog_r.x, hog_r.br().x-hog_roi.br().x, cv::BORDER_REPLICATE);
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
		std::vector<double>::iterator s_pro = pro.begin()+i*particles_per_thread;
		std::vector<double>::iterator s_item0 = item0.begin()+i*particles_per_thread;
		std::vector<double>::iterator s_item1 = item1.begin()+i*particles_per_thread;

		if (i != thread_num_-1)
			threads[i] = boost::thread(boost::bind(&PTUsingReIdandPF::p_yk_given_xk_multi_thread, this, 
			s_xk, s_pro, s_item0, s_item1, &si0, &si1, particles_per_thread));
		else
		{
			threads[i] = boost::thread(boost::bind(&PTUsingReIdandPF::p_yk_given_xk_multi_thread, this, 
				s_xk, s_pro, s_item0, s_item1, &si0, &si1, particles_per_thread+particles_num_%thread_num_));
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

	//x position
	if (previous_loc.x < 0)
	{
		boost::exponential_distribution<double> ed(3./pre_frame.cols);
		uk.push_back(ed(rng));
	}
	else if (previous_loc.br().x-1 > pre_frame.cols)
	{
		boost::exponential_distribution<double> ed(3./pre_frame.cols);
		uk.push_back(-ed(rng));
	}
	else
	{
		boost::normal_distribution<double> nd(0, sigma_[0]);
		uk.push_back(nd(rng));
	}

	//y position
	if (previous_loc.y < 0)
	{
		boost::exponential_distribution<double> ed(3./pre_frame.rows);
		uk.push_back(ed(rng));
	}
	if (previous_loc.br().y-1 > pre_frame.rows)
	{
		boost::exponential_distribution<double> ed(3./pre_frame.rows);
		uk.push_back(-ed(rng));
	}
	else
	{
		boost::normal_distribution<double> nd(0, sigma_[1]);
		uk.push_back(nd(rng));
	}

	//scale
	boost::normal_distribution<double> nd(0, sigma_[2]);
	uk.push_back(nd(rng));

	//for (int i = 0; i < 3; ++i)
	//{
	//	boost::normal_distribution<double> nd(0, sigma_[i]);
	//	
	//	uk.push_back(nd(rng));
	//}
}

void PTUsingReIdandPF::gen_obs_noise(vec_d& vk)
{

}

void PTUsingReIdandPF::sample_from_q(vec_d& xkm1, vec_d& yk, int n, vec_d& x)
{
	//version 3.0 use orb to propose proposal locations
	vec_d uk;
	//do
	//{
		rect2states(proposal_loc_, x);
		gen_sys_noise(uk);
		x[0] += uk[0];
		x[1] += uk[1];
		x[2] += uk[2];

	//}/*while(x[0] - x[2]/2 < 0 || x[1] - x[2]/aspect_ratio_/2 < 0 || x[2] < 10);*/
	//while(x[0] < 0 || x[1] < 0 || x[0] > pre_frame.cols || x[1] > pre_frame.rows || x[2] < 10 || x[2] > pre_frame.cols || x[2]/aspect_ratio_ > pre_frame.rows);
		if (x[0] < 0) x[0] = 0;
		else if (x[0] > pre_frame.cols) x[0] = pre_frame.cols;
		if (x[1] < 0) x[1] = 0;
		else if (x[1] > pre_frame.rows) x[1] = pre_frame.rows;
		if (x[2] < 10) x[2] = 10;
		else if (x[2]/aspect_ratio_ > pre_frame.rows) x[2] = pre_frame.rows*aspect_ratio_;
}

double PTUsingReIdandPF::update_weight(double wkm1, vec_d& xkm1, vec_d& xk, vec_d& yk)
{
	//version 4.0 only kinematic model is used
	return 1./particles_num_*p_yk_given_xk(xk, yk);
	//return p_yk_given_xk(xk, yk)/q_xk_given_xkm1_yk(xkm1, yk, xk);
}

void PTUsingReIdandPF::update_weight(vec_d& wkm1, std::vector<vec_d>& xkm1, std::vector<vec_d>& xk, vec_d& yk, vec_d& wk)
{
	std::vector<double> pro;		//normalized probability of p_yk_given_xk
	p_yk_given_xk(xk, yk, pro);
	wk = vec_d(xk.size(), 0.);
	for (int i = 0; i < xk.size(); ++i)
		//wk[i] = wkm1[i]*pro[i];
		wk[i] = 1./particles_num_*pro[i];
}

cv::Rect PTUsingReIdandPF::track(cv::Mat& frame)
{
	static int frame_num = 1;
	cur_frame = &frame;

	cv::Mat pre_gray, cur_gray;
	if (use_optical_flow_lk_)
	{
	//**************new added lk optical flow
		std::vector<uchar> status;
		std::vector<float> err;
		
		cv::cvtColor(pre_frame, pre_gray, CV_BGR2GRAY);
		cv::cvtColor(frame, cur_gray, CV_BGR2GRAY);
		if (lk_points_[0].size())
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

		get_proposal_location(lk_points_[0], lk_points_[1], previous_loc, proposal_loc_);
	}
	else
		proposal_loc_ = previous_loc;

	vec_d xk;
	std::vector<vec_d> p;
	filter(vec_d(), xk, p);

	states2rect(xk, previous_loc);

	//////////////////////////
	//particle_features_ << previous_loc.x << "," << previous_loc.y << "," << previous_loc.width << "," << previous_loc.height << std::endl;

	if (!update_online_)
	{
		if (!sdalf.probe_ready)
		{
			if (sdalf.probe.size() == probe_needs) {sdalf.save_probe();frame_num = 4;}
			if (frame_num == 3)
			{
				frame_num = 0;
				cv::Mat tmp;
				cv::resize(frame(previous_loc&cv::Rect(0,0,frame.cols, frame.rows)), tmp, cv::Size(64, 128));
				sdalf.probe_feature_extract(tmp);
			}
		}
		++frame_num;
	}
	else
	{
		cv::Mat tmp;
		cv::resize(frame(previous_loc&cv::Rect(0,0,frame.cols, frame.rows)), tmp, cv::Size(64, 128));

		std::vector<pe_re_id::SdalfFeature> gallery_feature;
		sdalf.gallery_feautre_extract(std::vector<cv::Mat>(1, tmp), gallery_feature);
		if (sdalf.probe.size() == 1) sdalf.probe.push_back(gallery_feature[0]);
		re_id_model_update(init_probe, gallery_feature[0], sdalf.probe[1], learning_rate_);

		//sdalf.probe_feature_extract(tmp);
		//if (sdalf.probe.size() > probe_needs) sdalf.probe.erase(sdalf.probe.begin()+1);
	}

	pre_frame = frame.clone();
	//**************new added lk optical flow
	if (use_optical_flow_lk_)
	{
		cv::Mat msk;
		lk_points_[0].clear();
		lk_points_[1].clear();
		generate_mask(msk, cur_frame->rows, cur_frame->cols, previous_loc);
		cv::goodFeaturesToTrack(cur_gray, lk_points_[0], MAX_COUNT, 0.01, 10, msk, 3, 0, 0.04);
		if (lk_points_[0].size())
			cornerSubPix(cur_gray, lk_points_[0], sub_pix_winSize, cv::Size(-1,-1), termcrit_);
	}

	////////////////////////////////////new added self-adaptive sigma
	//sigma_[0] = 1.*previous_loc.width/3.;
	sigma_[0] = 1.*previous_loc.width/3. < sigma_max_[0] ? 1.*previous_loc.width/3. : sigma_max_[0];
	//sigma_[1] = 1.*previous_loc.width/6.;
	sigma_[1] = 1.*previous_loc.width/6. < sigma_max_[1] ? 1.*previous_loc.width/6. : sigma_max_[1];
	//sigma_[2] = 1.*previous_loc.width/6.;
	sigma_[2] = 1.*previous_loc.width/6. < sigma_max_[2] ? 1.*previous_loc.width/6. : sigma_max_[2];

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

void PTUsingReIdandPF::key_points_center(std::vector<cv::KeyPoint>& kp, cv::Point2f& ctr)
{
	if (kp.size() == 0) return;
	std::vector<cv::KeyPoint>::iterator b = kp.begin(), e = kp.end();
	float x = 0., y = 0.;
	for (;b != e; ++b)
	{
		x += b->pt.x;
		y += b->pt.y;
	}
	ctr.x = x/kp.size();
	ctr.y = y/kp.size();
}

void PTUsingReIdandPF::get_proposal_location(std::vector<cv::Point2f>& pre_kp, std::vector<cv::Point2f>& cur_kp, cv::Rect& pre_loc, cv::Rect& cur_proposal_loc)
{
	CV_Assert(pre_kp.size() == cur_kp.size());
	if (pre_kp.size() == 0) {cur_proposal_loc = pre_loc; return;}

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

void PTUsingReIdandPF::normalize_(vec_d& data)
{
	double s = 0.;
	std::for_each(data.begin(), data.end(), [&s](double d){s += d;});
	std::for_each(data.begin(), data.end(), [s](double& d){d /= s;});
}

void PTUsingReIdandPF::hog_scores_mapping_(vec_d& scores_)
{
	//#define EXP_COEFF_HOG 3
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
	double s = 0; 
	double coeff = exp_coeff_re_id_;
	std::for_each(scores.begin(), scores.end(), [&s, coeff](double& d){d = exp(coeff*1./(d+DBL_MIN));s+=d;});
	std::for_each(scores.begin(), scores.end(), [s](double& d){d /= s;});
}