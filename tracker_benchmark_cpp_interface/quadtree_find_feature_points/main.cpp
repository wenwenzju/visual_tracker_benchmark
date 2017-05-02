//-s D:\data_seq\Jogging -x 111 -y 98 -w 25 -h 101
//-s D:\data_seq\David3 -x 83 -y 200 -w 35 -h 131

#include "opencv2/opencv.hpp"
#include "boost/program_options.hpp"
#include "boost/random.hpp"
#include <string>
#include <iomanip>
#include <fstream>

#define RANDOM_WALK_PARTICLES_RATIO 0.5

using namespace cv;
using namespace boost::program_options;
using namespace std;

typedef vector<double> vec_d;
boost::mt19937 rng(time(0));
double aspect_ratio = 0.43;
void gen_sys_noise(vec_d& uk, double* sigma_)
{
	uk.clear();

	for (int i = 0; i < 3; ++i)
	{
		boost::normal_distribution<double> nd(0, sigma_[i]);

		uk.push_back(nd(rng));
	}
}

double gen_gaussian_noise(double m, double s)
{
	boost::normal_distribution<double> nd(m, s);
	return nd(rng);
}
double gen_uniform_noise(double lb, double ub)
{
	boost::uniform_real<double> ur(lb, ub);
	return ur(rng);
}

void states2rect(vec_d& x, cv::Rect& r)
{
	int xx = x[0] - x[2]/2;
	int yy = x[1] - cvRound(x[2]/aspect_ratio)/2;
	r = cv::Rect(xx,yy,(int)x[2],cvRound(x[2]/aspect_ratio));
}

void rect2states(cv::Rect& r, vec_d& x)
{
	x.clear();
	x.push_back(r.x+r.width/2);
	x.push_back(r.y+r.height/2);
	x.push_back(r.width);
}

vec_d generate_particle(vec_d& x, double* sigma_)
{
	vec_d uk;
	vec_d xk;
	do
	{
		xk.clear();
		gen_sys_noise(uk, sigma_);
		xk.push_back(x[0] + uk[0]);
		xk.push_back(x[1] + uk[1]);
		xk.push_back(x[2] + uk[2]);
		xk.push_back(x[3] + gen_gaussian_noise(0,5));
		xk.push_back(x[4] + gen_gaussian_noise(0,5));
	}while(x[0] - x[2]/2 < 0 || x[1] - x[2]/aspect_ratio/2 < 0 || x[2] < 10);
	return xk;
}

Rect get_particles_bounding_box(vector<vec_d>& particles)
{
	int minx = INT_MAX, miny = INT_MAX, maxx = 0, maxy = 0;
	for (int i = 0; i < particles.size(); ++i)
	{
		Rect r;
		states2rect(particles[i], r);
		if (minx > r.x) minx = r.x;
		if (miny > r.y) miny = r.y;
		if (maxx < r.br().x) maxx = r.br().x;
		if (maxy < r.br().y) maxy = r.br().y;
	}
	return Rect(minx, miny, maxx-minx+1, maxy-miny+1);
}

Rect get_particles_bounding_box(vector<vec_d>::iterator& b, vector<vec_d>::iterator& e)
{
	int minx = INT_MAX, miny = INT_MAX, maxx = 0, maxy = 0;
	for (;b!=e;b++)
	{
		Rect r;
		states2rect(*b, r);
		if (minx > r.x) minx = r.x;
		if (miny > r.y) miny = r.y;
		if (maxx < r.br().x) maxx = r.br().x;
		if (maxy < r.br().y) maxy = r.br().y;
	}
	return Rect(minx, miny, maxx-minx+1, maxy-miny+1);
}

void generate_mask(cv::Mat& msk,int rows, int cols, cv::Rect& region)
{
	msk = cv::Mat::zeros(rows, cols, CV_8UC1);
	cv::Rect r = region & cv::Rect(0,0,cols, rows);
	msk(r) = cv::Mat::ones(r.height, r.width, msk.type());
}

//void sample_from_q(vector<vec_d>& pold, vector<vec_d>& pnew, vector<Point2f>& points1, vector<Point2f>& points2)
//{
//	pnew = pold;
//	boost::normal_distribution<double> nd(0, 5);
//	for (int i = 0; i < pold.size(); ++i)
//	{
//		Rect r; states2rect(pold[i], r);
//		vector<int> idx;
//		for (int j = 0; j < points1.size(); ++j)
//		{
//			if (r.contains(points1[j])) idx.push_back(j);
//		}
//		if (idx.size() == 0) continue;
//		Point2f c1(0,0), c2(0,0);
//		for (int j = 0; j < idx.size(); ++j)
//		{
//			c1 += points1[idx[j]];
//			c2 += points2[idx[j]];
//		}
//		c1 *= 1./idx.size();
//		c2 *= 1./idx.size();
//		pnew[i][0] += (c2.x-c1.x);
//		pnew[i][1] += (c2.y-c1.y);
//		//pnew[i][2] += nd(rng);
//	}
//}

double width_min = 10., vx_star = 3., vy_star = 3., width_star = 30;
int frame_w = 480, frame_h = 640;
void fpt_motion_update(vec_d& particle_pre, vec_d& particle_cur, double meanx, double meany)
{
	particle_cur[0] = particle_pre[0]+meanx+gen_gaussian_noise(0,5);
	particle_cur[1] = particle_pre[1]+meany+gen_gaussian_noise(0,5);
	do 
	{
		particle_cur[2] = particle_pre[2] + gen_gaussian_noise(0,3);
	} while (particle_cur[2] < width_min || particle_cur[2] > frame_h);
	particle_cur[3] = particle_cur[0]-particle_pre[0]+gen_gaussian_noise(0,width_star/4);
	particle_cur[4] = particle_cur[1] - particle_pre[1]+gen_gaussian_noise(0, width_star/4);
}
void rw_motion_update(vec_d& particle_pre, vec_d& particle_cur, double w_star, double vx_star, double vy_star, int frame_w, int frame_h)
{
	particle_cur[0] = particle_pre[0]+particle_pre[3]+gen_gaussian_noise(0,5);
	particle_cur[1] = particle_pre[1]+particle_pre[4]+gen_gaussian_noise(0,5);
	do 
	{
		particle_cur[2] = particle_pre[2] + gen_gaussian_noise(0,3);
	} while (particle_cur[2] < width_min || particle_cur[2] > frame_h);
	if (particle_pre[0]< w_star/2) particle_cur[3] = particle_pre[3]+gen_uniform_noise(0,w_star);
	else if(particle_pre[0] > frame_w-w_star/2) particle_cur[3] = particle_pre[3]-gen_uniform_noise(0,w_star);
	else particle_cur[3] = particle_pre[3]+gen_uniform_noise(min(particle_pre[3], vx_star), max(particle_pre[3], vx_star));

	if (particle_pre[1]< w_star) particle_cur[4] = particle_pre[4]+gen_uniform_noise(0,w_star);
	else if(particle_pre[1] > frame_h-w_star) particle_cur[4] = particle_pre[4]-gen_uniform_noise(0,w_star);
	else particle_cur[4] = particle_pre[4]+gen_uniform_noise(min(particle_pre[4], vy_star), max(particle_pre[4], vy_star));
}
#define NSIGMA 2
void sample_from_q(vector<vec_d>& pold, vector<vec_d>& pnew, vector<Point2f>& points1, vector<Point2f>& points2)
{
	CV_Assert(points1.size() == points2.size());
	vector<double> v(points1.size()), vx(points1.size()), vy(points1.size());
	double sumv = 0., sumv2 = 0.;
	for (int i = 0; i < points1.size(); ++i)
	{
		Point2f tmp = points2[i] - points1[i];
		vx[i] = tmp.x;
		vy[i] = tmp.y;
		v[i] = sqrt(tmp.x*tmp.x+tmp.y*tmp.y);
		sumv += v[i];
		sumv2 += v[i]*v[i];
	}
	double meanv = sumv/v.size(), stdv = sqrt(sumv2/v.size()-meanv*meanv);
	double lb = meanv-NSIGMA*stdv, ub = meanv+NSIGMA*stdv;
	double meanx = 0., meany = 0.;
	int cnt = 0;
	for (int i = 0; i < v.size(); ++i)
	{
		if (v[i] < lb || v[i] > ub) continue;
		meanx += vx[i];
		meany += vy[i];
		cnt ++;
	}
	meanx /= cnt; meany /= cnt;
	pnew = pold;
	vector<vec_d>::iterator fpt_ite_old = pold.begin()+pold.size()*RANDOM_WALK_PARTICLES_RATIO,
		fpt_ite_new = pnew.begin()+pnew.size()*RANDOM_WALK_PARTICLES_RATIO;
	for (;fpt_ite_old != pold.end(); fpt_ite_old++, fpt_ite_new++)
	{
		//(*fpt_ite_new)[0] = (*fpt_ite_old)[0] + meanx + gen_fpt_noise(0, 5);
		//(*fpt_ite_new)[1] = (*fpt_ite_old)[1] + meany + gen_fpt_noise(0, 5);
		//(*fpt_ite_new)[2] = (*fpt_ite_old)[2] ;//+ gen_fpt_noise(0,5);
		fpt_motion_update(*fpt_ite_old, *fpt_ite_new, meanx, meany);
	}

	vector<vec_d>::iterator rw_ite_old = pold.begin(),rw_ite_old_end = pold.begin()+pold.size()*RANDOM_WALK_PARTICLES_RATIO,
		rw_ite_new = pnew.begin();
	for (; rw_ite_old != rw_ite_old_end; rw_ite_old++, rw_ite_new++)
	{
		rw_motion_update(*rw_ite_old, *rw_ite_new, width_star, vx_star, vy_star, frame_w, frame_h);
	}
}

double pid_filter(double det)
{
	static int t = 1;
	static double pt_1 = 0., et = 0, sumet = 0., et_1 = 0.;
	static double Kp=0.2, Ki=0.003, Kd=0.03;
	if (t == 1)		//if the first frame
	{
		pt_1 = det;
		t++;
		return det;
	}
	if (t == 2)
	{
		et = det - pt_1;
		sumet += et;
		pt_1 = pt_1 + (Kp*et+Ki*sumet);
		et_1 = et;
		t++;
		return pt_1;
	}
	et = det - pt_1;
	sumet += et;
	pt_1 = pt_1 + (Kp*et+Ki*sumet+Kd*(et-et_1));
	et_1 = et;
	t++;
	return pt_1;
}

void mean_particles(vector<vec_d>& particles, vec_d& mean_p)
{
	int np = particles.size();
	mean_p = vec_d(particles[0].size(), 0);
	for (int i = 0; i < np; ++i)
	{
		mean_p[0] += particles[i][0];
		mean_p[1] += particles[i][1];
		mean_p[2] += particles[i][2];
	}
	mean_p[0] /= np; mean_p[1] /= np; mean_p[2] /= np;
	static double x_pre = mean_p[0], y_pre = mean_p[1];
	mean_p[3] = pid_filter(mean_p[0] - x_pre);
	mean_p[4] = pid_filter(mean_p[1] - y_pre);
	x_pre = mean_p[0];
	y_pre = mean_p[1];
}

void drawMatches(cv::Mat& img1, std::vector<cv::Point2f>& points1, std::vector<vec_d>& ptc1, cv::Mat& img2, std::vector<cv::Point2f>& points2,std::vector<vec_d>& ptc2, cv::Mat& res)
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

	vector<vec_d>::iterator iteb1 = ptc1.begin(), itee1 = ptc1.begin()+ptc1.size()*RANDOM_WALK_PARTICLES_RATIO,
		iteb2 = ptc2.begin(), itee2 = ptc2.begin()+ptc2.size()*RANDOM_WALK_PARTICLES_RATIO;
	for (;iteb1 != itee1; iteb1++,iteb2++)
	{
		Rect r1, r2;
		states2rect(*iteb1, r1);
		states2rect(*iteb2, r2);
		rectangle(res, r1, Scalar(255,0,0));
		rectangle(res, r2.tl()+offset, r2.br()+offset, Scalar(255,0,0));
	}
	iteb1 = itee1;iteb2 = itee2;itee1 = ptc1.end();
	for (;iteb1 != itee1; iteb1++,iteb2++)
	{
		Rect r1, r2;
		states2rect(*iteb1, r1);
		states2rect(*iteb2, r2);
		rectangle(res, r1, Scalar(0,0,255));
		rectangle(res, r2.tl()+offset, r2.br()+offset, Scalar(0,0,255));
	}
// 	for (int i = 0; i < ptc1.size(); ++i)
// 	{
// 		Rect r1, r2;
// 		states2rect(ptc1[i], r1);
// 		states2rect(ptc2[i], r2);
// 		rectangle(res, r1, Scalar(255,0,0));
// 		rectangle(res, r2.tl()+offset, r2.br()+offset, Scalar(255,0,0));
// 	}
}

void save_particles(std::vector<vec_d>& p, std::ofstream& f)
{
	for (int i = 0; i < p.size(); ++i)
	{
		for (int j = 0; j < p[i].size()-1; ++j) f << p[i][j] << " ";
		f << p[i][p[i].size()-1] << endl;
	}
}
void save_feature_points(std::vector<Point2f>* p, std::ofstream& f)
{
	for (int i = 0; i < p[0].size(); ++i)
	{
		f << p[0][i].x << " " << p[0][i].y << " ";
	}
	f << endl;

	for (int i = 0; i < p[1].size(); ++i)
	{
		f << p[1][i].x << " " << p[1][i].y << " ";
	}
	f << endl;
}

int main(int argc, char* argv[])
{
	options_description desc("find features points in rectangle using quadtree");
	desc.add_options()
		("help", "help messages")
		("seq,s", value<string>(), "sequences path")
		("init_x,x", value<int>(), "initial x")
		("init_y,y", value<int>(), "initial y")
		("init_w,w", value<int>(), "initial w")
		("init_h,h", value<int>(), "initial h")
		;
	string seq_path;
	Rect init;
	variables_map vm;
	try
	{
		store(parse_command_line(argc, argv, desc), vm);
		notify(vm);
		if (vm.count("help")) 
		{
			cout << desc << endl;
			return EXIT_SUCCESS;
		}
		if (vm.count("seq")) seq_path = vm["seq"].as<string>();
		if (seq_path[seq_path.size()-1] != '\\' && seq_path[seq_path.size()-1] != '/')
			seq_path += "/";
		if (vm.count("init_x")) init.x = vm["init_x"].as<int>();
		if (vm.count("init_y")) init.y = vm["init_y"].as<int>();
		if (vm.count("init_w")) init.width = vm["init_w"].as<int>();
		if (vm.count("init_h")) init.height = vm["init_h"].as<int>();
		width_star = init.width;/////////////////////////////

		ofstream pf("particle_states.txt");
		ofstream fp_file("feature_points.txt");
		vec_d initx; rect2states(init, initx);initx.push_back(0.);initx.push_back(0.);
		double sigma_[3];
		double sigma_max_[3] = {20,20,10};
		sigma_[0] = 1.*init.width/3. < sigma_max_[0] ? 1.*init.width/3. : sigma_max_[0];
		sigma_[1] = 1.*init.width/6. < sigma_max_[1] ? 1.*init.width/6. : sigma_max_[1];
		sigma_[2] = 1.*init.width/6. < sigma_max_[2] ? 1.*init.width/6. : sigma_max_[2];
		int pn = 100;
		vector<vec_d> particles, new_particles;
		for (int i = 0; i < pn; i++) particles.push_back(generate_particle(initx, sigma_));
		vec_d mp;
		mean_particles(particles, mp);
		save_particles(particles,pf);
		Mat msk;
		vector<Point2f> lk_points_[2];
		int MAX_COUNT = 500;
		TermCriteria termcrit_(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
		Size sub_pix_winSize(10,10), win_size(31,31);

		cv::Mat preimage = cv::imread(seq_path+"0001.jpg");
		frame_h = preimage.rows;frame_w = preimage.cols;/////////////////////////////////////
		vx_star = 0.; vy_star = 0.;
		/*generate_mask(msk, preimage.rows, preimage.cols, get_particles_bounding_box(particles));*/
		generate_mask(msk, preimage.rows, preimage.cols, get_particles_bounding_box(particles.begin()+particles.size()*RANDOM_WALK_PARTICLES_RATIO, particles.end()));
		cv::Mat pregray, curgray;
		cv::cvtColor(preimage, pregray, CV_BGR2GRAY);
		cv::goodFeaturesToTrack(pregray, lk_points_[0], MAX_COUNT, 0.01, 10, msk, 3, 0, 0.04);
		cornerSubPix(pregray, lk_points_[0], sub_pix_winSize, cv::Size(-1,-1), termcrit_);

		for (int i = 2; ; ++i) {
			stringstream ss;
			ss << seq_path;
			ss << setfill('0') << setw(4) << i;
			ss << ".jpg";
			string imgname;
			ss >> imgname;

			cv::Mat image = cv::imread(imgname);
			if (image.empty()) break;

			std::vector<uchar> status;
			std::vector<float> err;

			cv::cvtColor(image, curgray, CV_BGR2GRAY);
			lk_points_[0].clear();
			cv::calcOpticalFlowPyrLK(pregray, curgray, lk_points_[0], lk_points_[1], status, err, win_size,
				3, termcrit_, 0, 0.001);
			size_t j, k;
			for( j = k = 0; j < lk_points_[1].size(); j++ )
			{
				if( !status[j] )
					continue;

				lk_points_[1][k] = lk_points_[1][j];
				lk_points_[0][k++] = lk_points_[0][j];
			}
			lk_points_[1].resize(k);
			lk_points_[0].resize(k);
			save_feature_points(lk_points_, fp_file);
			sample_from_q(particles, new_particles, lk_points_[0], lk_points_[1]);
			save_particles(new_particles, pf);
			mean_particles(new_particles, mp);
			width_star = mp[2]; vx_star = mp[3]; vy_star = mp[4];

			//show result
			Mat toshow;
			drawMatches(preimage, lk_points_[0], particles, image, lk_points_[1], new_particles, toshow);
			imshow("result", toshow);
			waitKey(0);

			swap(particles, new_particles);
			swap(preimage, image);
			swap(pregray, curgray);
			swap(lk_points_[0], lk_points_[1]);

			generate_mask(msk, image.rows, image.cols, get_particles_bounding_box(particles));
			cv::goodFeaturesToTrack(pregray, lk_points_[0], MAX_COUNT, 0.01, 10, msk, 3, 0, 0.04);
			cornerSubPix(pregray, lk_points_[0], sub_pix_winSize, cv::Size(-1,-1), termcrit_);
		}
	}
	catch (invalid_command_line_syntax&)
	{
		cout << "syntax error" << endl;
		cout << desc << endl;
	}
	catch (boost::bad_lexical_cast&)
	{
		cout << "lexical error" << endl;
		cout << desc << endl;
	}
	catch (...)
	{
		cout << "Error caught!" << endl;
	}
}