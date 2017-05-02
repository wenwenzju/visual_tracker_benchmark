#include "particle_filter.h"

ParticleFilter::ParticleFilter(int sn, int pn) : states_num_(sn), particles_num_(pn), whether_resample(false)
{

}

void ParticleFilter::resample_(std::vector<vec_d>& xk, vec_d& wk, int strategy /* = SYSTEMATIC_RESAMPLE */)
{
	std::vector<vec_d> r_xk;
	switch(strategy)
	{
	case MULTINOMIAL_RESAMPLE:
	case SYSTEMATIC_RESAMPLE:
		vec_d edges;edges.push_back(0.);
		vec_d::iterator b = wk.begin(), e = wk.end();
		int i = 0;
		for (; b != e; ++b, ++i)
		{
			edges.push_back(edges[i] + *b);
		}
		edges[i] = 1;
		boost::mt19937 rng/*(time(0))*/;
		boost::uniform_01<> ud;
		double u1 = ud(rng)/particles_num_;
		double tmp = 1./particles_num_;
		int k = 0;		//k_th bin
		for (int j = 0; j < particles_num_;)
		{
			if (u1 >= edges[k] && u1 < edges[k+1])
			{
				r_xk.push_back(xk[k]);
				u1 += tmp;
				++j;
			}
			else
			{
				++k;
			}
		}
		break;
	}
	swap(xk, r_xk);
	swap(wk, vec_d(particles_num_, 1./particles_num_));
}

struct Parallel_Generate_Particles : public cv::ParallelLoopBody
{
	typedef std::vector<double> vec_d;
public:
	Parallel_Generate_Particles(boost::function<double (vec_d&)>* target_dis,
		boost::function<double (vec_d&, vec_d&)>* proposal_dis, boost::function<void (vec_d&, vec_d&)>* proposal_sam,
		std::vector<vec_d>* x, int* ite_num, vec_d* x0) : target_dis_(target_dis), proposal_dis_(proposal_dis), proposal_sam_(proposal_sam),
	x_(x), ite_num_(ite_num), x0_(x0){};
	void operator()(const cv::Range& range) const
	{
		boost::mt19937 rng(time(0));
		boost::uniform_01<> ud;
		vec_d x1 = *x0_, x2;
		for (int i = 0; i < *ite_num_; ++i)
		{
			double u = ud(rng);
			vec_d x_star;
			proposal_sam_->operator()(x1, x_star);
			double tmp = target_dis_->operator()(x_star)*proposal_dis_->operator()(x1, x_star)/
				target_dis_->operator()(x1)/proposal_dis_->operator()(x_star, x1);
			if (u < (1.<tmp?1.:tmp))
			{x2 = x_star;x1 = x2;}
			else
				x2 = x1;
		}
		x_->push_back(x2);
	}
private:

	boost::function<double (vec_d&)>* target_dis_;
	boost::function<double (vec_d&, vec_d&)>* proposal_dis_;
	boost::function<void (vec_d&, vec_d&)>* proposal_sam_;
	std::vector<vec_d>* x_;
	int* ite_num_;
	vec_d* x0_;
};

void ParticleFilter::mcmc(/*input*/boost::function<double (vec_d&)>& target_dis, 
	boost::function<double (vec_d&, vec_d&)>& proposal_dis, boost::function<void (vec_d&, vec_d&)>& proposal_sam, 
	/*output*/std::vector<vec_d>& x, /*input*/vec_d& x0, int ite_num/* = 2000*/, int n/* = 1*/)
{
	cv::parallel_for_(cv::Range(0, n), Parallel_Generate_Particles(&target_dis, &proposal_dis, &proposal_sam, &x, &ite_num, &x0));
}

void ParticleFilter::filter(vec_d& yk, vec_d& xk,std::vector<vec_d>& particles, int resample_strategy)
{
	std::vector<vec_d> xk_;
	vec_d wk_;
	double s = 0.;
	static bool is_first = true;
	if (is_first)
	{
		gen_x0(xkm1_, wkm1_);
		is_first = false;
	}
	sample_from_q(xkm1_, xk_);

	update_weight(wkm1_, xkm1_, xk_, yk, wk_);
	for (int i = 0; i < particles_num_; ++i) s += wk_[i];

	vec_d::iterator b = wk_.begin(), e = wk_.end();
	double ss = 0.;
	for (; b != e; ++b)
	{
		*b = *b/s;
		ss += (*b * *b);
	}
	
	double neff = 1./ss;
	if (neff < 0.8*particles_num_)
	{resample_(xk_, wk_, resample_strategy);whether_resample = true;}
	else whether_resample = false;
	//resample_(xk_, wk_, resample_strategy);

	xk.clear();
	/*weighted average*/
	for (int i = 0; i < states_num_; ++i)
	{
		double statei = 0.;
		for (int j = 0; j < particles_num_; ++j)
			statei += wk_[j]*xk_[j][i];
		xk.push_back(statei);
	}

	/*max*/
	//double max_weight = 0.; int idx = 0;
	//for (int i = 0; i < particles_num_; ++i)
	//{
	//	if (wk_[i]>max_weight) {max_weight = wk_[i];idx = i;}
	//}
	//xk = xk_[idx];

	/*nms*/
	//bbNms(xk_, wk_, xk, 0.8);

	/*median*/
	//sort_particles(xk_, wk_, sorted_particles);
	//if (particles_num_%2)
	//{
	//	for (int i = 0; i < states_num_; ++i)
	//		xk.push_back((sorted_particles[particles_num_/2][i]+sorted_particles[particles_num_/2-1][i])/2);
	//}
	//else
	//{
	//	for (int i = 0; i < states_num_; ++i)
	//		xk.push_back(sorted_particles[particles_num_/2][i]);
	//}

		//double statei = -FLT_MAX;
		//int idx = -1;
		//for (int j = 0; j < particles_num_; ++j)
		//	statei < wk_[j] ? (statei = wk_[j], idx = j):idx = idx;
		//xk = xk_[idx];

	/*first n average*/
	//std::vector<vec_d> sorted_particles;
	//sort_particles(xk_, wk_, sorted_particles);
	//int firstn = 20, l = sorted_particles[0].size();double sw = 0;
	//std::for_each(sorted_particles.begin(), sorted_particles.begin()+firstn,[&sw,l](vec_d& v){sw+=v[l-1];});
	//std::for_each(sorted_particles.begin(), sorted_particles.begin()+firstn,[sw,l](vec_d& v){v[l-1]/=sw;});
	//xk.clear();
	//for (int i = 0; i < states_num_; ++i)
	//{
	//	double statei = 0.;
	//	for (int j = 0; j < firstn; ++j)
	//		statei += sorted_particles[j][l-1]*sorted_particles[j][i];
	//	xk.push_back(statei);
	//}

	//n-sigma
	//calc_average(xk_, wk_, xk);

	particles = xk_;
	swap(wk_, wkm1_);
	swap(xk_, xkm1_);
}

void ParticleFilter::save_weights(std::ofstream& ofs)
{
	for (int i = 0; i < particles_num_-1; ++i)
		ofs << wkm1_[i] << "	";
	ofs << wkm1_[particles_num_-1] << std::endl;
}

//void ParticleFilter::bbNms(std::vector<vec_d>& bbs, vec_d& w, vec_d& res, double overlap /* = 0.65 */)
//{
//#define PARTICLES_KEEP 30
//	using namespace std;
//	if (bbs.empty()) return;
//	vector<vec_d> bbs_(bbs.size(), vec_d(5,0));
//	for (int i = 0; i < bbs.size(); ++i)
//	{
//		bbs_[i][0] = bbs[i][0];
//		bbs_[i][1] = bbs[i][1];
//		bbs_[i][2] = bbs[i][2];
//		bbs_[i][3] = 2*bbs[i][2];
//		bbs_[i][4] = w[i];
//	}
//	sort(bbs_.begin(), bbs_.end(), [](vec_d& a, vec_d& b)->bool{return a[4]>b[4];});
//
//	int n = bbs_.size();
//	std::vector<bool> kp(n,true);
//	for (int i = 0; i < n; i++)
//	{
//		if (!kp[i]) continue;
//		double xei = bbs_[i][0]+bbs_[i][2];
//		double yei = bbs_[i][1]+bbs_[i][3];
//		double xsi = bbs_[i][0], ysi = bbs_[i][1];
//		double asi = bbs_[i][2]*bbs_[i][3];
//		for (int j = i+1; j < n; j++)
//		{
//			if (!kp[i]) continue;
//			double xej = bbs_[j][0]+bbs_[j][2];
//			double yej = bbs_[j][1]+bbs_[j][3];
//			double xsj = bbs_[j][0], ysj = bbs_[j][1];
//			double asj = bbs_[j][2]*bbs_[j][3];
//			double iw = min(xei,xej)-max(xsi,xsj);
//			if (iw <= 0) continue;
//			double ih = min(yei, yej)-max(ysi, ysj);
//			if (ih <= 0) continue;
//			if ((iw*ih) / min(asi, asj) > overlap) {kp[j] = false;bbs_[i][4] += bbs_[j][4];}
//		}
//	}
//
//	res.clear();
//	res = vec_d(bbs[0].size(), 0);
//	double sw = 0.;
//	int cnt = 0;
//	for (int i = 0; i < n; ++i)
//	{
//		if (kp[i]) {sw += bbs_[i][4];cnt ++;}
//	}
//	cout << "Total rectangles after nms: " << cnt << endl;
//	for (int i = 0; i < n; i++)
//	{
//		if (kp[i])
//		{
//			res[0] += bbs_[i][0]*bbs_[i][4]/sw;
//			res[1] += bbs_[i][1]*bbs_[i][4]/sw;
//			res[2] += bbs_[i][2]*bbs_[i][4]/sw;
//			//res[0] += bbs_[i][0]/cnt;
//			//res[1] += bbs_[i][1]/cnt;
//			//res[2] += bbs_[i][2]/cnt;
//		}
//	}
//}
void ParticleFilter::bbNms(std::vector<vec_d>& bbs, vec_d& w, vec_d& res, double overlap /* = 0.65 */)
{
#define PARTICLES_KEEP 60
	using namespace std;
	if (bbs.empty()) return;

	sort_particles(bbs, w, sorted_particles);
	int ss = sorted_particles[0].size();
	int n = sorted_particles.size();

	for (int i = PARTICLES_KEEP; i < n; i++)
	{
		double xei = sorted_particles[i][0]+sorted_particles[i][2];
		double yei = sorted_particles[i][1]+sorted_particles[i][2]*2;
		double xsi = sorted_particles[i][0], ysi = sorted_particles[i][1];
		double asi = sorted_particles[i][2]*sorted_particles[i][2]*2;
		for (int j = 0; j < PARTICLES_KEEP; j++)
		{
			//if (!kp[i]) continue;
			double xej = sorted_particles[j][0]+sorted_particles[j][2];
			double yej = sorted_particles[j][1]+sorted_particles[j][2]*2;
			double xsj = sorted_particles[j][0], ysj = sorted_particles[j][1];
			double asj = sorted_particles[j][2]*sorted_particles[j][2]*2;
			double iw = min(xei,xej)-max(xsi,xsj);
			if (iw <= 0) continue;
			double ih = min(yei, yej)-max(ysi, ysj);
			if (ih <= 0) continue;
			if ((iw*ih) / min(asi, asj) > overlap) {sorted_particles[j][ss-1] += sorted_particles[i][ss-1];/*continue;*/}
		}
	}

	res.clear();
	res = vec_d(bbs[0].size(), 0);
	double sw = 0.;
	for (int i = 0; i < PARTICLES_KEEP; ++i)
	{
		sw += sorted_particles[i][ss-1];
	}
	for (int i = 0; i < PARTICLES_KEEP; i++)
	{
		res[0] += sorted_particles[i][0]*sorted_particles[i][ss-1]/sw;
		res[1] += sorted_particles[i][1]*sorted_particles[i][ss-1]/sw;
		res[2] += sorted_particles[i][2]*sorted_particles[i][ss-1]/sw;
		//res[0] += sorted_particles[i][0]/cnt;
		//res[1] += sorted_particles[i][1]/cnt;
		//res[2] += sorted_particles[i][2]/cnt;
	}
}

void ParticleFilter::sort_particles(std::vector<vec_d>& p, vec_d& w, std::vector<vec_d>& sp)
{
	using namespace std;
	if (p.empty()) return;
	int ps = p.size(), ss = p[0].size();
	sp = p;

	for (int i = 0; i < ps; ++i)
	{
		sp[i].push_back(w[i]);
	}

	sort(sp.begin(), sp.end(), [ss](vec_d& a, vec_d& b)->bool{return a[ss]>b[ss];});
}

void ParticleFilter::calc_average(std::vector<vec_d>& p, vec_d& w, vec_d& avr)
{
#define NSIGMA 1
	int n = p.size(), nn = p[0].size();
	double s = 0, ss = 0;
	std::for_each(w.begin(), w.end(),[&s,&ss](double d){s+=d;ss+=d*d;});
	double m = s/n, sig = ss/n-m*m;
	sig = std::sqrt(sig);
	double low = m-NSIGMA*sig, up = m+NSIGMA*sig;
	double sw = 0;
	std::vector<int> rmv_idx(n,0);
	avr = vec_d(nn,0);
	for (int i = 0; i < n; ++i)
	{
		if (w[i]<low||w[i]>up) rmv_idx[i] = 1;
		else sw+=w[i];
	}
	for (int i = 0; i < n; ++i)
	{
		if (rmv_idx[i]) continue;
		double wei = w[i]/sw;
		for (int j = 0; j < nn; ++j)
			avr[j] += wei*p[i][j];
	}

	std::cout << "Particles remove: ";
	for (int i = 0; i < n; ++i) if (rmv_idx[i]) std::cout << i << " ";
	if (whether_resample) std::cout << "Resample" ;
	std::cout << std::endl;
}