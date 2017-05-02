#ifndef ACF_FEATURE_EXTRACTOR_H
#define ACF_FEATURE_EXTRACTOR_H

#include "wwmatrix.hpp"
#include "rgbConvert.hpp"
#include "convConst.h"
#include "gradient.h"
#include "opencv2/opencv.hpp"
#include "boost/thread.hpp"

struct MatchModel
{
	WWMatrix<double> chnL;				//binlx1
	WWMatrix<double> chnU;				//binux1
	WWMatrix<double> chnV;				//binvx1
	std::vector<WWMatrix<double> > hog;	//(cellswxcellsh+1)x(binhogx1)
};

class AcfFeatureExtractor
{
public:
	//AcfFeatureExtractor():MINL(0.), MAXL(0.37), MINU(0.), MAXU(1.), MINV(0.), MAXV(0.89){}
	AcfFeatureExtractor(int binL=16, int binU=16, int binV=4, int binHOG=6, int csh=4, int csw=2, int ipn=5, int cpn=3):binl(binL),binu(binU),binv(binV),binhog(binHOG),cellsh(csh),cellsw(csw),init_probe_need(ipn),current_probe_need(cpn),
		//MINL(0.), MAXL(0.37), MINU(0.), MAXU(1.), MINV(0.), MAXV(0.89){}
		MINL(0.), MAXL(1.), MINU(0.), MAXU(1.), MINV(0.), MAXV(1.){}
	void feature_extract(cv::Mat& img, std::vector<cv::Mat>& feature);
	double calc_roi_scores(cv::Rect& roi, double lr);
	void update_match_model(cv::Rect& roi, int flag = 0);
	void feature_extract(cv::Mat& img, cv::Rect& roi, int binL, int binU, int binV, int binHOG, std::vector<double>& feature);
	void feature_extract(cv::Mat& img, cv::Rect& roi, int binL, int binU, int binV, int binHOG, std::vector<std::vector<double> >& feature);
	void feature_extract(cv::Mat& img, cv::Rect& roi, int binL, int binU, int binV, int binHOG, std::vector<cv::Mat>& feature);
	void feature_extract(cv::Mat& img, cv::Rect& roi, int binL, int binU, int binV, int binHOG, cv::Mat& feature);
	void feature_extract(cv::Mat& img);
	void feature_hist(std::vector<cv::Mat>& feature, cv::Rect& roi, char* bins, std::vector<double>& hist);
	void feature_hist(std::vector<cv::Mat>& feature, cv::Rect& roi, char* bins, std::vector<std::vector<double> >& hist);
private:
	int binl,			//number of bins of channel L
		binu,			//number of bins of channel U
		binv,			//number of bins of channel V
		binhog,			//number of orientations of HOG
		cellsh,			//number of cells along height
		cellsw,			//number of cells along width
		init_probe_need,//number of initial probes needed
		current_probe_need;		//number of near current probes needed
	WWMatrix<float> luv;		//hxwx3, h: image height, w: image width. Features of channels LUV
	cv::Mat orients;	//hxwxbinhog, h: image height, w: image width. integrogram of hog
	std::vector<MatchModel> probe;
	void rgbConvert(WWMatrix<uchar>& wwm, WWMatrix<float>& dst, int rxc, int chn, int flags = 2);
	void convTri(WWMatrix<float>& src, WWMatrix<float>& dst, const char* type, double r, double s);
	void addChn(WWMatrix<float>& data, std::vector<cv::Mat>& feature);
	void feature_hist(cv::Mat& fea, char bin, double minfea, double maxfea, std::vector<double>::iterator& b);
	void feature_hist(WWMatrix<float>& data, cv::Rect& roi, int ch, char bin, double minfea, double maxfea, std::vector<double>::iterator& b);
	void feature_hist(WWMatrix<float>& data, cv::Rect& roi, int ch, char bin, double minfea, double maxfea, cv::MatIterator_<double>& it);
	void feature_hist(WWMatrix<float>& data, cv::Rect& roi, int ch, char bin, double minfea, double maxfea, double* hist);
	void feature_hist(cv::Mat& integ_orient, cv::Rect& roi, double* hist);
	double distance(MatchModel& mm1, MatchModel& mm2);
	double bhattacharyya(double* d1, double* d2, int len, bool norm_flag = true);
	void calc_model(MatchModel& mm, cv::Rect& roi);
	double distance_from_probe(MatchModel& mm, double lr);
	double get_luv_cells_weight();
	const double MINL, MAXL, MINU, MAXU, MINV, MAXV;

	void print_freature(const MatchModel& mm);
};

#endif