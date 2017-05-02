#include "acf_feature_extractor.h"

#define SHRINK 1

#define START_THREAD_LOCK {boost::mutex::scoped_lock scop(boost::mutex());
#define END_THREAD_LOCK }

void AcfFeatureExtractor::feature_extract(cv::Mat& img)
{
	CV_Assert(img.channels() == 3);
	cv::Mat rgb_img;
	img.copyTo(rgb_img);

	{//convert bgr to rgb
		std::vector<cv::Mat> m1, m2;
		cv::split(rgb_img, m1);
		m2.push_back(m1[2]); m2.push_back(m1[1]); m2.push_back(m1[0]);
		cv::merge(m2, rgb_img);
	}

	WWMatrix<uchar> wwimg(rgb_img.rows, rgb_img.cols, rgb_img.channels());
	wwimg.copyfromMat(rgb_img);
	WWMatrix<float> wwdata;
	rgbConvert(wwimg, wwdata, wwimg.rows*wwimg.cols, wwimg.channels, 2);	//convert rgb to luv

	//compute color channels
	{
		int smooth = 1;
		convTri(wwdata, luv, "convTri1", 1.*12./smooth/(smooth+2)-2,1.);
	}

	//compute gradient magnitude channel
	WWMatrix<float> M(img.rows, img.cols), O(img.rows, img.cols);
	{
		int normRad = 5;
		double normConst = 0.005;
		gradMag(luv.data.get(), M.data.get(), O.data.get(), img.rows, img.cols, img.channels(), 0);
		WWMatrix<float> S;
		convTri(M, S, "convTri", normRad, 1);

		gradMagNorm(M.data.get(), S.data.get(), img.rows, img.cols, normConst);
	}
	//compute gradient histogram channels
	{
		int binSize = 1;
		int nOrients = binhog;
		int softBin = 0;
		WWMatrix<float> H(img.rows/binSize, img.cols/binSize, nOrients);
		gradHist(M.data.get(), O.data.get(), H.data.get(), img.rows, img.cols, binSize, nOrients, softBin, 0);

		orients = std::vector<cv::Mat>(nOrients);
		for (int i = 0; i < nOrients; ++i)
		{
			cv::Mat mH; H.copytoMat(mH, i);
			cv::integral(mH, orients[i], CV_64F);
		}
	}
}

double AcfFeatureExtractor::calc_roi_scores(cv::Rect& roi, double lr)
{
	MatchModel mmroi;
	calc_model(mmroi, roi);
	//print_freature(mmroi);
	//system("pause");

	double res = distance_from_probe(mmroi, lr);
	return res;
}

void AcfFeatureExtractor::print_freature(const MatchModel& mm)
{
	{
		boost::mutex::scoped_lock scop(boost::mutex());
		using namespace std;
		cout << "chnL: [";
		for (int i = 0; i <mm.chnL.rows*mm.chnL.cols*mm.chnL.channels; ++i) cout << mm.chnL.data[i] << " ";
		cout << "]" << endl;

		cout << "chnU: [";
		for (int i = 0; i <mm.chnU.rows*mm.chnU.cols*mm.chnU.channels; ++i) cout << mm.chnU.data[i] << " ";
		cout << "]" << endl;

		cout << "chnV: [";
		for (int i = 0; i <mm.chnV.rows*mm.chnV.cols*mm.chnV.channels; ++i) cout << mm.chnV.data[i] << " ";
		cout << "]" << endl;

		for (int i = 0; i < mm.hog.size(); ++i)
		{
			cout << "cell " << i << ": [";
			for (int j = 0; j <mm.hog[i].rows*mm.hog[i].cols*mm.hog[i].channels; ++j) cout << mm.hog[i].data[j] << " ";
			cout << "]" << endl;
		}

		cout << "features: [";
		for (int i = 0; i < mm.features.rows*mm.features.cols*mm.features.channels; ++i) cout << mm.features.data[i] << " ";
		cout << "]" << endl;
	}
}

double AcfFeatureExtractor::distance(MatchModel& mm1, MatchModel& mm2)
{
	return bhattacharyya(mm1.features.data.get(), mm2.features.data.get(), mm1.features.rows*mm1.features.cols*mm1.features.channels);
}

double AcfFeatureExtractor::get_luv_cells_weight()
{
	return 0.;
}

#define SUM(d,l,s,f) for (int i = 0; i < l; ++i){*(s) += (f)(*((d)+i));} 
double AcfFeatureExtractor::bhattacharyya(double* d1, double* d2, int len, bool norm_flag /* = true */)
{
	double s1 = 0., s2 = 0.;
	if (norm_flag)
	{
		SUM(d1, len, &s1, [](double d)->double{return d;});
		SUM(d2, len, &s2, [](double d)->double{return d;});
	}
	else
	{
		s1 = 1.; s2 = 1.;
	}
	double ss = 0.;
	for (int i = 0; i < len; ++i)
	{
		double tmp = (d1[i]/s1)*(d2[i]/s2);
		if (tmp < 0) continue;
		ss += std::sqrt(tmp);
	}
	ss = (ss>1?1:ss);
	return std::sqrt(1-ss)+0.00001;
}

#define INIT_PROBE_STEP 3
void AcfFeatureExtractor::update_match_model(cv::Rect& roi, int flag /* = 0 */)
{
	static int cnt = 0;
	MatchModel mmroi;
	calc_model(mmroi, roi);
	if (probe.size() < init_probe_need) 
	{
		if (cnt%INIT_PROBE_STEP == 0) 
		{
			probe.push_back(mmroi);
			//std::cout << "update initial probe..." << std::endl;
		}
		cnt++;
	}
	else if (probe.size() < init_probe_need+current_probe_need) 
	{
		probe.push_back(mmroi);
		//std::cout << "update current probe..." << std::endl;
	}
	else if (current_probe_need)
	{
		probe.erase(probe.begin()+init_probe_need);probe.push_back(mmroi);
		//std::cout << "erase and update current probe..." << std::endl;
	}
}

void AcfFeatureExtractor::calc_model(MatchModel& mmroi, cv::Rect& roi)
{
	mmroi.features.creat((binl+binu+binv+binhog)*(cellsh*cellsw));
	int totbins = binl+binu+binv+binhog;
	int totcells = cellsh*cellsw;
	int cell_width = roi.width/cellsw;
	int cell_height = roi.height/cellsh;
	int cell_cnt = 0;
	for (int i = 0; i < cellsw; ++i)
	{
		int cw = cell_width;
		if (i == cellsw-1) cw = roi.width - cell_width*i;		//boundary
		for (int j = 0; j < cellsh; ++j)
		{
			int ch = cell_height;
			if (j == cellsh-1) ch = roi.height - cell_height*j;	//boundary
			cv::Rect r(roi.x+i*cell_width, roi.y+j*cell_height, cw, ch);

			//int offset = cell_cnt*totbins;
			//feature_hist(luv, r, 0, binl, MINL, MAXL, mmroi.features.data.get()+offset);
			//normalize(mmroi.features.data.get()+offset, binl);

			//offset += binl;
			//feature_hist(luv, r, 1, binu, MINU, MAXU, mmroi.features.data.get()+offset);
			//normalize(mmroi.features.data.get()+offset, binu);

			//offset += binu;
			//feature_hist(luv, r, 2, binv, MINV, MAXV, mmroi.features.data.get()+offset);
			//normalize(mmroi.features.data.get()+offset, binv);

			//offset += binv;
			//feature_hist(orients, r, mmroi.features.data.get()+offset);
			//normalize(mmroi.features.data.get()+offset, binhog);
			//cell_cnt++;

			int offset = cell_cnt*binl;
			feature_hist(luv, r, 0, binl, MINL, MAXL, mmroi.features.data.get()+offset);

			offset = totcells*binl + cell_cnt*binu;
			feature_hist(luv, r, 1, binu, MINU, MAXU, mmroi.features.data.get()+offset);

			offset = totcells*binl + totcells*binu + cell_cnt*binv;
			feature_hist(luv, r, 2, binv, MINV, MAXV, mmroi.features.data.get()+offset);

			offset = totcells*binl + totcells*binu + totcells*binv + cell_cnt*binhog;
			feature_hist(orients, r, mmroi.features.data.get()+offset);
			cell_cnt++;
		}
	}
	normalize(mmroi.features.data.get(), totcells*binl);
	normalize(mmroi.features.data.get()+totcells*binl, totcells*binu);
	normalize(mmroi.features.data.get()+totcells*binl+totcells*binu, totcells*binv);
	normalize(mmroi.features.data.get()+totcells*binl+totcells*binu+totcells*binv, totcells*binhog);
}

double AcfFeatureExtractor::distance_from_probe(MatchModel& mm, double lr)
{
	if (probe.size() <= init_probe_need)
	{
		double mind1 = DBL_MAX;
		if (!probe.size()) system("pause");
		for (int i = 0; i < probe.size(); ++i)
		{
			double d = distance(mm, probe[i]);
			if (mind1 > d) mind1 = d;
		}
		return mind1;
	}
	else
	{
		double mind1 = DBL_MAX, mind2 = DBL_MAX;
		for (int i = 0; i < init_probe_need; ++i)
		{
			double d = distance(mm, probe[i]);
			if (mind1 > d) mind1 = d;
		}
		for (int i = init_probe_need; i < probe.size(); ++i)
		{
			double d = distance(mm, probe[i]);
			if (mind2 > d) mind2 = d;
		}
		return (1.-lr)*mind1 + lr*mind2;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////
void AcfFeatureExtractor::feature_extract(cv::Mat& img, std::vector<cv::Mat>& feature)
{
	CV_Assert(img.channels() == 3);
	feature.clear();
	cv::Mat rgb_img;
	img.copyTo(rgb_img);

	{//convert bgr to rgb
		std::vector<cv::Mat> m1, m2;
		cv::split(rgb_img, m1);
		m2.push_back(m1[2]); m2.push_back(m1[1]); m2.push_back(m1[0]);
		cv::merge(m2, rgb_img);
	}

	WWMatrix<uchar> wwimg(rgb_img.rows, rgb_img.cols, rgb_img.channels());
	wwimg.copyfromMat(rgb_img);
	WWMatrix<float> wwdata;
	rgbConvert(wwimg, wwdata, wwimg.rows*wwimg.cols, wwimg.channels);	//convert rgb to luv

	WWMatrix<float> convdata;
	//compute color channels
	{
		int smooth = 1;
		convTri(wwdata, convdata, "convTri1", 1.*12./smooth/(smooth+2)-2,1.);
		addChn(convdata, feature);
	}

	//compute gradient magnitude channel
	WWMatrix<float> M(img.rows, img.cols), O(img.rows, img.cols);
	{
		int normRad = 5;
		double normConst = 0.005;
		gradMag(convdata.data.get(), M.data.get(), O.data.get(), img.rows, img.cols, img.channels(), 0);
		WWMatrix<float> S;
		convTri(M, S, "convTri", normRad, 1);

		gradMagNorm(M.data.get(), S.data.get(), img.rows, img.cols, normConst);
		addChn(M, feature);
	}
	//compute gradient histogram channels
	{
		int binSize = SHRINK;
		int nOrients = 6;
		int softBin = 0;
		WWMatrix<float> H(img.rows/binSize, img.cols/binSize, nOrients);
		gradHist(M.data.get(), O.data.get(), H.data.get(), img.rows, img.cols, binSize, nOrients, softBin, 0);

		addChn(H, feature);
	}
}

void AcfFeatureExtractor::feature_extract(cv::Mat& img, cv::Rect& roi, int binL, int binU, int binV, int binHOG, std::vector<double>& feature)
{
	CV_Assert(img.channels() == 3);
	feature.clear();
	int nOrients = 6;
	cv::Mat rgb_img;
	int h = img.rows, w = img.cols;
	int crh = h%binHOG, crw = w%binHOG;
	h-=crh;w-=crw;
	feature = std::vector<double>(binL+binU+binV+h/binHOG*w/binHOG*nOrients, 0);

	img(cv::Rect(0,0,w,h)).copyTo(rgb_img);
	cv::Rect r = roi&cv::Rect(0,0,w,h);

	{//convert bgr to rgb
		std::vector<cv::Mat> m1, m2;
		cv::split(rgb_img, m1);
		m2.push_back(m1[2]); m2.push_back(m1[1]); m2.push_back(m1[0]);
		cv::merge(m2, rgb_img);
	}

	WWMatrix<uchar> wwimg(rgb_img.rows, rgb_img.cols, rgb_img.channels());
	wwimg.copyfromMat(rgb_img);
	WWMatrix<float> wwdata;
	rgbConvert(wwimg, wwdata, wwimg.rows*wwimg.cols, wwimg.channels);	//convert rgb to luv

	WWMatrix<float> convdata;
	//compute color channels
	{
		int smooth = 1;
		convTri(wwdata, convdata, "convTri1", 1.*12./smooth/(smooth+2)-2,1.);
		feature_hist(convdata, r, 0, binL, MINL, MAXL, feature.begin());
		feature_hist(convdata, r, 1, binU, MINU, MAXU, feature.begin()+binL);
		feature_hist(convdata, r, 2, binV, MINV, MAXV, feature.begin()+binL+binU);
	}

	//compute gradient magnitude channel
	WWMatrix<float> M(img.rows, img.cols), O(img.rows, img.cols);
	{
		int normRad = 5;
		double normConst = 0.005;
		gradMag(convdata.data.get(), M.data.get(), O.data.get(), img.rows, img.cols, img.channels(), 0);
		WWMatrix<float> S;
		convTri(M, S, "convTri", normRad, 1);

		gradMagNorm(M.data.get(), S.data.get(), img.rows, img.cols, normConst);
	}
	//compute gradient histogram channels
	{
		int binSize = binHOG;
		int softBin = 0;
		WWMatrix<float> H(h/binSize, w/binSize, nOrients);
		gradHist(M.data.get(), O.data.get(), H.data.get(), img.rows, img.cols, binSize, nOrients, softBin, 0);

		std::vector<double>::iterator it = feature.begin()+binL+binU+binV, it_end = feature.end();
		float* data = H.data.get();
		int i = 0;
		for (; it!=it_end; ++it, ++i) *it = data[i];
	}
}

void AcfFeatureExtractor::feature_extract(cv::Mat& img, cv::Rect& roi, int binL, int binU, int binV, int binHOG, std::vector<std::vector<double> >& feature)
{
	CV_Assert(img.channels() == 3);
	feature.clear();
	int nOrients = 6;
	cv::Mat rgb_img;
	int h = img.rows, w = img.cols;
	int crh = h%binHOG, crw = w%binHOG;
	h-=crh;w-=crw;
	feature.push_back(std::vector<double>(binL, 0));
	feature.push_back(std::vector<double>(binU, 0));
	feature.push_back(std::vector<double>(binV, 0));
	for (int i = 0; i < nOrients; ++i)
		feature.push_back(std::vector<double>(h/binHOG*w/binHOG, 0));

	img(cv::Rect(0,0,w,h)).copyTo(rgb_img);
	cv::Rect r = roi&cv::Rect(0,0,w,h);

	{//convert bgr to rgb
		std::vector<cv::Mat> m1, m2;
		cv::split(rgb_img, m1);
		m2.push_back(m1[2]); m2.push_back(m1[1]); m2.push_back(m1[0]);
		cv::merge(m2, rgb_img);
	}

	WWMatrix<uchar> wwimg(rgb_img.rows, rgb_img.cols, rgb_img.channels());
	wwimg.copyfromMat(rgb_img);
	WWMatrix<float> wwdata;
	rgbConvert(wwimg, wwdata, wwimg.rows*wwimg.cols, wwimg.channels);	//convert rgb to luv

	WWMatrix<float> convdata;
	//compute color channels
	{
		int smooth = 1;
		convTri(wwdata, convdata, "convTri1", 1.*12./smooth/(smooth+2)-2,1.);
		feature_hist(convdata, r, 0, binL, MINL, MAXL, feature[0].begin());
		feature_hist(convdata, r, 1, binU, MINU, MAXU, feature[1].begin());
		feature_hist(convdata, r, 2, binV, MINV, MAXV, feature[2].begin());
	}

	//compute gradient magnitude channel
	WWMatrix<float> M(img.rows, img.cols), O(img.rows, img.cols);
	{
		int normRad = 5;
		double normConst = 0.005;
		gradMag(convdata.data.get(), M.data.get(), O.data.get(), img.rows, img.cols, img.channels(), 0);
		WWMatrix<float> S;
		convTri(M, S, "convTri", normRad, 1);

		gradMagNorm(M.data.get(), S.data.get(), img.rows, img.cols, normConst);
	}
	//compute gradient histogram channels
	{
		int binSize = binHOG;
		int softBin = 0;
		WWMatrix<float> H(h/binSize, w/binSize, nOrients);
		gradHist(M.data.get(), O.data.get(), H.data.get(), img.rows, img.cols, binSize, nOrients, softBin, 0);

		float* data = H.data.get();
		int i = 0;
		for (int j = 0; j < nOrients; ++j)
		{
			std::vector<double>::iterator it = feature[j+3].begin(), it_end = feature[j+3].end();
			for (; it!=it_end; ++it, ++i) *it = data[i];
		}
	}
}

void AcfFeatureExtractor::feature_extract(cv::Mat& img, cv::Rect& roi, int binL, int binU, int binV, int binHOG, cv::Mat& feature)
{
	CV_Assert(img.channels() == 3);
	int nOrients = 6;
	cv::Mat rgb_img;
	int h = img.rows, w = img.cols;
	int crh = h%binHOG, crw = w%binHOG;
	h-=crh;w-=crw;
	feature = cv::Mat::zeros(binL+binU+binV+h/binHOG*w/binHOG*nOrients,1,CV_64FC1);

	img(cv::Rect(0,0,w,h)).copyTo(rgb_img);
	cv::Rect r = roi&cv::Rect(0,0,w,h);

	{//convert bgr to rgb
		std::vector<cv::Mat> m1, m2;
		cv::split(rgb_img, m1);
		m2.push_back(m1[2]); m2.push_back(m1[1]); m2.push_back(m1[0]);
		cv::merge(m2, rgb_img);
	}

	WWMatrix<uchar> wwimg(rgb_img.rows, rgb_img.cols, rgb_img.channels());
	wwimg.copyfromMat(rgb_img);
	WWMatrix<float> wwdata;
	rgbConvert(wwimg, wwdata, wwimg.rows*wwimg.cols, wwimg.channels);	//convert rgb to luv

	WWMatrix<float> convdata;
	//compute color channels
	{
		int smooth = 1;
		convTri(wwdata, convdata, "convTri1", 1.*12./smooth/(smooth+2)-2,1.);
		feature_hist(convdata, r, 0, binL, MINL, MAXL, feature.begin<double>());
		feature_hist(convdata, r, 1, binU, MINU, MAXU, feature.begin<double>()+binL);
		feature_hist(convdata, r, 2, binV, MINV, MAXV, feature.begin<double>()+binL+binU);
	}

	//compute gradient magnitude channel
	WWMatrix<float> M(img.rows, img.cols), O(img.rows, img.cols);
	{
		int normRad = 5;
		double normConst = 0.005;
		gradMag(convdata.data.get(), M.data.get(), O.data.get(), img.rows, img.cols, img.channels(), 0);
		WWMatrix<float> S;
		convTri(M, S, "convTri", normRad, 1);

		gradMagNorm(M.data.get(), S.data.get(), img.rows, img.cols, normConst);
	}
	//compute gradient histogram channels
	{
		int binSize = binHOG;
		int softBin = 0;
		WWMatrix<float> H(h/binSize, w/binSize, nOrients);
		gradHist(M.data.get(), O.data.get(), H.data.get(), img.rows, img.cols, binSize, nOrients, softBin, 0);

		cv::MatIterator_<double> it = feature.begin<double>()+binL+binU+binV, it_end = feature.end<double>();
		float* data = H.data.get();
		int i = 0;
		for (; it!=it_end; ++it, ++i) *it = data[i];
	}
}

void AcfFeatureExtractor::feature_extract(cv::Mat& img, cv::Rect& roi, int binL, int binU, int binV, int binHOG, std::vector<cv::Mat>& feature)
{
	CV_Assert(img.channels() == 3);
	feature.clear();
	int nOrients = 6;
	cv::Mat rgb_img;
	int h = img.rows, w = img.cols;
	int crh = h%binHOG, crw = w%binHOG;
	h-=crh;w-=crw;
	feature.push_back(cv::Mat::zeros(binL,1,CV_64FC1));
	feature.push_back(cv::Mat::zeros(binU,1,CV_64FC1));
	feature.push_back(cv::Mat::zeros(binV,1,CV_64FC1));
	for (int i = 0; i < nOrients; ++i)
		feature.push_back(cv::Mat::zeros(h/binHOG*w/binHOG,1,CV_64FC1));

	img(cv::Rect(0,0,w,h)).copyTo(rgb_img);
	cv::Rect r = roi&cv::Rect(0,0,w,h);

	{//convert bgr to rgb
		std::vector<cv::Mat> m1, m2;
		cv::split(rgb_img, m1);
		m2.push_back(m1[2]); m2.push_back(m1[1]); m2.push_back(m1[0]);
		cv::merge(m2, rgb_img);
	}

	WWMatrix<uchar> wwimg(rgb_img.rows, rgb_img.cols, rgb_img.channels());
	wwimg.copyfromMat(rgb_img);
	WWMatrix<float> wwdata;
	rgbConvert(wwimg, wwdata, wwimg.rows*wwimg.cols, wwimg.channels);	//convert rgb to luv

	WWMatrix<float> convdata;
	//compute color channels
	{
		int smooth = 1;
		convTri(wwdata, convdata, "convTri1", 1.*12./smooth/(smooth+2)-2,1.);
		feature_hist(convdata, r, 0, binL, MINL, MAXL, feature[0].begin<double>());
		feature_hist(convdata, r, 1, binU, MINU, MAXU, feature[1].begin<double>());
		feature_hist(convdata, r, 2, binV, MINV, MAXV, feature[2].begin<double>());
	}

	//compute gradient magnitude channel
	WWMatrix<float> M(img.rows, img.cols), O(img.rows, img.cols);
	{
		int normRad = 5;
		double normConst = 0.005;
		gradMag(convdata.data.get(), M.data.get(), O.data.get(), img.rows, img.cols, img.channels(), 0);
		WWMatrix<float> S;
		convTri(M, S, "convTri", normRad, 1);

		gradMagNorm(M.data.get(), S.data.get(), img.rows, img.cols, normConst);
	}
	//compute gradient histogram channels
	{
		int binSize = binHOG;
		int softBin = 0;
		WWMatrix<float> H(h/binSize, w/binSize, nOrients);
		gradHist(M.data.get(), O.data.get(), H.data.get(), img.rows, img.cols, binSize, nOrients, softBin, 0);

		float* data = H.data.get();
		int i = 0;
		for (int j = 0; j < nOrients; ++j)
		{
			cv::MatIterator_<double> it = feature[j+3].begin<double>(), it_end = feature[j+3].end<double>();
			for (; it!=it_end; ++it, ++i) *it = data[i];
		}
	}
}

void AcfFeatureExtractor::feature_hist(std::vector<cv::Mat>& feature, cv::Rect& roi, char* bins, std::vector<std::vector<double> >& hist)
{

}

void AcfFeatureExtractor::feature_hist(std::vector<cv::Mat>& feature, cv::Rect& roi, char* bins, std::vector<double>& hist)
{

}

void AcfFeatureExtractor::rgbConvert(WWMatrix<uchar>& wwm, WWMatrix<float>& dst, int rxc, int chn, int flags /* = 2 */)
{
	dst = WWMatrix<float>::WWMatrix(wwm.rows, wwm.cols, wwm.channels);
	rgbconvert(wwm.data.get(), dst.data.get(), rxc, chn, flags, 1.0f/255);
}

void AcfFeatureExtractor::convTri(WWMatrix<float>& src, WWMatrix<float>& dst, const char* type, double r, double s)
{
	assert(s>=1);assert(r>=0);
	int ms0 = src.rows/(int)s, ms1 = src.cols/(int)s, ms2 = src.channels;
	dst = WWMatrix<float>::WWMatrix(ms0, ms1, ms2);

	// perform appropriate type of convolution
	if (!strcmp(type, "convBox"))
	{
		assert(r < std::min(src.rows, src.cols)/2);
		convBox(src.data.get(), dst.data.get(), src.rows, src.cols, src.channels, (int)r, (int)s);
	}
	else if (!strcmp(type, "convTri"))
	{
		assert(r < std::min(src.rows, src.cols)/2);
		convtri(src.data.get(), dst.data.get(), src.rows, src.cols, src.channels, (int)r, (int)s);
	}
	else if (!strcmp(type, "conv11"))
	{
		assert(s<=2);
		conv11(src.data.get(), dst.data.get(), src.rows, src.cols, src.channels, (int)r, (int)s);
	}
	else if (!strcmp(type, "convTri1"))
	{
		assert(s<=2);
		convTri1(src.data.get(), dst.data.get(), src.rows, src.cols, src.channels, r, (int)s);
	}
	else if (!strcmp(type, "convMax"))
	{
		assert(s<=1);
		convMax(src.data.get(), dst.data.get(), src.rows, src.cols, src.channels, (int)r);
	}
	else
		assert(false);
}

void AcfFeatureExtractor::addChn(WWMatrix<float>& data, std::vector<cv::Mat>& feature)
{
	for (int i = 0; i < data.channels; ++i)
	{
		cv::Mat tmp;
		data.copytoMat(tmp, i);
		feature.push_back(tmp);
	}
}

void AcfFeatureExtractor::feature_hist(cv::Mat& fea, char bin, double minfea, double maxfea, std::vector<double>::iterator& b)
{
	CV_Assert(fea.type() == CV_32FC1);

	cv::MatConstIterator_<float> it=fea.begin<float>(), it_end = fea.end<float>();
	for (; it != it_end; ++it)
	{
		int idx = (*it-minfea)*bin/(maxfea-minfea);
		*(b+idx) += 1.;
	}
}

void AcfFeatureExtractor::feature_hist(WWMatrix<float>& data, cv::Rect& roi, int ch, char bin, double minfea, double maxfea, std::vector<double>::iterator& b)
{
	double delta = maxfea-minfea;
	float* fea = data.data.get();
	int h = data.rows, w = data.cols;

	for (int x = roi.x; x < roi.x+roi.width; ++x)
		for (int y = roi.y; y < roi.y+roi.height; ++y)
		{
			int idx = (fea[h*x+y+ch*h*w])*bin/delta;
			*(b+idx) += 1.;
		}
}

void AcfFeatureExtractor::feature_hist(WWMatrix<float>& data, cv::Rect& roi, int ch, char bin, double minfea, double maxfea, cv::MatIterator_<double>& it)
{
	double delta = maxfea-minfea;
	float* fea = data.data.get();
	int h = data.rows, w = data.cols;

	for (int x = roi.x; x < roi.x+roi.width; ++x)
		for (int y = roi.y; y < roi.y+roi.height; ++y)
		{
			int idx = (fea[h*x+y+ch*h*w])*bin/delta;
			*(it+idx) += 1.;
		}
}

void AcfFeatureExtractor::feature_hist(WWMatrix<float>& data, cv::Rect& roi, int ch, char bin, double minfea, double maxfea, double* hist)
{
	double delta = maxfea-minfea;
	float* fea = data.data.get();
	int h = data.rows, w = data.cols;

	for (int x = roi.x; x < roi.x+roi.width; ++x)
		for (int y = roi.y; y < roi.y+roi.height; ++y)
		{
			int idx = (fea[h*x+y+ch*h*w])*bin/delta;
			*(hist+idx) += 1.;
		}
}

void AcfFeatureExtractor::feature_hist(cv::Mat& integ_orient, cv::Rect& roi, double* hist)
{
	CV_Assert(integ_orient.isContinuous());

	//std::cout << "x = " << roi.x << "; y = " << roi.y << "; width = " << roi.width << "; height = " << roi.height << std::endl;
	int chns = integ_orient.channels();
	const double* toprow = integ_orient.ptr<double>(roi.y);
	const double* botrow = integ_orient.ptr<double>(roi.y+roi.height);
	int leftcol_offset = (roi.x)*chns;
	int rightcol_offset = (roi.x+roi.width)*chns;
	for (int i = 0; i < chns; ++i)
	{
		hist[i] = *(botrow+rightcol_offset+i)+*(toprow+leftcol_offset+i)-*(toprow+rightcol_offset+i)-*(botrow+leftcol_offset+i);
	}
}

void AcfFeatureExtractor::feature_hist(std::vector<cv::Mat>& integ_orient, cv::Rect& roi, double* hist)
{
	int chns = integ_orient.size();
	for (int i = 0; i < chns; ++i)
	{
		hist[i] =	integ_orient[i].at<double>(roi.y+roi.height, roi.x+roi.width)+
					integ_orient[i].at<double>(roi.y, roi.x)-
					integ_orient[i].at<double>(roi.y, roi.x+roi.width)-
					integ_orient[i].at<double>(roi.y+roi.height,roi.x);
	}
}