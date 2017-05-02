/************************************************************************/
/* Copyright(C), Zhejiang University                                    */
/* FileName: sdalf_re_id.cpp                                            */
/* Author: Wen Wang                                                     */
/* Version: 1.0.0                                                       */
/* Date:                                                                */
/* Description: implementation of Bazzani, L., Cristani, M., Murino, V.: 
   Symmetry-driven accumulation of local features for human characterization 
   and re-identification. Comput. Vis. Image Underst. 117(2), 130¨C144(2013)
   Project page: http://www.lorisbazzani.info/sdalf.html                */
/************************************************************************/

#include "sdalf_re_id.h"

namespace pe_re_id
{
	using namespace std;
	void SdalfPeReId::load_param()
	{
		string param_file(__FILE__);
#ifdef _WIN32 || _WIN64
		param_file = param_file.substr(0,param_file.find_last_of('\\'));
		param_file += "\\config\\sdalf_param.xml";
#endif
#ifdef __unix
		param_file = param_file.substr(0,param_file.find_last_of('/'));
		param_file += "/config/sdalf_param.xml";
#endif
		cv::FileStorage fs(param_file, cv::FileStorage::READ);
		cv::FileNode fn = fs["img_param"];
		SUBfac = (double)fn["SUBfac"];
		H = 128*SUBfac; W = 64*SUBfac;

		fn = fs["sym_var"];
		val = (int)fn["val"];
		delta[0] = H/val;delta[1] = W/val;
		varW = W/8.;
		alpha = (double)fn["alpha"];

		fn = fs["hsv"];
		cv::FileNodeIterator fnit = fn["NBINS"].begin(), fnitend = fn["NBINS"].end();
		for (int i = 0; fnit != fnitend; ++fnit, ++i) NBINs[i] = (int)(*fnit);

		fn = fs["MSCR"];
		parMSCR.min_margin = (double)fn["min_margin"];
		parMSCR.ainc = (double)fn["ainc"];
		parMSCR.min_size = (int)fn["min_size"];
		parMSCR.filter_size = (int)fn["filter_size"];
		parMSCR.verbosefl = (int)fn["verbosefl"];

		fn = fs["dynMSCR"];
		kmin = (int)fn["kmin"];
		kmax = (int)fn["kmax"];
		regularize = (int)fn["regularize"];
		th = (double)fn["th"];
		covoption = (int)fn["covoption"];

		fn = fs["Match"];
		pyy = (double)fn["pyy"];
		pcc = (double)fn["pcc"];
		pee = (double)fn["pee"];

		fn = fs["tex_patch"];
		tex_patch.N = (int)fn["N"];
		fnit = fn["fac"].begin(), fnitend = fn["fac"].end();
		for (int i = 0; fnit != fnitend; ++fnit, ++i) tex_patch.fac[i] = SUBfac*(int)(*fnit);
		tex_patch.var = SUBfac*(double)fn["var"];
		tex_patch.NTrans = (int)fn["NTrans"];
		tex_patch.DIM_OP[0] = SUBfac*(int)(*fn["DIM_OP"].begin());
		tex_patch.DIM_OP[1] = SUBfac*(int)(*(++fn["DIM_OP"].begin()));
		tex_patch.thresh_entr = (double)fn["Thresh_entr"];
		tex_patch.thresh_entr = (double)fn["Thresh_CC"];

		fn = fs["user_choice"];
		maskon = (int)fn["maskon"];
		dethead = (int)fn["dethead"];

		save_match = (int)fs["save_match"];
		probe_need_imgs = (int)fs["probe_need_imgs"];
		distance_thresh = (double)fs["distance_thresh"];
	}

	void SdalfPeReId::load_probe(const std::string& probe_model_path)
	{
		using namespace cv;
		using namespace boost::filesystem;

		FileStorage fs(probe_model_path, FileStorage::READ);

		FileNode fn = fs["probe"];
		FileNodeIterator fni_begin = fn.begin(), fni_end = fn.end();
		for (; fni_begin != fni_end; ++fni_begin)
		{
			SdalfFeature sf;

			FileNode tmp = (*fni_begin)["mapkrnl_div3"];
			tmp["map_krnl"] >> sf.mapkrnl_div3.map_krnl;
			tmp["TLanti"] >> sf.mapkrnl_div3.TLanti;
			tmp["BUsim"] >> sf.mapkrnl_div3.BUsim;
			tmp["LEGsim"] >> sf.mapkrnl_div3.LEGsim;
			tmp["HDanti"] >> sf.mapkrnl_div3.HDanti;
			tmp["head_det"] >> sf.mapkrnl_div3.head_det;
			tmp["head_det_flag"] >> sf.mapkrnl_div3.head_det_flag;
			tmp["is_ready"] >> sf.mapkrnl_div3.is_ready;

			tmp = (*fni_begin)["Blobs"];
			tmp["mvec"] >> sf.Blobs.mvec;
			tmp["pvec"] >> sf.Blobs.pvec;
			tmp["is_ready"] >> sf.Blobs.is_ready;

			tmp = (*fni_begin)["whisto2"];
			tmp["whisto"] >> sf.whisto2.whisto;
			tmp["is_ready"] >> sf.whisto2.is_ready;

			probe.push_back(sf);
		}
		fs.release();
	}

	void SdalfPeReId::save_probe()
	{
		using namespace cv;
		using namespace boost::filesystem;

		FileStorage fs;
		path model_path(__FILE__);
		model_path = model_path.parent_path();
		model_path /= "model";
		if (!is_directory(model_path / probe_des))
		{create_directories(model_path / probe_des);}
		model_path /= probe_des;

		fs.open((model_path / (probe_des+".xml")).string(), FileStorage::WRITE);
		fs << "probe" << "[";
		for (int i = 0; i < probe.size(); ++i)
		{
			fs << "{:";

			fs << "mapkrnl_div3" <<"{";
			fs << "map_krnl" << probe[i].mapkrnl_div3.map_krnl;
			fs << "TLanti" << probe[i].mapkrnl_div3.TLanti;
			fs << "BUsim" << probe[i].mapkrnl_div3.BUsim;
			fs << "LEGsim" << probe[i].mapkrnl_div3.LEGsim;
			fs << "HDanti" << probe[i].mapkrnl_div3.HDanti;
			fs << "head_det" << probe[i].mapkrnl_div3.head_det;
			fs << "head_det_flag" << probe[i].mapkrnl_div3.head_det_flag;
			fs << "is_ready" << probe[i].mapkrnl_div3.is_ready;
			fs << "}";

			fs << "Blobs" << "{";
			fs << "mvec" << probe[i].Blobs.mvec;
			fs << "pvec" << probe[i].Blobs.pvec;
			fs << "is_ready" << probe[i].Blobs.is_ready;
			fs << "}";

			fs << "whisto2" << "{";
			fs << "whisto" << probe[i].whisto2.whisto;
			fs << "is_ready" << probe[i].whisto2.is_ready;
			fs << "}";

			fs << "}";
		}
		fs << "]";
		fs.release();

		for (int i = 0; i < probe_imgs.size(); ++i)
		{
			std::stringstream ss;
			ss << setfill('0') << setw(3) << i;
			imwrite((model_path / ss.str()).string()+".jpg", probe_imgs[i]);
		}
	}

	SdalfPeReId::SdalfPeReId()
	{
		load_param();
		save_model = false;
		probe_ready = false;
	}

	SdalfPeReId::SdalfPeReId(const std::string& des)
	{
		using namespace boost::filesystem;
		load_param();
		save_model = true;
		probe_ready = false;
		probe_des = des;
		path model_path(__FILE__);
		model_path = model_path.parent_path();
		model_path /= "model";
		if (!is_directory(model_path / des))
		{create_directories(model_path / des);probe_ready = false;}
		else
		{
			model_path /= des;
			model_path /= (des+".xml");
			if (is_regular_file(model_path)){load_probe(model_path.string());probe_ready = true;}	//if .xml exists, then probe_ready is true
			else probe_feature_extract(model_path.parent_path().string());//probe_ready =false;
		}
	}

	void SdalfPeReId::probe_feature_extract(const std::string& probe_img_path)
	{
		probe_images_path_ = probe_img_path;
		if (probe_ready) return;
		using namespace boost::filesystem;
		using boost::filesystem::is_empty;

		path probe_path(probe_img_path);
		if (is_empty(probe_path)) {probe_ready = false; return;}	//if no files in probe_img_path, then probe_ready = false;

		auto xb = begin(directory_iterator(probe_path));
		auto xe = end(directory_iterator(probe_path));

		probe_imgs.clear();

		int imgs = 0;
		for (; xb != xe; ++xb)
		{
			SdalfFeature sf;
			if (".jpg" == xb->path().extension() || ".png" == xb->path().extension())
			{
				imgs++;
				cv::Mat img = cv::imread(xb->path().string());
				//WWMatrix<uchar> frame(img.rows, img.cols, img.channels());
				//frame.copyfromMat(img);
				feature_extract(img, &sf);
				probe_imgs.push_back(img);
				probe.push_back(sf);
			}
		}
		if (imgs) {probe_ready = true; save_probe();}	//if images exist and feature extracting done, then probe_ready is true
		else probe_ready = false;									//if no images in probe_img_path, then probe_ready = false;
	}

	void SdalfPeReId::probe_feature_extract(std::vector<cv::Mat>& imgs)
	{
		probe_imgs.clear();
		for (int i = 0; i < imgs.size(); ++i)
		{
			SdalfFeature sf;
			feature_extract(imgs[i], &sf);
			probe_imgs.push_back(imgs[i]);
			probe.push_back(sf);
		}
		if (imgs.size()) {probe_ready = true; save_probe();}	//if images exist and feature extracting done, then probe_ready is true
		else probe_ready = false;									//if no images in probe_img_path, then probe_ready = false;
	}

	void SdalfPeReId::probe_feature_extract(cv::Mat& img)
	{
		SdalfFeature sf;
		feature_extract(img, &sf);
		probe_imgs.push_back(img);
		probe.push_back(sf);
	}

	void SdalfPeReId::gallery_feautre_extract(const std::string& gallery_img_path)
	{
		using namespace boost::filesystem;
		using boost::filesystem::is_empty;

		path gallery_path(gallery_img_path);
		if (is_empty(gallery_path)) { return;}

		auto xb = begin(directory_iterator(gallery_path));
		auto xe = end(directory_iterator(gallery_path));

		if (save_match) if (gallery_imgs.size()) gallery_imgs.clear();
		
		int imgs = 0;
		for (; xb != xe; ++xb)
		{
			SdalfFeature sf;
			if (".jpg" == xb->path().extension() || ".png" == xb->path().extension())
			{
				imgs++;
				cv::Mat img = cv::imread(xb->path().string());
				//WWMatrix<uchar> frame(img.rows, img.cols, img.channels());
				//frame.copyfromMat(img);
				feature_extract(img, &sf);
				if (save_match) gallery_imgs.push_back(img);
				gallery.push_back(sf);
			}
		}								
	}

	//void SdalfPeReId::gallery_feautre_extract(std::vector<WWMatrix<uchar> >& imgs)
	void SdalfPeReId::gallery_feautre_extract(std::vector< cv::Mat >& imgs)
	{
		if (save_match) if (gallery_imgs.size()) gallery_imgs.clear();
		for (int i = 0; i < imgs.size(); ++i)
		{
			SdalfFeature sf;
			feature_extract(imgs[i], &sf);
			if (save_match) gallery_imgs.push_back(imgs[i]);
			gallery.push_back(sf);
		}
	}

	void SdalfPeReId::gallery_feautre_extract(std::vector< cv::Mat >& imgs, std::vector<SdalfFeature>& g)
	{
		for (int i = 0; i < imgs.size(); ++i)
		{
			SdalfFeature sf;
			feature_extract(imgs[i], &sf);
			g.push_back(sf);
		}
	}

	//void SdalfPeReId::feature_extract(WWMatrix<uchar>& img, void* features)
	void SdalfPeReId::feature_extract(cv::Mat& img, void* features)
	{
		mapkrnl_div3(img, (SdalfFeature*)features);
		//extractMSCR(img, (SdalfFeature*)features);
		extractwHSV(img, (SdalfFeature*)features);
		extractTxpatch(img, (SdalfFeature*)features);
	}

	int SdalfPeReId::feature_match(double* min_dis)
	{
		std::vector<double> final_dist_hist;
		wHSVmatch(probe, gallery, final_dist_hist);

		int idx = -1;
		double tmp = FLT_MAX;
		for (int i = 0; i < final_dist_hist.size(); ++i)
		{
			if (tmp > final_dist_hist[i])
			{
				tmp = final_dist_hist[i];
				idx = i;
			}
		}

		gallery.clear();
		if (min_dis) *min_dis = tmp;
		return idx;
	}

	int SdalfPeReId::feature_match(std::vector<SdalfFeature>& g, double* min_dis /* = NULL */)
	{
		std::vector<double> final_dist_hist;
		wHSVmatch(probe, g, final_dist_hist);

		int idx = -1;
		double tmp = FLT_MAX;
		for (int i = 0; i < final_dist_hist.size(); ++i)
		{
			if (tmp > final_dist_hist[i])
			{
				tmp = final_dist_hist[i];
				idx = i;
			}
		}

		//gallery.clear();
		if (min_dis) *min_dis = tmp;
		return idx;
	}

	int SdalfPeReId::feature_match(std::vector<int>& inds, double* min_dis)
	{
		std::vector<double> final_dist_hist;
		wHSVmatch(probe, gallery, final_dist_hist);
		std::vector<pair<double, int> > tmp_dis;
		for (int i = 0; i < final_dist_hist.size(); ++i)
			tmp_dis.push_back(pair<double, int>(final_dist_hist[i], i));
		std::sort(tmp_dis.begin(), tmp_dis.end(),[](pair<double, int> i, pair<double, int> j)->bool{return i.first < j.first;});
		
		inds.clear();
		for (int i = 0; i < tmp_dis.size(); ++i)
			inds.push_back(tmp_dis[i].second);
		if (min_dis) *min_dis = tmp_dis[0].first;

		return inds.size() ? inds[0] : -1;
	}

	void SdalfPeReId::wHSVmatch(std::vector<SdalfFeature>& p, std::vector<SdalfFeature>& g, std::vector<double>& dis)
	{
		int probes = p.size(), gallerys = g.size();
		if (dis.size()) dis.clear();
		for (int j = 0; j < gallerys; ++j)
		{
			double tmp = FLT_MAX;
			for (int ii = 0; ii < probes; ++ii)
			{
				double b = bhattacharyya(p[ii].whisto2.whisto, g[j].whisto2.whisto);
				if (tmp > b) tmp = b;
			}
			dis.push_back(tmp);
		}
	}

	void SdalfPeReId::MSCRmatch(std::vector<SdalfFeature>& p, std::vector<SdalfFeature>& g, std::vector<double>& dis)
	{

	}

	int SdalfPeReId::person_re_id(cv::Mat& frame, std::vector<cv::Rect>& det)
	{
		static bool isfirst = true;
		static int probe_num = 0;
		if (det.size() == 0) return -1;

		if (isfirst)					//The first frame
		{
			double if1 = 1.*det[0].height / frame.rows;								//height must be high enough
			double if2 = 1.*(det[0].tl().x + det[0].br().x) / 2 / frame.cols;		//center x must be near the image center
			if (if1 >= 0.8 && if2 >= 0.35 && if2 <= 0.65)
			{
				cv::Mat tmp;
				frame(det[0]).copyTo(tmp);
				cv::resize(tmp, tmp, cv::Size(W, H));
				probe_feature_extract(tmp);
				isfirst = false;
				probe_ready = true;
				probe_num++;
				return 0;
			}
			else return -1;
		}

		if (probe_ready)
		{
			//std::vector<WWMatrix<uchar> > gal;
			std::vector<cv::Mat> gal;
			for (int i = 0; i < det.size(); ++i)
			{
				cv::Mat g = frame(det[i]);
				cv::resize(g, g, cv::Size(W, H));
				gal.push_back(g.clone());
			}
			gallery_feautre_extract(gal);
			double min_dis;
			int idx = feature_match(&min_dis);

			if (min_dis > distance_thresh) return -1;

			if (probe_num < probe_need_imgs)
			{
				probe_feature_extract(gal[idx]);
				probe_num ++;
			}
			else if (probe_num == probe_need_imgs)
			{save_probe();probe_num++;}
			return idx;
		}
	}

	int SdalfPeReId::person_re_id(cv::Mat& frame, std::vector<cv::Rect>& det, std::vector<int>& inds)
	{
		static bool isfirst = true;
		static int probe_num = 0;
		if (det.size() == 0) return -1;

		if (isfirst)					//The first frame
		{
			double if1 = 1.*det[0].height / frame.rows;								//height must be high enough
			double if2 = 1.*(det[0].tl().x + det[0].br().x) / 2 / frame.cols;		//center x must be near the image center
			if (if1 >= 0.8 && if2 >= 0.35 && if2 <= 0.65)
			{
				cv::Mat tmp;
				frame(det[0]).copyTo(tmp);
				cv::resize(tmp, tmp, cv::Size(W, H));
				probe_feature_extract(tmp);
				isfirst = false;
				probe_ready = true;
				probe_num++;
				return 0;
			}
			else return -1;
		}

		if (probe_ready)
		{
			//std::vector<WWMatrix<uchar> > gal;
			std::vector<cv::Mat> gal;
			for (int i = 0; i < det.size(); ++i)
			{
				cv::Mat g = frame(det[i]);
				cv::resize(g, g, cv::Size(W, H));
				gal.push_back(g.clone());
			}
			gallery_feautre_extract(gal);
			double min_dis;
			int idx = feature_match(inds, &min_dis);

			if (min_dis > distance_thresh) return -1;

			if (probe_num < probe_need_imgs)
			{
				probe_feature_extract(gal[idx]);
				probe_num ++;
			}
			else if (probe_num == probe_need_imgs)
			{save_probe();probe_num++;}
			return idx;
		}
	}

	void SdalfPeReId::mapkrnl_div3(cv::Mat& img, SdalfFeature* sf)
	{
		//mask on will add later
		using namespace cv;
		//std::vector<Mat> bgrs,rgbs;
		//split(img, bgrs);
		//rgbs.push_back(bgrs[2]); rgbs.push_back(bgrs[1]); rgbs.push_back(bgrs[0]);

		//Mat rgb_img,img_d;
		//merge(rgbs, rgb_img);
		//rgb_img.convertTo(img_d, CV_64FC3, 1./255);
		//WWMatrix<double> wwimg(img.rows, img.cols, img.channels());
		//WWMatrix<double> wwimg_hsv(img.rows, img.cols, img.channels());
		//wwimg.copyfromMat(img_d);
		//rgbconvert(wwimg.data.get(), wwimg_hsv.data.get(), wwimg.rows*wwimg.cols, 
		//	wwimg.channels, 3, 1.);
		//Mat img_hsv(img.rows, img.cols, CV_64FC3);
		//wwimg_hsv.copytoMat(img_hsv);
		//std::vector<Mat> chs;
		//split(img_hsv, chs);
		Mat img_hsv_;
		//if (img_hsv.empty())
		//{cvtColor(img, img_hsv, CV_BGR2HSV_FULL);}
		cvtColor(img, img_hsv_, CV_BGR2HSV_FULL);
		img_hsv_.convertTo(img_hsv_, CV_64FC3, 1./255);
		//std::cout << chs[0] << std::endl;
		Mat msk = Mat::ones(img_hsv_.rows, img_hsv_.cols, img_hsv_.type());
		//func f = &pe_re_id::SdalfPeReId::dissym_div;
		boost::function<double(int/*, cv::Mat&, cv::Mat&, int, double*/)> f = 
			boost::bind(&SdalfPeReId::dissym_div, this, _1, img_hsv_, Mat(), delta[0], alpha);
		sf->mapkrnl_div3.TLanti = fminbnd(f, delta[0], H-delta[0]);
		//sf->mapkrnl_div3.TLanti = fminbnd(f, 2*delta[0], H-delta[0]);

		f = boost::bind(&SdalfPeReId::sym_div, this, _1, 
			img_hsv_(Rect(0,0,img_hsv_.cols,sf->mapkrnl_div3.TLanti+1)), Mat(), delta[1], alpha);
		sf->mapkrnl_div3.BUsim = fminbnd(f, delta[1], W-delta[1]);

		f = boost::bind(&SdalfPeReId::sym_div, this, _1, 
			img_hsv_(Rect(0,sf->mapkrnl_div3.TLanti+1, img_hsv_.cols, img_hsv_.rows-sf->mapkrnl_div3.TLanti-1)), Mat(), delta[1], alpha);
		sf->mapkrnl_div3.LEGsim = fminbnd(f, delta[1], W-delta[1]);

		f = boost::bind(&SdalfPeReId::sym_dissimilar, this, _1, 
			Mat(), msk, delta[0], 0);
		sf->mapkrnl_div3.HDanti = fminbnd(f, 20, sf->mapkrnl_div3.TLanti);
		if (sf->mapkrnl_div3.TLanti <= 20) sf->mapkrnl_div3.TLanti = H/2;
		if (sf->mapkrnl_div3.HDanti >= sf->mapkrnl_div3.TLanti) sf->mapkrnl_div3.HDanti = 20;

		//kernel-map computation
		sf->mapkrnl_div3.map_krnl = Mat::ones(H, W, CV_64FC1);		//or ones*-1???
		Rect up(0, sf->mapkrnl_div3.HDanti+1, W, sf->mapkrnl_div3.TLanti - sf->mapkrnl_div3.HDanti);
		Rect down(0, sf->mapkrnl_div3.TLanti+1, W, H - sf->mapkrnl_div3.TLanti - 1);
		Rect head(0, 0, W, sf->mapkrnl_div3.HDanti);
		gau_kernel(W/2, varW, head.height, W, sf->mapkrnl_div3.map_krnl(head));
		gau_kernel(sf->mapkrnl_div3.BUsim, varW, up.height, W, sf->mapkrnl_div3.map_krnl(up));
		gau_kernel(sf->mapkrnl_div3.LEGsim, varW, down.height, W, sf->mapkrnl_div3.map_krnl(down));
		double upmax = 1., downmax = 1., headmax = 1.;
		//std::cout << sf->mapkrnl_div3.map_krnl << std::endl;
		//minMaxLoc(sf->mapkrnl_div3.map_krnl(head), &headmax);
		//minMaxLoc(sf->mapkrnl_div3.map_krnl(up), NULL, &upmax);
		//minMaxLoc(sf->mapkrnl_div3.map_krnl(down), NULL, &downmax);
		//sf->mapkrnl_div3.map_krnl(head) /= headmax;
		//sf->mapkrnl_div3.map_krnl(up) /= upmax;
		//sf->mapkrnl_div3.map_krnl(down) /= downmax;
		//std::cout << sf->mapkrnl_div3.map_krnl << std::endl;

		sf->mapkrnl_div3.is_ready = true;
	}

	//void SdalfPeReId::extractMSCR(WWMatrix<uchar>& img, SdalfFeature* sf)
	void SdalfPeReId::extractMSCR(cv::Mat& img, SdalfFeature* sf)
	{
		//no mask use in this version
		using namespace cv;
		Mat in_img, hsv_img;
		illuminant_normalization(img, in_img);		// is it necessary?
		cvtColor(in_img, hsv_img, CV_BGR2HSV_FULL);
		std::vector<Mat> hsv;
		split(hsv_img, hsv);
		equalizeHist(hsv[2], hsv[2]);
		merge(hsv, hsv_img);
		cvtColor(hsv_img, in_img, CV_HSV2BGR_FULL);

		Mat mah, pah, mab, pab, mal, pal;
		Mat msk = Mat::ones(in_img.rows, in_img.cols, CV_8UC1);
		if (!sf->mapkrnl_div3.is_ready) mapkrnl_div3(img, sf);
		detection(in_img, msk, Rect(0,0,in_img.cols, sf->mapkrnl_div3.HDanti), mah, pah);
		detection(in_img, msk, Rect(0, sf->mapkrnl_div3.HDanti, in_img.cols, sf->mapkrnl_div3.TLanti),
			mab, pab);
		detection(in_img, msk, Rect(0, sf->mapkrnl_div3.TLanti, in_img.cols, in_img.rows-sf->mapkrnl_div3.TLanti),
			mal, pal);
		mab += sf->mapkrnl_div3.HDanti;
		mal += sf->mapkrnl_div3.TLanti;
		Mat tmp = mah.t();
		sf->Blobs.mvec.push_back(tmp);
		tmp = mab.t();
		sf->Blobs.mvec.push_back(tmp);
		tmp = mal.t();
		sf->Blobs.mvec.push_back(tmp);
		sf->Blobs.mvec = sf->Blobs.mvec.t();

		tmp = pah.t();
		sf->Blobs.pvec.push_back(tmp);
		tmp = pab.t();
		sf->Blobs.pvec.push_back(tmp);
		tmp = pal.t();
		sf->Blobs.pvec.push_back(tmp);
		sf->Blobs.pvec = sf->Blobs.pvec.t();

		sf->Blobs.is_ready = true;
	}

	//void SdalfPeReId::extractwHSV(WWMatrix<uchar>& img, SdalfFeature* sf)
	void SdalfPeReId::extractwHSV(cv::Mat& img, SdalfFeature* sf)
	{
		if (!sf->mapkrnl_div3.is_ready) mapkrnl_div3(img, sf);
		//if (img_hsv.empty())
		//{cvtColor(img, img_hsv, CV_BGR2HSV_FULL);}
		cv::Mat img_hsv_;
		cvtColor(img, img_hsv_, CV_BGR2HSV_FULL);

		using namespace cv;
		std::vector<Mat> hsv;
		split(img_hsv_, hsv);
		equalizeHist(hsv[2], hsv[2]);

		hsv[0].convertTo(hsv[0], CV_64FC1, 1./255);
		hsv[1].convertTo(hsv[1], CV_64FC1, 1./255);
		hsv[2].convertTo(hsv[2], CV_64FC1, 1./255);

		/*Just for weighted HSV showing*/
		//std::vector<Mat> whsv;
		//whsv.push_back(255.*hsv[0].mul(sf->mapkrnl_div3.map_krnl));
		//whsv.push_back(255.*hsv[1].mul(sf->mapkrnl_div3.map_krnl));
		//whsv.push_back(255.*hsv[2].mul(sf->mapkrnl_div3.map_krnl));
		//whsv[0].convertTo(whsv[0], CV_8UC1);
		//whsv[1].convertTo(whsv[1], CV_8UC1);
		//whsv[2].convertTo(whsv[2], CV_8UC1);
		//Mat wimg, whsv1;
		//merge(whsv, whsv1);
		//cvtColor(whsv1, wimg, CV_HSV2BGR_FULL);
		//imshow("wimg", wimg);
		//imwrite("weighted_img.jpg", wimg);
		//std::cout << sf->mapkrnl_div3.map_krnl << std::endl;
		//Mat map;
		//map = sf->mapkrnl_div3.map_krnl*255;
		//map.convertTo(map, CV_8UC1);
		//imshow("map", map);
		//imwrite("map.jpg", map);
		//waitKey(0);
		/*Just for weighted HSV showing*/

		//Mat img_hsv_eq;
		//merge(hsv, img_hsv_eq);

		Rect up(0, sf->mapkrnl_div3.HDanti+1, W, sf->mapkrnl_div3.TLanti - sf->mapkrnl_div3.HDanti);
		Rect down(0, sf->mapkrnl_div3.TLanti+1, W, H - sf->mapkrnl_div3.TLanti - 1);
		Rect head(0, 0, W, sf->mapkrnl_div3.HDanti);

		int chbins = NBINs[0]+NBINs[1]+NBINs[2], totbins = chbins * 3;
		sf->whisto2.whisto = Mat::zeros(totbins, 1, CV_64FC1);

		//head
		whistcY(hsv[0](head), sf->mapkrnl_div3.map_krnl(head), NBINs[0], sf->whisto2.whisto(Rect(0,0,1,NBINs[0])));
		whistcY(hsv[1](head), sf->mapkrnl_div3.map_krnl(head), NBINs[1], sf->whisto2.whisto(Rect(0,NBINs[0],1,NBINs[1])));
		whistcY(hsv[2](head), sf->mapkrnl_div3.map_krnl(head), NBINs[2], sf->whisto2.whisto(Rect(0,NBINs[0]+NBINs[1],1,NBINs[2])));

		//body
		whistcY(hsv[0](up), sf->mapkrnl_div3.map_krnl(up), NBINs[0], sf->whisto2.whisto(Rect(0,chbins, 1, NBINs[0])));
		whistcY(hsv[1](up), sf->mapkrnl_div3.map_krnl(up), NBINs[1], sf->whisto2.whisto(Rect(0,chbins+NBINs[0], 1, NBINs[1])));
		whistcY(hsv[2](up), sf->mapkrnl_div3.map_krnl(up), NBINs[2], sf->whisto2.whisto(Rect(0,chbins+NBINs[0]+NBINs[1], 1, NBINs[2])));

		whistcY(hsv[0](down), sf->mapkrnl_div3.map_krnl(down), NBINs[0], sf->whisto2.whisto(Rect(0,2*chbins, 1, NBINs[0])));
		whistcY(hsv[1](down), sf->mapkrnl_div3.map_krnl(down), NBINs[1], sf->whisto2.whisto(Rect(0,2*chbins+NBINs[0], 1, NBINs[1])));
		whistcY(hsv[2](down), sf->mapkrnl_div3.map_krnl(down), NBINs[2], sf->whisto2.whisto(Rect(0,2*chbins+NBINs[0]+NBINs[1], 1, NBINs[2])));

		sf->whisto2.is_ready = true;
	}

	//void SdalfPeReId::extractTxpatch(WWMatrix<uchar>& img, SdalfFeature* sf)
	void SdalfPeReId::extractTxpatch(cv::Mat& img, SdalfFeature* sf)
	{

	}

	double SdalfPeReId::dissym_div(int x, cv::Mat& img_hsv_, cv::Mat& msk, int delta1, double alpha)
	{
		using namespace cv;
		Rect img_rect(0,0,img_hsv_.cols, img_hsv_.rows);
		Rect up_rect(0,x-delta1,img_hsv_.cols, delta1);
		Rect down_rect(0, x+1, img_hsv_.cols, delta1);

		//Mat tmp;img_hsv_.copyTo(tmp);
		//rectangle(tmp, up_rect, Scalar(0,255,0));
		//rectangle(tmp, down_rect, Scalar(0,0,255));
		//imshow("slide",tmp);waitKey(5);

		Mat imgUP, imgDOWN;
		img_hsv_(up_rect & img_rect).copyTo(imgUP);
		img_hsv_(down_rect & img_rect).copyTo(imgDOWN);
		//imgUP.convertTo(imgUP, CV_64FC3, 1./255);
		flip(imgDOWN, imgDOWN, 0);
		//imgDOWN.convertTo(imgDOWN, CV_64FC3, 1./255);
		Mat sub_ud = imgUP - imgDOWN;
		Mat mul_ud = sub_ud.mul(sub_ud);
		Scalar s = sum(mul_ud);

		double d1 = alpha*(1-sqrt(s[0]+s[1]+s[2]+s[3])/delta1);
		double ups = img_hsv_.cols * (x+1), downs = img_hsv_.cols*(img_hsv_.rows-x);
		double d2 = 1.*abs( ups-downs ) / max(ups, downs);			//mask area: not use

		return d1+(1-alpha)*d2;
	}

	double SdalfPeReId::sym_div(int x, cv::Mat& img_hsv_, cv::Mat& msk, int delta2, double alpha)
	{
		using namespace cv;
		Rect img_rect(0,0,img_hsv_.cols, img_hsv_.rows);
		Rect l_rect(x-delta2,0,delta2, img_hsv_.rows);
		Rect r_rect(x+1, 0, delta2, img_hsv_.rows);
		Mat imgL, imgR;
		img_hsv_(l_rect & img_rect).copyTo(imgL);
		img_hsv_(r_rect & img_rect).copyTo(imgR);
		//imgL.convertTo(imgL, CV_64FC3);
		flip(imgR, imgR, 1);
		//imgR.convertTo(imgR, CV_64FC3);
		Mat sub_lr = imgL - imgR;
		Scalar s = sum(sub_lr.mul(sub_lr));

		double d1 = alpha*(sqrt(s[0]+s[1]+s[2]+s[3])/delta2);
		double ups = img_hsv_.cols * (x+1), downs = img_hsv_.cols*(img_hsv_.rows-x);
		double d2 = 1.*abs( ups-downs ) / max(ups, downs);			//mask area: not use

		return d1+(1-alpha)*d2;
	}

	double SdalfPeReId::sym_dissimilar(int x, cv::Mat& img_hsv, cv::Mat& msk, int delta1, double nan)
	{
		using namespace cv;
		Rect img_rect(0,0,msk.cols, msk.rows);
		Rect up_rect(0,x-delta1,msk.cols, delta1);
		Rect down_rect(0, x+1, msk.cols, delta1);
		Mat mskUP, mskDOWN;
		msk(up_rect & img_rect).copyTo(mskUP);
		msk(down_rect & img_rect).copyTo(mskDOWN);
		//mskUP.convertTo(mskUP, CV_64F);
		//mskDOWN.convertTo(mskDOWN, CV_64F);
		//Mat sub_ud = mskUP - mskDOWN;
		Scalar s_up = sum(mskUP);
		Scalar s_down = sum(mskDOWN);
		double s = s_up[0]+s_up[1]+s_up[2]+s_up[3]-(s_down[0]+s_down[1]+s_down[2]+s_down[3]);
		return s>0?-s:s;
	}

	int SdalfPeReId::fminbnd(func f, int low, int up)
	{
		// use golden section
		static double phi = (1+sqrt(5.))/2.;
		static double es = 0.00001;
		double xu = up, xl = low;
		int maxit = up - low;
		int iter = 0;
		double xopt;
		while(1)
		{
			double d = (phi-1)*(xu-xl);
			double x1 = xl + d;
			double x2 = xu - d;
			double d1 = f(x1), d2 = f(x2);
			if (d1 < d2) {xopt = x1; xl = x2;}
			else {xopt = x2; xu = x1;}
			iter ++;
			double ea = (2-phi)*abs((xu-xl)/xopt)*100;
			if (ea <= es || iter >= maxit) break;
		}
		return (int)xopt;
	}

	void SdalfPeReId::illuminant_normalization(cv::Mat& src, cv::Mat& dst)
	{
		using namespace cv;
		std::vector<Mat> chs;
		split(src, chs);
		Mat r,g,b;
		chs[0].convertTo(chs[0], CV_64FC1);
		chs[1].convertTo(chs[1], CV_64FC1);
		chs[2].convertTo(chs[2], CV_64FC1);
		Scalar m;
		m = mean(chs[0]);
		chs[0] = (chs[0] / m[0]) * 98;
		m = mean(chs[1]);
		chs[1] = (chs[1] / m[0]) * 98;
		m = mean(chs[2]);
		chs[2] = (chs[2] / m[0]) * 98;
		
		merge(chs, dst);
		dst.convertTo(dst, src.type());
	}

	void SdalfPeReId::detect_mscr_masked(/*input*/WWMatrix<double>& img, WWMatrix<double>& mask,
		/*output*/WWMatrix<double>& mvec, WWMatrix<double>& pvec)
	{
		int msize[]={0,0,0};
		int arg_ndims;
		int out_dims[]={0,0,0};
		/*buffer *bf_image = new buffer,*bf_pvec = new buffer,
			*bf_pvec2 = new buffer, *bf_mask = new buffer;
		ebuffer *bf_elist = new ebuffer,*bf_thres = new ebuffer;
		buffer *bf_mvec = new buffer,*bf_mvec2 = new buffer,
			*bf_arate = new buffer,*bf_arate2 = new buffer;
		buffer *bf_ecdf = new buffer;*/

		buffer *bf_mvec1, *bf_pvec1, 
			*bf_arate1;
		ebuffer *bf_thres1;
		using boost::shared_ptr;
		shared_ptr<buffer> bf_image(new buffer);
		//shared_ptr<buffer> bf_pvec(bf_pvec1);
		shared_ptr<buffer> bf_pvec2(new buffer);
		shared_ptr<buffer> bf_mask(new buffer);
		shared_ptr<ebuffer> bf_elist(new ebuffer);
		//shared_ptr<ebuffer> bf_thres(new ebuffer);
		//shared_ptr<buffer> bf_mvec(bf_mvec1);
		shared_ptr<buffer> bf_mvec2(new buffer);
		//shared_ptr<buffer> bf_arate(bf_arate1);
		shared_ptr<buffer> bf_arate2(new buffer);
		shared_ptr<buffer> bf_ecdf(new buffer);

		int validcnt,nofedges,rows,cols,ndim;
		edgeval d_max;
		double d_mean;

		int tslist_flag=1;
		int min_size=60;           /* Default */
		double ainc=1.05;          /* Default */
		double min_marginf=0.0015; /* Default */
		edgeval min_margin;        /* Value after conversion */
		fpnum res=1e-4;            /* Default */
		int Ns=10;                 /* Default */
		int timesteps=200;         /* Default */
		int filter_size=3;         /* Default */
		int n8flag=0;              /* Default */
		int normfl=1;              /* Default */
		int blurfl=0;              /* Default */
		int verbosefl=1;           /* Default */

		bf_image->data = img.data.get();
		rows = bf_image->rows = img.rows;
		cols = bf_image->cols = img.cols;
		ndim = bf_image->ndim = img.channels;
		if((ndim!=1)&&(ndim!=3)) {
			printf("Size mismatch: <img> should be MxNxD where D=1 or 3.\n");
			return;
		}

		bf_mask->data = mask.data.get();
		rows = bf_mask->rows = mask.rows;
		cols = bf_mask->cols = mask.cols;
		ndim = bf_mask->ndim = mask.channels;
		if((ndim!=1)&&(rows!=bf_image->rows)&&cols!=bf_image->cols) {
			printf("Size mismatch: <mask> should be of the same dimension as <image>.\n");
			return;
		}

		min_marginf = parMSCR.min_margin;
		min_margin = min_marginf*EDGE_SCALE + EDGE_OFFSET;
		min_size = parMSCR.min_size;
		ainc = parMSCR.ainc;
		filter_size = parMSCR.filter_size;
		verbosefl = parMSCR.verbosefl;

		if(verbosefl) {
			/* Display parameter settings */
			printf("\nDescriptor parameter settings:\n");
			printf(" min_margin: %g\n",min_marginf);
			printf("  timesteps: %d\n",timesteps);
			printf("   min_size: %d\n",min_size);
			printf("       ainc: %g\n",ainc);
			printf("filter_size: %d\n",filter_size);
			printf("     n8flag: %d\n",n8flag);
			printf("     normfl: %d\n",normfl);
			printf("     blurfl: %d\n",blurfl);
			printf("  verbosefl: %d\n",verbosefl);
		}

		if (n8flag) nofedges = 4*rows*cols-3*rows-3*cols+2;
		else nofedges = 2*rows*cols-rows-cols;

		WWMatrix<double> ww_elist(4, nofedges);
		bf_elist->data = ww_elist.data.get();
		bf_elist->rows = ww_elist.rows;
		bf_elist->cols = ww_elist.cols;
		bf_elist->ndim = ww_elist.channels;
		
		if (blurfl)
		{
			blur_buffer(bf_image.get(), filter_size);
			filter_size = 1;
		}
		if (filter_size%2)
		{
			if (n8flag)
				if (normfl)
					d_max = image_to_edgelist_blur_n8_norm(bf_image.get(), bf_elist.get(), filter_size, verbosefl);
				else
					d_max = image_to_edgelist_blur_n8(bf_image.get(), bf_elist.get(), filter_size, verbosefl);
			else
				if (normfl)
					d_max = image_to_edgelist_blur_norm(bf_image.get(), bf_elist.get(), filter_size, verbosefl);
				else
					d_max = image_to_edgelist_blur(bf_image.get(), bf_elist.get(), filter_size, verbosefl);
		}
		else
		{printf("filter size should be odd.\n");exit(EXIT_FAILURE);}

		if (tslist_flag)
		{
			bf_thres1 = ebuffer_new(1, timesteps, 1);
			if (verbosefl) printf("order=%d\n", ndim);
			d_mean = evolution_thresholds2(bf_elist.get(), bf_thres1, ndim);
			if (verbosefl) printf("d_mean=%g\n", d_mean);
		}
		
		edgelist_to_bloblist_masked(&bf_mvec1, &bf_pvec1, &bf_arate1, bf_image.get(), bf_mask.get(),
			bf_elist.get(), bf_thres1, min_size, ainc, res, verbosefl);
		/*center_moments(bf_mvec.get(), bf_pvec.get());
		validcnt = bloblist_mark_invalid(bf_mvec.get(), min_size, bf_arate.get(), (fpnum)min_margin);
		validcnt = bloblist_shape_invalid(bf_mvec.get());*/
		center_moments(bf_mvec1, bf_pvec1);
		validcnt = bloblist_mark_invalid(bf_mvec1, min_size, bf_arate1, (fpnum)min_margin);
		validcnt = bloblist_shape_invalid(bf_mvec1);
		if (verbosefl) printf("validcnt=%d\n", validcnt);

		if (tslist_flag) ebuffer_free(bf_thres1);

		mvec = WWMatrix<double>(6, validcnt);
		bf_mvec2->data = mvec.data.get();
		bf_mvec2->rows = mvec.rows;
		bf_mvec2->cols = mvec.cols;
		bf_mvec2->ndim = mvec.channels;

		pvec = WWMatrix<double>(3, validcnt);
		bf_pvec2->data = pvec.data.get();
		bf_pvec2->rows = pvec.rows;
		bf_pvec2->cols = pvec.cols;
		bf_pvec2->ndim = pvec.channels;

		WWMatrix<double> mm_arate(bf_arate1->rows, validcnt);
		bf_arate2->data = mm_arate.data.get();
		bf_arate2->rows = mm_arate.rows;
		bf_arate2->cols = mm_arate.cols;
		bf_arate2->ndim = mm_arate.channels;

		bloblist_compact(bf_mvec1, bf_mvec2.get(), bf_pvec1, 
			bf_pvec2.get(), bf_arate1, bf_arate2.get());

		buffer_free(bf_mvec1);
		buffer_free(bf_pvec1);
		buffer_free(bf_arate1);

	}

	void SdalfPeReId::detection(cv::Mat& img, cv::Mat& mask, cv::Rect& region, cv::Mat& mvec, cv::Mat& pvec)
	{
		using namespace cv;
		Mat reg_img = img(region).clone();
		Mat reg_mask = mask(region).clone();
		cvtColor(reg_img, reg_img, CV_BGR2RGB);

		reg_img.convertTo(reg_img, CV_64FC3, 1./255);
		reg_mask.convertTo(reg_mask, CV_64FC1);

		WWMatrix<double> ww_img(reg_img.rows, reg_img.cols, reg_img.channels());
		ww_img.copyfromMat(reg_img);
		WWMatrix<double> ww_mask(reg_mask.rows, reg_mask.cols, reg_mask.channels());
		ww_mask.copyfromMat(reg_mask);
		WWMatrix<double> ww_mvec, ww_pvec;
		detect_mscr_masked(ww_img, ww_mask, ww_mvec, ww_pvec);
		Mat mmvec = Mat(ww_mvec.rows, ww_mvec.cols, CV_64FC1);
		Mat ppvec = Mat(ww_pvec.rows, ww_pvec.cols, CV_64FC1);
		ww_mvec.copytoMat(mmvec);
		ww_pvec.copytoMat(ppvec);

		ppvec /= 256.;

		eliminate_equivalentblobs(mmvec, ppvec, mvec, pvec);

	}

	void SdalfPeReId::eliminate_equivalentblobs(cv::Mat& mvec, cv::Mat& pvec, 
		cv::Mat& mv, cv::Mat& pv)
	{
		using namespace cv;
		int cols = mvec.cols;
		std::vector<bool> eliminate(cols, false);
		for (int i = 0; i < cols; ++i)
		{
			double c1_1 = mvec.at<double>(1, i);
			double c1_2 = mvec.at<double>(2, i);
			for (int j = i+1; j < cols; ++j)
			{
				double c2_1 = mvec.at<double>(1, j);
				double c2_2 = mvec.at<double>(2, j);
				double if1 = sqrt((c1_1-c2_1)*(c1_1-c2_1)+(c1_2-c2_2)*(c1_2-c2_2));
				double if2 = mvec.at<double>(0, i) / mvec.at<double>(0, j);
				Mat tmp = (pvec.col(i)-pvec.col(j));
				Mat tmp1 = tmp.mul(tmp);
				Scalar s = sum(tmp1);
				double if3 = sqrt(s[0]);
				if (if1 < 10 && if2 > 0.6 && if2 < 1.4 && if3 < 0.1)
					eliminate[j] = true;
			}
		}
		int el_num = std::accumulate(eliminate.begin(), eliminate.end(), 0);
		mv = Mat(mvec.rows, mvec.cols-el_num, mvec.type());
		pv = Mat(pvec.rows, pvec.cols-el_num, pvec.type());
		for (int i = 0, j = 0; i < cols; ++i)
		{
			if (eliminate[i]) {continue;}
			mvec.col(i).copyTo(mv.col(j));
			pvec.col(i).copyTo(pv.col(j++));
		}
	}

	void SdalfPeReId::gau_kernel(int sim, double var, int h, int w, cv::Mat& knl)
	{
		cv::Mat n;
		normpdf(0, W-1, (double)sim, var, n);
		for (int i = 0; i < h; ++i)
			n.copyTo(knl.row(i));
	}

	void SdalfPeReId::normpdf(int s, int e, double m, double sigm, cv::Mat& n)
	{
		if (sigm <= 0) return;
		if (std::max(n.rows, n.cols) != (e-s+1)) n = cv::Mat::zeros(1, e-s+1, CV_64FC1);
		for (int i = s; i <= e; ++i)
		{
			//n.at<double>(0, i) = exp(-0.5*(1.*(i-m)/sigm)*(1.*(i-m)/sigm))/(sqrt(2*PI)*sigm);
			n.at<double>(0,i) = 1./(e-s+1);
		}
	}

	void SdalfPeReId::whistcY(cv::Mat& img, cv::Mat& weight, int nbins, cv::Mat& whist)
	{
		//whist must be a binsX1 column vector
		if (whist.rows != nbins && whist.cols != 1) whist = cv::Mat::zeros(nbins, 1, CV_64FC1);

		double* bins = new double[nbins];
		for (int i = 0; i < nbins; ++i) 
		{
			bins[i] = 1.*i/(nbins-1);
		}
		int rows = img.rows, cols = img.cols;
		for (int i = 0;i < rows; ++i)
		{
			double* imgdata = img.ptr<double>(i);
			double* weightdata = weight.ptr<double>(i);
			for (int j = 0; j < cols; ++j)
			{
				int c = 0;
				while (c < nbins)
				{
					if ( (c == nbins-1) || (imgdata[j] < bins[c+1] && imgdata[j] >= bins[c]) )
					{
						whist.at<double>(c, 0) += weightdata[j];
						c = nbins + 1;
					}
					else
						c++;
				}
			}
		}
		delete [] bins;
	}

	double SdalfPeReId::bhattacharyya(const cv::Mat& k, const cv::Mat& q)
	{
		using namespace cv;
		Scalar sk = sum(k);
		Scalar sq = sum(q);

		Mat normk = k/sk[0];		//normalize
		Mat normq = q/sq[0];		//normalize

		Mat tmp;
		sqrt(normk.mul(normq), tmp);
		Scalar tmps = sum(tmp);

		double tmpd = 1 - tmps[0];
		return tmpd < 0 ? 0 : sqrt(tmpd);

		//cv::Mat tmp = k.t()*q;
		//cv::Mat lk = k.t()*k;
		//cv::Mat lq = q.t()*q;
		//return tmp.at<double>(0,0)/(sqrt(lk.at<double>(0,0))*sqrt(lq.at<double>(0,0)));
	}

	void SdalfPeReId::save_result(std::vector<cv::Mat>& g, std::vector<double>& dis)
	{
		using namespace cv;
		using namespace std;
		using namespace boost::filesystem;
		if (!probe_imgs.size())
		{
			using boost::filesystem::is_empty;

			path probe_path(probe_images_path_);
			if (is_empty(probe_path)) {return;}	//if no files in probe_img_path, then probe_ready = false;

			auto xb = begin(directory_iterator(probe_path));
			auto xe = end(directory_iterator(probe_path));

			for (; xb != xe; ++xb)
			{
				if (".jpg" == xb->path().extension() || ".png" == xb->path().extension())
				{
					cv::Mat img = cv::imread(xb->path().string());
					probe_imgs.push_back(img);
					break;
				}
			}
		}

		vector<pair<double, int> > tmp_dis;
		for (int i = 0; i < dis.size(); ++i)
			tmp_dis.push_back(pair<double, int>(dis[i], i));
		sort(tmp_dis.begin(), tmp_dis.end(),[](pair<double, int> i, pair<double, int> j)->bool{return i.first < j.first;});

		path save_path(__FILE__);
		save_path = save_path.parent_path();
		save_path /= "result";
		if (!is_directory(save_path))
		{create_directories(save_path);}
		
		boost::posix_time::ptime pt = boost::date_time::second_clock<boost::posix_time::ptime>::local_time();
		
		save_path /= (probe_des + to_iso_string(pt) + ".jpg");

		int tot = g.size()+1, cols = 10, rows = tot / cols + (tot % cols != 0);
		Mat img = Mat::zeros(rows*(128+10), cols*(64+10), CV_8UC3);
		Rect r(5,5,64,128);
		probe_imgs[0].copyTo(img(r));
		for (int i = 0; i < tmp_dis.size(); ++i)
		{
			r = Rect(((i+1)%10)*74+5,((i+1)/10)*138+5,64,128);
			g[tmp_dis[i].second].copyTo(img(r));
		}
		imshow(probe_des, img);
		waitKey(0);
		imwrite(save_path.string(), img);
	}

	void SdalfPeReId::save_feature(std::ofstream& to_file, SdalfFeature& f)
	{
		int rows = f.whisto2.whisto.rows;
		for (int i = 0; i < rows; ++i)
		{
			if (i == rows-1) to_file << f.whisto2.whisto.at<double>(i,0) << endl;
			else to_file << f.whisto2.whisto.at<double>(i,0) << ",";
		}
	}
}