/************************************************************************/
/* Copyright(C), Zhejiang University                                    */
/* FileName: sdalf_re_id.hpp                                            */
/* Author: Wen Wang                                                     */
/* Version: 1.0.0                                                       */
/* Date:                                                                */
/* Description: implementation of Bazzani, L., Cristani, M., Murino, V.: 
   Symmetry-driven accumulation of local features for human characterization 
   and re-identification. Comput. Vis. Image Underst. 117(2), 130¨C144(2013)
   Project page: http://www.lorisbazzani.info/sdalf.html                */
/************************************************************************/
#include "person_re_id.h"
#include <string>
#include "opencv2/opencv.hpp"
#include "boost/filesystem.hpp"
#include "boost/function.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/bind.hpp"
#include "boost/date_time.hpp"
//#include "mixtures.h"
#include "wwmatrix.hpp"
#include <algorithm>
#include <numeric>

#ifdef __cplusplus
extern "C" {
#endif

#include "image_buffer.h"
#include "msr_util.h"

#ifdef __cplusplus
}
#endif

namespace pe_re_id
{

	class SdalfFeature
	{
	public:
		struct										//Division in 3 part and kernel map computation
		{
			//WWMatrix<double> map_krnl;
			cv::Mat map_krnl;
			int TLanti;
			int BUsim;
			int LEGsim;
			int HDanti;
			cv::Rect head_det;
			bool head_det_flag;
			bool is_ready;
		} mapkrnl_div3;

		struct
		{
			//WWMatrix<double> mvec;
			//WWMatrix<double> pvec;
			cv::Mat mvec;
			cv::Mat pvec;
			bool is_ready;
		} Blobs;									//MSCR

		//WWMatrix<double> whisto2;					//Weighted HSV histogram
		struct
		{
			cv::Mat whisto;
			bool is_ready;
		}whisto2;

		struct
		{
			//std::vector< WWMatrix<int> > patch;
			std::vector< cv::Mat > patch;
			std::vector<double> entr;
			std::vector<double> w_ncc;
			std::vector<int> pos;
			std::vector<double> lbph;
			std::vector<int> numel;
			bool is_ready;
		} max_txpatch[3];
		SdalfFeature(){mapkrnl_div3.is_ready = false;Blobs.is_ready = false;
		whisto2.is_ready = false;max_txpatch[0].is_ready = false;
		max_txpatch[1].is_ready = false;max_txpatch[2].is_ready = false;}
	};

	class SdalfPeReId : public PersonReId
	{
	public:
		SdalfPeReId();								//default constructor, re_id model is estimated on line, and no save
		SdalfPeReId(const std::string& des);				//des is description of the probe, for example, des="wangwen_bluejacket_blackpants".
													//The program will search the existing models to find if des is already exist, if so, 
													//the model is loaded, if not, re_id model is estimated on line and saved titled des.
		int person_re_id(cv::Mat& frame, std::vector<cv::Rect>& det);
													//person re_id. frame contains person candidates bounded by det. If probe model is not
													//ready, then the probe model is estimated first. The matched index of det is returned.
		int person_re_id(/*input*/cv::Mat& frame, std::vector<cv::Rect>& det,
			/*output*/std::vector<int>& inds);		//person re_id. frame contains person candidates bounded by det. If probe model is not
													//ready, then the probe model is estimated first. The matched index of det is returned.
													//inds is the ranked indexes of det.

		void feature_extract(cv::Mat& img, void* features);	//extract feature used in person re_id of input img
		void probe_feature_extract(const std::string& probe_img_path);
													//Given path of probe images, extract features of images there.
		void probe_feature_extract(std::vector<cv::Mat>& imgs);
		void probe_feature_extract(cv::Mat& img);
		void gallery_feautre_extract(const std::string& gallery_img_path);
		//void gallery_feautre_extract(std::vector<WWMatrix<uchar> >& imgs);
		void gallery_feautre_extract(std::vector< cv::Mat >& imgs);
		void gallery_feautre_extract(std::vector< cv::Mat >& imgs, std::vector<SdalfFeature>& g);
		int feature_match(double* min_dis = NULL);
		int feature_match(std::vector<SdalfFeature>& g, double* min_dis = NULL);
		int feature_match(std::vector<int>& inds, double* min_dis = NULL);

		void save_feature(std::ofstream& to_file, SdalfFeature& f);
	//private:
		double SUBfac;								//subsampling factor
		int H,W;									//NORMALIZED dimensions
		int val;
		int delta[2];								//border limit (in order to avoid the local minimums at the image border) 
		double varW;								//variance of gaussian kernel (torso-legs)
		double alpha;								

		int NBINs[3];								//hsv quantization

		struct ParMSCR
		{
			double min_margin;						//Set margin parameter
			double ainc;							//
			int min_size;
			int filter_size;
			int verbosefl;
		} parMSCR;

		//dynamic MSCR clustering parameters
		int kmin;
		int kmax;
		int regularize;
		double th;
		int covoption;

		//Matching param
		double pyy;
		double pcc;
		double pee;

		struct
		{
			int N;
			int fac[6];
			double var;
			int NTrans;
			int DIM_OP[2];
			double thresh_entr;
			double thresh_cc;
		}tex_patch;

		bool maskon;
		bool dethead;

		std::vector<SdalfFeature> probe;			//probe
		std::vector<SdalfFeature> gallery;			//gallery

		bool probe_ready;
		bool save_model;
		bool save_match;
		std::string probe_des;						//probe model description
		std::string probe_images_path_;
		int probe_need_imgs;						//number of images needed to model probe
		double distance_thresh;						//distance threshold
		cv::Mat img_hsv;
		std::vector<cv::Mat> probe_imgs;
		std::vector<cv::Mat> gallery_imgs;

		void load_param();
		void load_probe(const std::string& probe_model_path);
		void save_probe();
		//void mapkrnl_div3(WWMatrix<uchar>& img, SdalfFeature* sf);
		void mapkrnl_div3(cv::Mat& img, SdalfFeature* sf);
													//Division in 3 part and kernel map computation
		//void extractMSCR(WWMatrix<uchar>& img, SdalfFeature* sf);
		void extractMSCR(cv::Mat& img, SdalfFeature* sf);
													//Features Extraction: MSCR
		//void extractwHSV(WWMatrix<uchar>& img, SdalfFeature* sf);
		void extractwHSV(cv::Mat& img, SdalfFeature* sf);
													//Features Extraction: part-based weighted HSV histogram
		//void extractTxpatch(WWMatrix<uchar>& img, SdalfFeature* sf);
		void extractTxpatch(cv::Mat& img, SdalfFeature* sf);
													//Maximally-CC textured patch computation with clustering with Mean Shift on LBP histogram and patch position + 1NN
		void wHSVmatch(/*input*/std::vector<SdalfFeature>& p, std::vector<SdalfFeature>& g,
			/*output*/std::vector<double>& dis);
		void MSCRmatch(/*input*/std::vector<SdalfFeature>& p, std::vector<SdalfFeature>& g,
			/*output*/std::vector<double>& dis);
		void save_result(std::vector<cv::Mat>& g, std::vector<double>& dis);
		double dissym_div(int x, cv::Mat& img_hsv, cv::Mat& msk, int delta1, double alpha);
		double sym_div(int x, cv::Mat& img_hsv, cv::Mat& msk, int delta2, double alpha);
		double sym_dissimilar(int x, cv::Mat& img_hsv, cv::Mat& msk, int delta1, double nan = 0);
		typedef boost::function<double(int/*, cv::Mat, cv::Mat, int, double*/)> func;
		//int fminbnd(double SdalfPeReId::*f (int, cv::Mat, cv::Mat, int, double), int low, int up, cv::Mat& img_hsv, cv::Mat& msk, int delta, double alpha);
		int fminbnd(func f, int low, int up);
		void illuminant_normalization(cv::Mat& src, cv::Mat& dst);
		void detect_mscr_masked(/*input*/WWMatrix<double>& img, WWMatrix<double>& mask,
			/*output*/WWMatrix<double>& mvec, WWMatrix<double>& pvec);
		void detection(/*input*/cv::Mat& img, cv::Mat& mask, cv::Rect& region, 
			/*output*/cv::Mat& mvec, cv::Mat& pvec);
		void eliminate_equivalentblobs(/*input*/cv::Mat& mvec, cv::Mat& pvec, 
			/*output*/cv::Mat& mv, cv::Mat& pv);
		void gau_kernel(/*input*/int sim, double var, int h, int w, 
			/*output*/cv::Mat& knl);
		void normpdf(int s, int e, double m, double sigm, cv::Mat& n);
		void whistcY(/*input*/cv::Mat& img, cv::Mat& weight, int bins,
			/*output*/cv::Mat& whist);
		double bhattacharyya(const cv::Mat& k, const cv::Mat& q);
	};
}