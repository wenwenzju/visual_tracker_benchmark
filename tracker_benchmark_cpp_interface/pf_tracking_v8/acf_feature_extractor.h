/// @file acf_feature_extractor.h
/// @brief ������ȡ��װ����������LUVɫ�ʿռ��ֱ��ͼ���ݶȷ���ֱ��ͼ\n
/// �����ļ���ʹ��Piotr's Computer Vision Matlab Toolbox�ṩ�Ľӿڣ�convConst.h convConst.cpp gradient.h gradient.cpp rgbConvert.hpp sse.hpp wrappers.hpp wrappers.cpp\n
/// ��ַ��https://pdollar.github.io/toolbox/
/// @author ����
/// @version 8.0
/// @date 2017-9-27

#ifndef ACF_FEATURE_EXTRACTOR_H
#define ACF_FEATURE_EXTRACTOR_H

#include "wwmatrix.hpp"
#include "rgbConvert.hpp"
#include "convConst.h"
#include "gradient.h"
#include "opencv2/opencv.hpp"
#include "boost/thread.hpp"

/// @brief ģ����е�ģ�ͣ����������
struct MatchModel
{
	WWMatrix<double> chnL;				//binlx1
	WWMatrix<double> chnU;				//binux1
	WWMatrix<double> chnV;				//binvx1
	std::vector<WWMatrix<double> > hog;	//(cellswxcellsh+1)x(binhogx1)
};

/// @brief ������ȡ�Լ�ģ��ƥ�䶼������
class AcfFeatureExtractor
{
public:
	//AcfFeatureExtractor():MINL(0.), MAXL(0.37), MINU(0.), MAXU(1.), MINV(0.), MAXV(0.89){}
	/// @brief ���캯��
	/// @param [in] binL Lͨ��ֱ��ͼbin����
	/// @param [in] binU Uͨ��ֱ��ͼbin����
	/// @param [in] binV Vͨ��ֱ��ͼbin����
	/// @param [in] binHOG �ݶ�ֱ��ͼ�������
	/// @param [in] csh 
	/// @param [in] csw ģ�屻�ֳ�csh x csw ������ÿ������ֱ���ȡ����
	/// @param [in] ipn ģ����г�ʼʱ��ģ��
	/// @param {in] cpn ģ����е�ǰʱ��ģ��
	AcfFeatureExtractor(int binL=16, int binU=16, int binV=4, int binHOG=6, int csh=4, int csw=2, int ipn=1, int cpn=1):binl(binL),binu(binU),binv(binV),binhog(binHOG),cellsh(csh),cellsw(csw),init_probe_need(ipn),current_probe_need(cpn),
		MINL(0.), MAXL(0.37), MINU(0.), MAXU(1.), MINV(0.), MAXV(0.89){}
		//MINL(0.), MAXL(1.), MINU(0.), MAXU(1.), MINV(0.), MAXV(1.){}
	void feature_extract(cv::Mat& img, std::vector<cv::Mat>& feature);

	/// @brief ����۲�ģ��target-specific detector�У�������ģ��صľ��룬����֮ǰ��Ӧ�ȵ���feature_extract(Mat)��updata_match_model(Rect)
	/// @param [in] roi ����
	/// @param [in] lr ѧϰ�ʣ������е�1-gamma
	/// @return ������ģ��صľ���
	double calc_roi_scores(cv::Rect& roi, double lr);

	/// @brief ����ģ��ء����ȼ���roi�ڸ�ͨ�����ݶ�ֱ��ͼ��Ȼ���������й�ʽ(7)�Ϸ��Ĺ������˽�б���probe����ģ���
	/// @param [in] roi ���ٽ��
	/// @return none.
	void update_match_model(cv::Rect& roi);
	void feature_extract(cv::Mat& img, cv::Rect& roi, int binL, int binU, int binV, int binHOG, std::vector<double>& feature);
	void feature_extract(cv::Mat& img, cv::Rect& roi, int binL, int binU, int binV, int binHOG, std::vector<std::vector<double> >& feature);
	void feature_extract(cv::Mat& img, cv::Rect& roi, int binL, int binU, int binV, int binHOG, std::vector<cv::Mat>& feature);
	void feature_extract(cv::Mat& img, cv::Rect& roi, int binL, int binU, int binV, int binHOG, cv::Mat& feature);

	/// @brief ��ȡͼ���������Ƚ�BGR��imgת��RGB��img��Ȼ�����LUVͨ�����ݶȷ�ֵ�Լ��ݶȷ���
	/// @param [in] img ����ͼ��
	/// @return none.
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
	//cv::Mat orients;	//hxwxbinhog, h: image height, w: image width. integrogram of hog
	std::vector<cv::Mat> orients;
	std::vector<MatchModel> probe;

	/// @brief ��RGBͼ��ת��ָ����ɫ�ʿռ�
	/// @param [in] wwm �����RGBͼ�񣬴�Сr x c x chn
	/// @param [out] dst ת�����ɫ�ʿռ�
	/// @param [in] rxc RGBͼ��ͨ���Ĵ�С��r x c
	/// @param [in] chn RGBͼ���ͨ����
	/// @param [in] flags ָ��ת���ɵ�ɫ�ʿռ䣬flags=2��ʾת����LUV
	void rgbConvert(WWMatrix<uchar>& wwm, WWMatrix<float>& dst, int rxc, int chn, int flags = 2);
	/// @brief �˲�
	/// @param [in] src ���˲�����
	/// @param [out] dst �˲��������
	/// @param [in] type �˲���ʽ
	/// @param [in] r �˲��뾶
	/// @param [in] s stride
	void convTri(WWMatrix<float>& src, WWMatrix<float>& dst, const char* type, double r, double s);
	void addChn(WWMatrix<float>& data, std::vector<cv::Mat>& feature);
	void feature_hist(cv::Mat& fea, char bin, double minfea, double maxfea, std::vector<double>::iterator& b);
	void feature_hist(WWMatrix<float>& data, cv::Rect& roi, int ch, char bin, double minfea, double maxfea, std::vector<double>::iterator& b);
	void feature_hist(WWMatrix<float>& data, cv::Rect& roi, int ch, char bin, double minfea, double maxfea, cv::MatIterator_<double>& it);

	/// @brief ����roi��LUVͨ��������ֱ��ͼ
	/// @param [in] data L��U��V��ͨ��
	/// @param [in] roi roi
	/// @param [in] ch �ڼ���ͨ������ָ��L��U��Vͨ��
	/// @param [in] bin ֱ��ͼbin����
	/// @param [in] minfea ͨ�����ݵ���Сֵ
	/// @param [in] maxfea ͨ�����ݵ����ֵ
	/// @param [out] hist ֱ��ͼ
	void feature_hist(WWMatrix<float>& data, cv::Rect& roi, int ch, char bin, double minfea, double maxfea, double* hist);
	void feature_hist(cv::Mat& integ_orient, cv::Rect& roi, double* hist);

	/// @brief �û���ͼ���㷽���ݶ�ֱ��ͼ
	/// @param [in] integ_orient �����ݶȵĻ���ͼ��size=binhog
	/// @param [in] roi roi
	/// @param [out] hist ֱ��ͼ
	void feature_hist(std::vector<cv::Mat>& integ_orient, cv::Rect& roi, double* hist);

	/// @brief ��������ģ�ͼ�ľ���
	double distance(MatchModel& mm1, MatchModel& mm2);
	/// @brief �����������ݼ�İ��Ͼ���
	double bhattacharyya(double* d1, double* d2, int len, bool norm_flag = true);

	/// @brief ����roi�����ڸ���ͨ����ֱ��ͼ������Ӧ�ȵ���feature_extract(Mat)�������ͨ��
	/// @param [out] mm ��װ����ģ������
	/// @param [in] roi roi
	void calc_model(MatchModel& mm, cv::Rect& roi);

	/// @brief �����ѡ��ģ��صľ���
	/// @param [in] mm ��ѡ
	/// @param [in] lr ѧϰ�ʣ������е�1-gamma
	double distance_from_probe(MatchModel& mm, double lr);
	double get_luv_cells_weight();
	const double MINL, MAXL, MINU, MAXU, MINV, MAXV;

	void print_freature(const MatchModel& mm);
};

#endif