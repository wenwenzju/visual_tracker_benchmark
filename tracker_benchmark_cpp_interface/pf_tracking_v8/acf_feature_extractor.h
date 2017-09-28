/// @file acf_feature_extractor.h
/// @brief 特征提取封装，特征包括LUV色彩空间的直方图、梯度方向直方图\n
/// 特征的计算使用Piotr's Computer Vision Matlab Toolbox提供的接口：convConst.h convConst.cpp gradient.h gradient.cpp rgbConvert.hpp sse.hpp wrappers.hpp wrappers.cpp\n
/// 网址：https://pdollar.github.io/toolbox/
/// @author 王文
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

/// @brief 模板池中的模型，存的是特征
struct MatchModel
{
	WWMatrix<double> chnL;				//binlx1
	WWMatrix<double> chnU;				//binux1
	WWMatrix<double> chnV;				//binvx1
	std::vector<WWMatrix<double> > hog;	//(cellswxcellsh+1)x(binhogx1)
};

/// @brief 特征提取以及模板匹配都在这里
class AcfFeatureExtractor
{
public:
	//AcfFeatureExtractor():MINL(0.), MAXL(0.37), MINU(0.), MAXU(1.), MINV(0.), MAXV(0.89){}
	/// @brief 构造函数
	/// @param [in] binL L通道直方图bin个数
	/// @param [in] binU U通道直方图bin个数
	/// @param [in] binV V通道直方图bin个数
	/// @param [in] binHOG 梯度直方图方向个数
	/// @param [in] csh 
	/// @param [in] csw 模板被分成csh x csw 的网格，每个网格分别提取特征
	/// @param [in] ipn 模板池中初始时刻模板
	/// @param {in] cpn 模板池中当前时刻模板
	AcfFeatureExtractor(int binL=16, int binU=16, int binV=4, int binHOG=6, int csh=4, int csw=2, int ipn=1, int cpn=1):binl(binL),binu(binU),binv(binV),binhog(binHOG),cellsh(csh),cellsw(csw),init_probe_need(ipn),current_probe_need(cpn),
		MINL(0.), MAXL(0.37), MINU(0.), MAXU(1.), MINV(0.), MAXV(0.89){}
		//MINL(0.), MAXL(1.), MINU(0.), MAXU(1.), MINV(0.), MAXV(1.){}
	void feature_extract(cv::Mat& img, std::vector<cv::Mat>& feature);

	/// @brief 计算观测模型target-specific detector中，粒子与模板池的距离，调用之前，应先调用feature_extract(Mat)和updata_match_model(Rect)
	/// @param [in] roi 粒子
	/// @param [in] lr 学习率，论文中的1-gamma
	/// @return 粒子与模板池的距离
	double calc_roi_scores(cv::Rect& roi, double lr);

	/// @brief 更新模板池。首先计算roi内各通道的梯度直方图，然后按照论文中公式(7)上方的规则存入私有变量probe，即模板池
	/// @param [in] roi 跟踪结果
	/// @return none.
	void update_match_model(cv::Rect& roi);
	void feature_extract(cv::Mat& img, cv::Rect& roi, int binL, int binU, int binV, int binHOG, std::vector<double>& feature);
	void feature_extract(cv::Mat& img, cv::Rect& roi, int binL, int binU, int binV, int binHOG, std::vector<std::vector<double> >& feature);
	void feature_extract(cv::Mat& img, cv::Rect& roi, int binL, int binU, int binV, int binHOG, std::vector<cv::Mat>& feature);
	void feature_extract(cv::Mat& img, cv::Rect& roi, int binL, int binU, int binV, int binHOG, cv::Mat& feature);

	/// @brief 提取图像特征，先将BGR的img转成RGB的img，然后计算LUV通道和梯度幅值以及梯度方向
	/// @param [in] img 输入图像
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

	/// @brief 将RGB图像转成指定的色彩空间
	/// @param [in] wwm 输入的RGB图像，大小r x c x chn
	/// @param [out] dst 转化后的色彩空间
	/// @param [in] rxc RGB图像单通道的大小，r x c
	/// @param [in] chn RGB图像的通道数
	/// @param [in] flags 指定转化成的色彩空间，flags=2表示转化成LUV
	void rgbConvert(WWMatrix<uchar>& wwm, WWMatrix<float>& dst, int rxc, int chn, int flags = 2);
	/// @brief 滤波
	/// @param [in] src 待滤波数据
	/// @param [out] dst 滤波后的数据
	/// @param [in] type 滤波形式
	/// @param [in] r 滤波半径
	/// @param [in] s stride
	void convTri(WWMatrix<float>& src, WWMatrix<float>& dst, const char* type, double r, double s);
	void addChn(WWMatrix<float>& data, std::vector<cv::Mat>& feature);
	void feature_hist(cv::Mat& fea, char bin, double minfea, double maxfea, std::vector<double>::iterator& b);
	void feature_hist(WWMatrix<float>& data, cv::Rect& roi, int ch, char bin, double minfea, double maxfea, std::vector<double>::iterator& b);
	void feature_hist(WWMatrix<float>& data, cv::Rect& roi, int ch, char bin, double minfea, double maxfea, cv::MatIterator_<double>& it);

	/// @brief 计算roi内LUV通道的特征直方图
	/// @param [in] data L、U或V单通道
	/// @param [in] roi roi
	/// @param [in] ch 第几个通道，即指定L或U或V通道
	/// @param [in] bin 直方图bin个数
	/// @param [in] minfea 通道数据的最小值
	/// @param [in] maxfea 通道数据的最大值
	/// @param [out] hist 直方图
	void feature_hist(WWMatrix<float>& data, cv::Rect& roi, int ch, char bin, double minfea, double maxfea, double* hist);
	void feature_hist(cv::Mat& integ_orient, cv::Rect& roi, double* hist);

	/// @brief 用积分图计算方向梯度直方图
	/// @param [in] integ_orient 方向梯度的积分图，size=binhog
	/// @param [in] roi roi
	/// @param [out] hist 直方图
	void feature_hist(std::vector<cv::Mat>& integ_orient, cv::Rect& roi, double* hist);

	/// @brief 计算两个模型间的距离
	double distance(MatchModel& mm1, MatchModel& mm2);
	/// @brief 计算两组数据间的巴氏距离
	double bhattacharyya(double* d1, double* d2, int len, bool norm_flag = true);

	/// @brief 计算roi区域内各个通道的直方图特征，应先调用feature_extract(Mat)计算各个通道
	/// @param [out] mm 封装过的模板特征
	/// @param [in] roi roi
	void calc_model(MatchModel& mm, cv::Rect& roi);

	/// @brief 计算候选与模板池的距离
	/// @param [in] mm 候选
	/// @param [in] lr 学习率，论文中的1-gamma
	double distance_from_probe(MatchModel& mm, double lr);
	double get_luv_cells_weight();
	const double MINL, MAXL, MINU, MAXU, MINV, MAXV;

	void print_freature(const MatchModel& mm);
};

#endif