/************************************************************************/
/* Copyright(C), Zhejiang University                                    */
/* FileName: person_re_id.hpp                                           */
/* Author: Wen Wang                                                     */
/* Version: 1.0.0                                                       */
/* Date:                                                                */
/* Description: base class inherited by specific person re-identification
   method. There are two virtual member functions: feature_extract and 
   feature_match, that is two common steps used in methods of person re-id*/
/************************************************************************/
#ifndef	PERSON_RE_IDENTIFICATION_
#define PERSON_RE_IDENTIFICATION_

#include <vector>
#include <string>
//#include "wwmatrix.hpp"
#include "opencv2/opencv.hpp"

namespace pe_re_id
{
	class PersonReId
	{
	public:
		//virtual void feature_extract(std::vector< WWMatrix<uchar> >& img) = 0;
		//virtual void feature_extract(WWMatrix<uchar>& img, void* features) = 0;
		virtual void feature_extract(cv::Mat& img, void* features) = 0;
		//virtual void feature_extract(std::string& probe_img_path) = 0;
		//virtual void set_probe_model(std::string& probe_path, std::string& des) = 0;
		virtual int feature_match(double* min_dis) = 0;
	};
}

#endif