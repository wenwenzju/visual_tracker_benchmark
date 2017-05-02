#ifndef _IMWRITE_H_
#define _IMWRITE_H_

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iomanip>

bool imwrite(const cv::Mat& img)
{
	static bool rr = false;
	static int cnt = 0;
	static char fn[20];
	if (img.empty()){cnt = 0; rr = false;return true;}
	if(!rr)
	{
		time_t ft = time(0);
		strftime( fn, sizeof(fn), "%Y%m%d%H%M%S",localtime(&ft) ); 
		char maked[25] = "md ";
		strcat(maked, fn);
		system(maked);
		rr = true;
	}
	char cntc[20];
	itoa(cnt, cntc, 10);
	strcat(cntc, ".jpg");
	std::string imgname(fn);
	imgname += "/";
	imgname += cntc;
	cnt ++;
	return cv::imwrite(imgname, img);
}

#endif