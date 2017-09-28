/// @file imwrite.h
/// @brief 用来保存图像，第一次调用时新建一个以当前时间命名的文件夹，随后每调用一次，则将图像保存在该文件夹中，命名为n.jpg
/// @author 王文
/// @version 9.0
/// @date 2017-9-27

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