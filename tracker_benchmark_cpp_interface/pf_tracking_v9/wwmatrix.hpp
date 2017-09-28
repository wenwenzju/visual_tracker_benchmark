/// @file wwmatrix.hpp
/// @brief 很好用的矩阵类，用于opencv的Mat与Piotr's Computer Vision Matlab Toolbox工具箱接口的兼容性过渡
/// @author 王文
/// @version 9.0
/// @date 2017-9-27

#ifndef __WWMATRIX__
#define __WWMATRIX__

#include "opencv2/opencv.hpp"
#include "boost/smart_ptr.hpp"

template<class T>
class WWMatrix
{
public:
	WWMatrix();
	WWMatrix(int n);		//one dimension
	WWMatrix(int r, int c);	//two dimension
	WWMatrix(int r, int c, int h);	//three dimension
	//WWMatrix(cv::Mat& img);
	int channels;
	int rows, cols;
	boost::shared_array<T> data;
	T & operator() (int x);
	T & operator() (int x, int y);
	T & operator() (int x, int y, int z);
	void creat(int n);
	void creat(int r, int c);
	void creat(int r, int c, int h);
	void copyfromMat(cv::Mat& img);
	void copytoMat(cv::Mat& img);
	void copytoMat(cv::Mat& img, int c);
	void copyTo(WWMatrix<T>& wwm);
	void release();
private:
	inline void init(){for (int i = 0; i < rows*cols*channels; i++) data[i] = 0;}
};

template<class T>
WWMatrix<T>::WWMatrix()
{
	data = boost::shared_array<T>(new T[0]);
	channels = 0;
	rows = cols = 0;
	init();
}

template<class T>
void WWMatrix<T>::creat(int n)
{
	data = boost::shared_array<T>(new T[n]);
	channels = 1;
	cols = 1;
	rows = n;
	init();
}

template<class T>
WWMatrix<T>::WWMatrix(int n)
{
	data =boost::shared_array<T>( new T[n] );
	channels = 1;
	rows = n;
	cols = 1;
	init();
}

template<class T>
void WWMatrix<T>::creat(int r, int c)
{
	data = boost::shared_array<T>(new T[r*c]);
	channels = 1;
	rows = r; cols = c;
	init();
}

template<class T>
WWMatrix<T>::WWMatrix(int r, int c)
{
	data =boost::shared_array<T>( new T[r*c] );
	channels = 1;
	rows = r;
	cols = c;
	init();
}

template<class T>
void WWMatrix<T>::creat(int r, int c, int h)
{
	data = boost::shared_array<T>(new T[r*c*h]);
	channels = h; rows = r; cols = c;
	init();
}

template<class T>
WWMatrix<T>::WWMatrix(int r, int c, int h)
{
	data =boost::shared_array<T>( new T[r*c*h] );
	channels = h;
	rows = r;
	cols = c;
	init();
}

template<class T>
void WWMatrix<T>::copyfromMat(cv::Mat& img)
{
	using namespace cv;
	assert(sizeof(data[0]) == (img.elemSize()/img.channels()));

	std::vector<Mat> chs;
	split(img, chs);

	int dep = sizeof(data[0]);
	for (int i = 0; i < chs.size(); i++)
	{
		Mat tmp = chs[i].t();

		memcpy(data.get()+i*rows*cols, tmp.data, dep*rows*cols);
	}
}

template<class T>
void WWMatrix<T>::copytoMat(cv::Mat& img)
{
	using namespace cv;

	std::vector<Mat> chs;
	int dep = sizeof(data[0]);
	switch (dep)
	{
	case 1:		//unsigned char
		for (int i = 0; i < channels; i++)
		{
			Mat tmp(cols, rows, CV_8UC1);
			chs.push_back(tmp);
		}
		break;
	case 4:		//float
		for (int i = 0; i < channels; i++)
		{
			Mat tmp(cols, rows, CV_32FC1);
			chs.push_back(tmp);
		}
		break;
	case 8:		//double
		for (int i = 0; i < channels; i++)
		{
			Mat tmp(cols, rows, CV_64FC1);
			chs.push_back(tmp);
		}
		break;
	default:
		std::cout << __FILE__<<" " <<__FUNCTION__<<" Unknown type "<<std::endl;
	}
	for (int i = 0; i < channels; i++)
	{
		memcpy(chs[i].data, data.get()+i*rows*cols,  dep*rows*cols);
		chs[i] = chs[i].t();
	}

	merge(chs, img);
}

template<class T>
void WWMatrix<T>::copytoMat(cv::Mat& img, int c)
{
	using namespace cv;

	int dep = sizeof(data[0]);
	switch (dep)
	{
	case 1:		//unsigned char
		for (int i = 0; i < channels; i++)
		{
			Mat tmp(cols, rows, CV_8UC1);
			tmp.copyTo(img);
		}
		break;
	case 4:		//float
		for (int i = 0; i < channels; i++)
		{
			Mat tmp(cols, rows, CV_32FC1);
			tmp.copyTo(img);
		}
		break;
	case 8:		//double
		for (int i = 0; i < channels; i++)
		{
			Mat tmp(cols, rows, CV_64FC1);
			tmp.copyTo(img);
		}
		break;
	default:
		std::cout << __FILE__<<" " <<__FUNCTION__<<" Unknown type "<<std::endl;
	}
	memcpy(img.data, data.get()+c*rows*cols,  dep*rows*cols);
	img = img.t();
}

template<class T>
T & WWMatrix<T>::operator() (int x)
{
	if (x < 0)
		return data[0];
	if (x >= channels*rows*cols)
		return data[channels*rows*cols-1];
	return data[x];
}

template<class T>
T & WWMatrix<T>::operator() (int x, int y)
{
	if (y < 0)
		y = 0;
	else if (y >= rows)
		y = rows - 1;
	if (x < 0)
		x = 0; 
	else if (x >= cols)
		x = cols - 1;
	return data[x*rows+y];
}

template<class T>
T & WWMatrix<T>::operator() (int x, int y, int z)
{
	if (y < 0)
		y = 0;
	else if (y >= rows)
		y = rows - 1;
	if (x < 0)
		x = 0; 
	else if (x >= cols)
		x = cols - 1;
	if (z < 0)
		z = 0; 
	else if (z >= channels)
		z = channels - 1;
	return data[x*rows+y+z*rows*cols];
}

template<class T>
void WWMatrix<T>::copyTo(WWMatrix<T>& wwm)
{
	wwm.release();
    wwm = WWMatrix<T>(rows, cols, channels);
	memcpy(wwm.data.get(), data.get(), sizeof(data[0])*rows*cols*channels);
}

template<class T>
void WWMatrix<T>::release()
{
	rows = 0;cols = 0; channels = 0;
	data.reset();
}

#endif
