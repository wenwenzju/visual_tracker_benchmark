/// @file gradient.h
/// @brief ≤Œ’’Piotr's Computer Vision Matlab Toolbox Õ¯÷∑£∫https://pdollar.github.io/toolbox/
/// @date 2017-9-27

#ifndef __GRADIENT__
#define __GRADIENT__

void gradMagNorm( float *M, float *S, int h, int w, float norm );
void gradMag( float *I, float *M, float *O, int h, int w, int d, bool full );
void gradHist( float *M, float *O, float *H, int h, int w,
	int bin, int nOrients, int softBin, bool full );

#endif