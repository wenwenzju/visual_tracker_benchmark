/// @file convConst.h
/// @brief ≤Œ’’Piotr's Computer Vision Matlab Toolbox Õ¯÷∑£∫https://pdollar.github.io/toolbox/
/// @date 2017-9-27

#ifndef __CONVCONST__
#define __CONVCONST__

void convBoxY( float *I, float *O, int h, int r, int s );
void convBox( float *I, float *O, int h, int w, int d, int r, int s );
void conv11Y( float *I, float *O, int h, int side, int s );
void conv11( float *I, float *O, int h, int w, int d, int side, int s );
void convTriY( float *I, float *O, int h, int r, int s );
void convtri( float *I, float *O, int h, int w, int d, int r, int s );
void convTri1Y( float *I, float *O, int h, float p, int s );
void convTri1( float *I, float *O, int h, int w, int d, float p, int s );
void convMaxY( float *I, float *O, float *T, int h, int r );
void convMax( float *I, float *O, int h, int w, int d, int r );

#endif