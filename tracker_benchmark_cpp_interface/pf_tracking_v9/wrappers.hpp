/// @file wrappers.hpp
/// @brief ≤Œ’’Piotr's Computer Vision Matlab Toolbox Õ¯÷∑£∫https://pdollar.github.io/toolbox/
/// @date 2017-9-27

/*******************************************************************************
* Piotr's Computer Vision Matlab Toolbox      Version 3.00
* Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
* Licensed under the Simplified BSD License [see external/bsd.txt]
*******************************************************************************/
#ifndef _WRAPPERS_HPP_
#define _WRAPPERS_HPP_
#include <malloc.h>
#ifdef MATLAB_MEX_FILE

// wrapper functions if compiling from Matlab
#include "mex.h"
inline void wrError(const char *errormsg) { mexErrMsgTxt(errormsg); }
inline void* wrCalloc( size_t num, size_t size ) { return mxCalloc(num,size); }
inline void* wrMalloc( size_t size ) { return mxMalloc(size); }
inline void wrFree( void * ptr ) { mxFree(ptr); }

#else

// wrapper functions if compiling from C/C++
inline void wrError(const char *errormsg) { throw errormsg; }
inline void* wrCalloc( size_t num, size_t size ) { return calloc(num,size); }
inline void* wrMalloc( size_t size ) { return malloc(size); }
inline void wrFree( void * ptr ) { free(ptr); }

#endif

// platform independent aligned memory allocation (see also alFree)
void* alMalloc( size_t size, int alignment ) ;

// platform independent alignned memory de-allocation (see also alMalloc)
void alFree(void* aligned) ;

#endif
