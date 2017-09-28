/// @file timer.h
/// @brief 可以像matlab的tic toc那样使用的计时工具
/// @version 8.0
/// @date 2017-9-27

#ifndef TIMER_H
#define TIMER_H

#ifdef __linux__
#include <sys/time.h>
#else
#include <time.h>
#endif
#include <stdio.h>

double getTime(){
#ifdef __linux__
	timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + tv.tv_usec * 1e-6;
#else
	return 1.0 * clock() / CLOCKS_PER_SEC;
#endif
}

#define tic tic_f();
#define toc toc_f();
double _lastRdtsc = 0;

inline void tic_f(){
	_lastRdtsc = getTime();
}

inline double toc_f(){
	double t = getTime();
	printf("run time: %f\n", t - _lastRdtsc);
	return t - _lastRdtsc;
}
#endif