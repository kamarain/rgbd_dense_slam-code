/*
stereo-gen
Copyright (c) 2014, Tommi Tykkälä, All rights reserved.

This source code is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3.0 of the License, or (at your option) any later version.

This source code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this source code.
*/

#ifndef _PERFORMANCECOUNTER_H_
#define _PERFORMANCECOUNTER_H_

/**************** LINUX/MAC OS X COUNTER *******************/

#if (defined __unix__) || (defined __APPLE__)

#include "stdlib.h"
#include "sys/time.h"

class PerformanceCounter
{
  protected:

  long startCountSec,stopCountSec,startCountMicroSec,stopCountMicroSec;

  public:

  PerformanceCounter()
  {
    // also, reset the starting counter
    StartCounter();
  }

  void StartCounter(); // call this before your code block
  void StopCounter(); // call this after your code block

  // read elapsed time (units are seconds, accuracy is up to microseconds)
  double GetElapsedTime();

};

inline void PerformanceCounter::StartCounter()
{
  struct timeval tv;

  gettimeofday(&tv,NULL);

  startCountSec = tv.tv_sec;
  startCountMicroSec = tv.tv_usec;

}

inline void PerformanceCounter::StopCounter()
{
  struct timeval tv;

  gettimeofday(&tv,NULL);

  stopCountSec = tv.tv_sec;
  stopCountMicroSec = tv.tv_usec;
}


inline double PerformanceCounter::GetElapsedTime()
{
  float elapsedTime = 1.0 * (stopCountSec-startCountSec) + 1E-6 * (stopCountMicroSec - startCountMicroSec);
  return elapsedTime;
}

#endif

#ifdef WIN32

/**************** WINDOWS COUNTER *******************/

#include <windows.h>

class PerformanceCounter
{
  protected:

  LARGE_INTEGER timerFrequency;
  LARGE_INTEGER startCount,stopCount;

  public:

  PerformanceCounter() 
  {
    // reset the counter frequency
    QueryPerformanceFrequency(&timerFrequency);
    // also, reset the starting counter
    StartCounter();
  }

  void StartCounter(); // call this before your code block
  void StopCounter(); // call this after your code block
  unsigned int getTicks(); // return tick count


  // read elapsed time (units are seconds, accuracy is up to microseconds)
  double GetElapsedTime();
};

inline void PerformanceCounter::StartCounter()
{
  QueryPerformanceCounter(&startCount);
}

inline void PerformanceCounter::StopCounter()
{
  QueryPerformanceCounter(&stopCount);
}

inline unsigned int PerformanceCounter::getTicks()
{
	LARGE_INTEGER ticks;
	QueryPerformanceCounter(&ticks);
	unsigned int ticksInt = ticks.LowPart;
	return ticksInt;
}


inline double PerformanceCounter::GetElapsedTime()
{
  return ((double)(stopCount.QuadPart - startCount.QuadPart))
    / ((double)timerFrequency.QuadPart);
}

#endif
#endif

