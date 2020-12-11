/*
rgbd-tracker
Copyright (c) 2014, Tommi Tykkälä, All rights reserved.

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3.0 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library.
*/

#if !defined(__COMMONMATH_H__)
#define __COMMONMATH_H__

#include <float.h>

#if !defined(USE_DOUBLE)
#  define BIG_REAL FLT_MAX
#  define SMALL_REAL FLT_MIN
#else
#  define BIG_REAL DBL_MAX
#  define SMALL_REAL DBL_MIN
#endif

#if !defined(mPI)
#define mPI 3.141592653589793f
#endif

// conversion functions
inline float deg2rad(float deg)
{
    return deg * ((2.0f * mPI) / 360.0f);
}

inline float rad2deg(float rad)
{
    return rad * (360.0f / (2 * mPI));
}

#endif //!__COMMONMATH_H__
