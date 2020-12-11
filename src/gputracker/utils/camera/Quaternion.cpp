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

#include <math.h>
#include <assert.h>
#include "Matrix.hpp"
#include "Quaternion.hpp"

Quaternion::Quaternion()
{
	w = 0;
	x = 0;
	y = 0; 
	z = 0;
}

Quaternion::Quaternion(float w, float x, float y, float z) : w(w), x(x), y(y), z(z)
{
}

Quaternion::Quaternion(const Quaternion &q)
{
    w = q.w;
    x = q.x;
    y = q.y;
    z = q.z;
}

const Quaternion &Quaternion::operator = (const Quaternion &q)
{
    w = q.w;
    x = q.x;
    y = q.y;
    z = q.z;
    return *this;
}

void Quaternion::set(const float ww, const float xx, const float yy,const float zz)
{
    w = ww;
    x = xx;
    y = yy;
    z = zz;
}

void Quaternion::normalize()
{
    float len = length();
    w /= len;
    x /= len;
    y /= len;
    z /= len;
    return;
}

void Quaternion::loadAxisAngle(const float ang, const float x, const float y, const float z)
{
	set(cos(ang/2),sin(ang/2)*x,sin(ang/2)*y,sin(ang/2)*z);
}

void Quaternion::setLength(float newLen)
{
    normalize();
    w *= newLen;
    x *= newLen;
    y *= newLen;
    z *= newLen;
    return;
}

int Quaternion::getMatrix(Matrix &m)
{
   if (m.rows() < 3 || m.cols() < 3) return 0;
   m.set(0,0,w*w+x*x-y*y-z*z);
   m.set(0,1,2*(x*y-w*z));
   m.set(0,2,2*(x*z+w*y));
   
   m.set(1,0,2*(x*y+w*z));
   m.set(1,1,w*w-x*x+y*y-z*z);
   m.set(1,2,2*(y*z-w*x));

   m.set(2,0,2*(x*z-w*y));
   m.set(2,1,2*(y*z+w*x));
   m.set(2,2,w*w-x*x-y*y+z*z);
   return 1;
}

Quaternion Quaternion::conjugate()
{
	Quaternion tmp(w,-x,-y,-z);
	return tmp;
}
