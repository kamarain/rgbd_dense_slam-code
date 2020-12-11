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

#if !defined(__QUATERNION_HPP__)
#define __QUATERNION_HPP__

#include "commonmath.h"

class Matrix;

class Quaternion {
public:
    Quaternion();
        Quaternion(float w, float x, float y, float z);
    Quaternion(const Quaternion &q);

    float w, x, y, z;

    const Quaternion & operator = (const Quaternion &q);
    inline const Quaternion & operator += (const Quaternion &q);
    inline const Quaternion & operator -= (const Quaternion &q);
    inline const Quaternion & operator *= (const float s);
    inline const Quaternion & operator /= (const float s);
    inline const float length() const;
    inline const float lengthSquared() const;
        void loadAxisAngle(const float ang, const float x, const float y, const float z);
    void setLength(float len);
    void normalize();
    void set(const float ww, const float xx, const float yy,const float zz);
	Quaternion conjugate();
    int getMatrix(Matrix &m);
};



inline const Quaternion &Quaternion::operator += (const Quaternion &q)
{
    w += q.w;
    x += q.x;
    y += q.y;
    z += q.z;

    return *this;
}

inline const Quaternion &Quaternion::operator -= (const Quaternion &q)
{
    w -= q.w;
    x -= q.x;
    y -= q.y;
    z -= q.z;

    return *this;
}

inline const Quaternion &Quaternion::operator *= (const float s)
{
    w *= s;
    x *= s;
    y *= s;
    z *= s;

    return *this;
}

inline const Quaternion& Quaternion::operator /= (const float s)
{
    w /= s;
    x /= s;
    y /= s;
    z /= s;
    return *this;
}

inline const float Quaternion::length() const
{
   return (float)sqrt(w*w + x*x + y*y + z*z);
}

inline const float Quaternion::lengthSquared() const
{
    return (float)(w*w + x*x + y*y + z*z);
}

inline Quaternion operator + (const Quaternion &v, const Quaternion &w)
{
    Quaternion tmp(v.w+w.w, v.x + w.x, v.y + w.y, v.z + w.z);
    return tmp;
}

inline Quaternion operator - (const Quaternion &v, const Quaternion &w)
{
    Quaternion tmp(v.w-w.w, v.x - w.x, v.y - w.y, v.z - w.z);
    return tmp;
}

inline Quaternion operator * (const Quaternion &v, const float s)
{
    Quaternion tmp(v.w * s, v.x * s, v.y * s, v.z * s);
    return tmp;
}

inline Quaternion operator / (const Quaternion &v, const float s)
{
    Quaternion tmp(v.w / s, v.x / s, v.y / s, v.z / s);
    return tmp;
}

inline Quaternion operator * (const Quaternion &p, const Quaternion &q)
{
    Quaternion tmp(p.w*q.w-(p.x*q.x + p.y*q.y + p.z*q.z),
                   p.y*q.z-p.z*q.y + p.w*q.x + q.w*p.x,
                   p.z*q.x-p.x*q.z + p.w*q.y + q.w*p.y,
                   p.x*q.y-p.y*q.x + p.w*q.z + q.w*p.z);
    return tmp;
}

inline Quaternion operator * (const float s, const Quaternion &q) {
    Quaternion tmp(q.w * s, q.x * s, q.y * s, q.z * s);
    return tmp;
}

#endif //!__QUATERNION_HPP__
