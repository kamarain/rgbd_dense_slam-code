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

#pragma once

#include "commonmath.h"
#include <math.h>

class Vector {
public:
    Vector();
    Vector(float x, float y, float z);
    Vector(const Vector &v);

    float x, y, z;

    const Vector & operator = (const Vector &v);
    const Vector & operator += (const Vector &v);
    const Vector & operator -= (const Vector &v);
    const Vector & operator *= (const float s);
    const Vector & operator /= (const float s);
    const float length() const;
    const float lengthSquared() const;
    void setLength(float len);
    void normalize();
    void cross(const Vector &v1, const Vector &v2);
    void set(float x, float y, float z);
};
/*
Vector operator + (const Vector &v, const Vector &w)
{
    Vector tmp(v.x + w.x, v.y + w.y, v.z + w.z);
    return tmp;
}

Vector operator - (const Vector &v, const Vector &w)
{
    Vector tmp(v.x - w.x, v.y - w.y, v.z - w.z);
    return tmp;
}

Vector operator * (const Vector &v, const float s)
{
    Vector tmp(v.x * s, v.y * s, v.z * s);
    return tmp;
}

Vector operator / (const Vector &v, const float s)
{
    Vector tmp(v.x / s, v.y / s, v.z / s);
    return tmp;
}

float operator * (const Vector &v, const Vector &w)
{
    return v.x * w.x + v.y * w.y + v.z * w.z;
}

Vector operator * (const float s, const Vector &v)
{
    Vector tmp(v.x * s, v.y * s, v.z * s);
    return tmp;
}
*/
