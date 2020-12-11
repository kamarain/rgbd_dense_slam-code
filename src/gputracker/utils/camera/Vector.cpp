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
#include "Vector.hpp"
#include "commonmath.h"

const Vector &Vector::operator += (const Vector &v)
{
    x += v.x;
    y += v.y;
    z += v.z;

    return *this;
}

const Vector &Vector::operator -= (const Vector &v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;

    return *this;
}

const Vector &Vector::operator *= (const float s)
{
    x *= s;
    y *= s;
    z *= s;

    return *this;
}

const Vector& Vector::operator /= (const float s)
{
    x /= s;
    y /= s;
    z /= s;
    return *this;
}

const float Vector::length() const
{
   return (float)sqrt(x*x + y*y + z*z);
}

const float Vector::lengthSquared() const
{
    return (float)(x*x + y*y + z*z);
}

Vector::Vector() : x(0), y(0), z(0)
{
}

Vector::Vector(float x, float y, float z) : x(x), y(y), z(z)
{
}

Vector::Vector(const Vector &v)
{
    x = v.x;
    y = v.y;
    z = v.z;
}

const Vector &Vector::operator = (const Vector &v)
{
    x = v.x;
    y = v.y;
    z = v.z;
    return *this;
}

void Vector::set(float xx, float yy, float zz)
{
    x = xx;
    y = yy;
    z = zz;
}


void Vector::normalize()
{
    float len = length();
    x /= len;
    y /= len;
    z /= len;
    return;
}

void Vector::setLength(float newLen)
{
    normalize();
    x *= newLen;
    y *= newLen;
    z *= newLen;
    return;
}

void Vector::cross(const Vector &v1, const Vector &v2)
{
    x = (v1.y * v2.z) - (v1.z * v2.y);
    y = (v1.z * v2.x) - (v1.x * v2.z);
    z = (v1.x * v2.y) - (v1.y * v2.x);
}
