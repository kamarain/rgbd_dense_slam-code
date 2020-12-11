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

#if !defined(__MATRIX_HPP__)
#define __MATRIX_HPP__

#include "commonmath.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "Vector.hpp"
class Quaternion;

class Matrix {
private:
    float *cell;
    int  m_rows;
    int  m_cols;
public:
    Matrix(int numRows = 4, int numCols = 4);
    Matrix(const Matrix &m);
    ~Matrix();
    
    float operator() (int row, int col) const;
    void set(int row, int col, float value);
    void identity();
    void zero();
    void transpose();
    const Matrix & operator = (const Matrix &m);
    inline const Matrix & operator += (const Matrix &m);
    inline const Matrix & operator -= (const Matrix &m);
    const Matrix & operator *= (const Matrix &m);
        inline const Matrix & operator *= (const float s);
    inline const Matrix & operator /= (const float s);
    inline int rows() const;
    inline int cols() const;
    int getQuaternion(Quaternion &q) const;
    float trace() const;
    void loadQuaternion(const Quaternion &q);
    float *getData() { return cell;}
    void resize(unsigned int w, unsigned int h);
    void dump();
};

const Matrix& operator * (const Matrix &m1, const Matrix &m2);
const Vector operator * (const Vector &v, const Matrix &m);

inline int Matrix::rows() const
{
    return m_rows;
}

inline int Matrix::cols() const
{
    return m_cols;
}

inline const Matrix& Matrix::operator += (const Matrix &m)
{
    assert(m.rows() == m_rows && m.cols() == m_cols);
    Matrix &t = *this;
    for (int j = 0; j < m_cols; j++)
        for (int i = 0; i < m_rows; i++)
            set(i,j,t(i,j)+m(i,j));
    return *this;
}

inline const Matrix& Matrix::operator -= (const Matrix &m)
{
    assert(m.rows() == m_rows && m.cols() == m_cols);
    Matrix &t = *this;
    for (int j = 0; j < m_cols; j++)
        for (int i = 0; i < m_rows; i++)
            set(i,j,t(i,j)-m(i,j));
    return *this;
}

inline const Matrix& Matrix::operator *= (const float s)
{
    Matrix &t = *this;
    for (int i = 0; i < m_rows; i++)
        for (int j = 0; j < m_cols; j++)
            set(i,j,t(i,j)*s);
    return *this;
}

inline const Matrix& Matrix::operator /= (const float s)
{
    assert(s != 0);
    Matrix &t = *this;
    for (int i = 0; i < m_rows; i++)
        for (int j = 0; j < m_cols; j++)
            set(i,j,t(i,j)/s);
    return *this;
}

#endif //!__MATRIX_HPP__
