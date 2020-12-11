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

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "Matrix.hpp"
#include "Quaternion.hpp"
#include "Vector.hpp"

namespace matrix_globals
{
        float global_temp[4][4];
}
using namespace matrix_globals;

Matrix::Matrix(int numRows, int numCols) 
{ 
    //printf("creating %dx%d matrix...\n",numRows,numCols);
    m_rows = numRows;
    m_cols = numCols;
    cell = new float[m_rows*m_cols];
    if (m_rows == m_cols) identity(); 
    else zero();
}

Matrix::~Matrix()
{
    delete[] cell;
}

Matrix::Matrix(const Matrix &m) 
{
    m_rows = m.rows();
    m_cols = m.cols();
    cell = new float[m_rows*m_cols];
    for (int j=0; j < m_rows; j++)
        for (int i=0; i < m_cols; i++)
            set(i,j,m(i,j));    
}

const Matrix& Matrix::operator = (const Matrix &m)
{
    //printf("assigning matrix %dx%d to %dx%d.\n",m.rows(),m.cols(),m_rows,m_cols);
    if (m_rows != m.rows() || m_cols != m.cols())
	{
		m_rows = m.rows();
		m_cols = m.cols();
		delete[] cell;
                cell = new float[m_rows*m_cols];
	}
    for (int i=0; i < m_rows; i++)
        for (int j=0; j < m_cols; j++)
        {
            set(i,j,m(i,j));    
        }
    return *this;
}

float Matrix::operator() (int row, int col) const
{
    //printf("row:%d, col%d, m_rows:%d, m_cols:%d\n",row,col,m_rows,m_cols);
    assert((row >= 0) && (row < m_rows));
    assert((col >= 0) && (col < m_cols));
    return cell[row*m_cols+col];
}

void Matrix::set(int row, int col, float value)
{
    assert(row >= 0 && row < m_rows);
    assert(col >= 0 && col < m_cols);
    cell[row*m_cols+col] = value;
}

void Matrix::transpose()
{
    Matrix t = *this;
    assert(m_rows == m_cols);
    for (int j = 0; j < m_rows; j++)
        for (int i = 0; i < m_cols; i++)
            (*this).set(i,j,t(j,i));
    return;

}
     
void Matrix::identity()
{
    assert(m_rows == m_cols);
    memset(cell,0,sizeof(float)*m_rows*m_cols);
    for (int i = 0; i < m_rows; i++) set(i,i,1);
}

void Matrix::zero()
{
    memset(cell,0,sizeof(float)*m_rows*m_cols);
}

void Matrix::resize(unsigned int w, unsigned int h)
{
	if (cell != NULL) delete[] cell;		
	m_rows = h;
	m_cols = w;
        cell = new float[m_rows*m_cols];
	zero();
}


float Matrix::trace() const
{
    int i;
    const Matrix &t = *this;
    float trace = 0;
    for (i = 0; i < cols(); i++) trace += t(i,i);
    return trace;
}

int Matrix::getQuaternion(Quaternion &q) const
{
    if (rows() < 3 || cols() < 3) return 0;
    const Matrix &t = *this;
    float tr = t(0,0)+t(1,1)+t(2,2)+1;
    if (tr > 0)
    {
        float s = 0.5f / sqrt(tr);
        q.w = 0.25f / s;
        q.x = (t(2,1) - t(1,2))*s;
        q.y = (t(0,2) - t(2,0))*s;
        q.z = (t(1,0) - t(0,1))*s;
        return 1;
    }
    if (t(0,0) > t(1,1) && t(0,0) > t(2,2))
    {
        float s = 2.0f * sqrt(1.0f+t(0,0)-t(1,1)-t(2,2));
        q.x = 0.5f / s;
        q.y = (t(0,1)+t(1,0)) / s;
        q.z = (t(0,2)+t(2,0)) / s;
        q.w = (t(1,2)+t(2,1)) / s;
    } else
    if (t(1,1) > t(0,0) && t(1,1) > t(2,2))
    {
        float s = 2.0f * sqrt(1.0f + t(1,1) - t(0,0) - t(2,2));
        q.x = (t(0,1) + t(1,0)) / s;
        q.y = 0.5f / s;
        q.z = (t(1,2)+t(2,1)) / s;
        q.w = (t(0,2)+t(2,0)) / s;
    } else {
        float s = 2.0f * sqrt(1.0f + t(2,2) - t(0,0) - t(1,1));
        q.x = (t(0,2)+t(2,0)) / s;
        q.y = (t(1,2)+t(2,1)) / s;
        q.z = 0.5f / s;
        q.w = (t(0,1)+t(1,0)) / s;
    }
    return 1;
}

void Matrix::loadQuaternion(const Quaternion &q)
{
    if (rows() < 3 || cols() < 3) return;
	
	Matrix &rs = *this;

	identity();

    float wx, wy, wz, xx, yy, yz, xy, xz, zz, x2, y2, z2;

    x2 = q.x + q.x; y2 = q.y + q.y; z2 = q.z + q.z;
    xx = q.x * x2;   xy = q.x * y2;   xz = q.x * z2;
    yy = q.y * y2;   yz = q.y * z2;   zz = q.z * z2;
    wx = q.w * x2;   wy = q.w * y2;   wz = q.w * z2;

    rs.set(0,0,1.0f - (yy + zz));
	rs.set(0,1,xy - wz);
    rs.set(0,2,xz + wy);
    
    rs.set(1,0,xy + wz);
    rs.set(1,1,1.0f - (xx + zz));
    rs.set(1,2,yz - wx);
    
    rs.set(2,0,xz - wy);
    rs.set(2,1,yz + wx);
    rs.set(2,2,1.0f - (xx + yy));
}

const Matrix& operator * (const Matrix &m1, const Matrix &m2)
{
    assert(m1.cols() == m2.rows());
    //printf("mulling...\n");
 
    Matrix *r = new Matrix(m1.rows(),m2.cols());
    //Matrix r(m1.rows(),m2.cols());
    r->zero();

    int j=0;
    while(j < m2.cols())
    {
        int i=0;
        while(i < m1.rows())
        {
            float value = 0;
            for (int x = 0; x < m1.cols(); x++) value += m1(i,x)*m2(x,j);
            r->set(i,j, value);
            i++;
        }
        j++;
    }
    return *r;
}

const Vector operator * (const Vector &v, const Matrix &m)
{
    assert(m.rows() == 3 || m.rows() == 4); 
    Vector rvec(0,0,0);
    if (m.rows() == 3)
    {
        rvec.x = v.x*m(0,0) + v.y*m(1,0) + v.z*m(2,0);
        rvec.y = v.x*m(0,1) + v.y*m(1,1) + v.z*m(2,1);
        rvec.z = v.x*m(0,2) + v.y*m(1,2) + v.z*m(2,2);
       
    } else {
        rvec.x = v.x*m(0,0) + v.y*m(1,0) + v.z*m(2,0) + m(3,0);
        rvec.y = v.x*m(0,1) + v.y*m(1,1) + v.z*m(2,1) + m(3,1);
        rvec.z = v.x*m(0,2) + v.y*m(1,2) + v.z*m(2,2) + m(3,2);
    }
    return rvec;
}



const Matrix& Matrix::operator *= (const Matrix &m)
{
	assert(m_cols == m.rows());

	// slow method for non 4x4 matrices
	if (m_cols != 4 || m.cols() != 4  || m_rows != 4 || m.rows() != 4)
	{	
		m_cols = m.cols();
		delete[] cell;
                cell = new float[m_rows*m_cols];
		Matrix t = *this;
		int j=0;
		while(j < m_cols)
		{	
			int i=0;
			while(i < m_rows)
			{
                                float value = 0;
				for (int x = 0; x < m_cols; x++) value += t(i,x)*m(x,j);
				set(i,j, value);
				i++;
			}
			j++;
		}
	} else 
	{
		// faster replacement:
                memcpy(global_temp,this->cell,4*4*sizeof(float));
		int j=0;
		while(j < m_cols)
		{	
			int i=0;
			while(i < m_rows)
			{
                                float value = 0;
				for (int x = 0; x < m_cols; x++) value += global_temp[i][x]*m(x,j);
				set(i,j, value);
				i++;
			}
			j++;
		}
	}
	return *this;
}

void Matrix::dump()
{
	const Matrix &t = *this;
	printf("matrix dump:\n");
	for (int j=0; j < m_rows; j++)
	{
        for (int i=0; i < m_cols; i++)
			printf("%f ",t(j,i));
		printf("\n");
	}
}
