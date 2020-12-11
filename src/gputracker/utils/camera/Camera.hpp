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
#include "Vector.hpp"
#include "Matrix.hpp"

namespace customCameraTools {

class Plane {
public:
    Vector normal;
    float d;
    int indexlist[6];
};

class Frustum {
public:
    Frustum() {
        for (int i = 0; i < 6; i++) {
            planes[i].normal.set(0,0,0);
            planes[i].d = 0;
            for (int j = 0; j < 6; j++)
                planes[i].indexlist[j] = 0;
        }
    }
    Plane planes[6];
};

class Camera
{
protected:
    Vector pos;
    Frustum frustum;
    Matrix rot;
    Matrix matrix;
    Matrix projection;
//    Outreach::Vector dir;
//    Outreach::Quaternion q;
    float fovx, fovy;
    float near_dist, far_dist;

    enum
    {
        LEFT_PLANE,
        RIGHT_PLANE,
        TOP_PLANE,
        BOTTOM_PLANE,
        FAR_PLANE,
        NEAR_PLANE
    };

public:
    Camera();

    void setPerspective(float fov, float aspectratio, float znear, float zfar);
        void setOrtho(float left, float right, float bottom, float top, float znear, float zfar);
    void setPosition(const Vector &pos);
    void setPosition(float x, float y, float z);
    void getPosition(Vector &pos);
    void getPosition(float *x, float *y, float *z);
    void setRotationMatrix(const Matrix &m);
	void getCameraRotationMatrix(Matrix &crot);
        void setDirection(const float dx, const float dy, const float dz, const float ux, const float uy, const float uz);
	void setDirection(const Vector &d, const Vector &u);
	void getDirection(Vector &v) const;
    void getFOV(float *fovx, float *fovy);
   void activate();
	void getCameraMatrix(Matrix &m);
    int projectVertex(float x, float y, float z, float *px, float *py);
    void getCameraSpaceVertex(float x, float y, float z, float *cx, float *cy, float *cz);
    void setupFrustum();
    int boxVisible(float *bbox);
    int vertexVisible(float x, float y, float z);
	float getZNear();
	float *getProjectionMatrix();
	float *getModelViewMatrix();
};
}
