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
#include <math.h>
#include "commonmath.h"
#include "Camera.hpp"
#include "Matrix.hpp"

using namespace customCameraTools;

Camera::Camera()
{
	near_dist = 1;
    far_dist = 1000;
    pos.set(0,0,0);
 //   rot.resize(3,3);
	rot.identity();
    matrix.identity();
    projection.identity();
    setupFrustum();
    setPerspective(45.0f, 1, 1, 10000.0f);
//    q.w = 1; q.x = 0; q.y=0; q.z = 0;
//    dir.set(0,0,-1);

    fovx = 0;
    fovy = 0;
}

float *Camera::getProjectionMatrix()
{
	return projection.getData();
}

float *Camera::getModelViewMatrix()
{
	return matrix.getData();
}


void Camera::setOrtho(float left, float right, float bottom, float top, float znear, float zfar)
{
	near_dist = znear;
	far_dist = zfar;
	fovx = 0;
	fovy = 0;

	Matrix T;
	T.set(3,0,-(right+left)/(right-left));
	T.set(3,1,-(top+bottom)/(top-bottom));
	T.set(3,2,-(zfar+znear)/(zfar-znear));
	Matrix S;
	S.set(0,0,2.0f/(right-left));
	S.set(1,1,2.0f/(top-bottom));
	S.set(2,2,2.0f/(znear-zfar));
	Matrix Morth;
	Morth.set(2,2,0);
	//Morth.set(2,2,-0.0002);
	
	projection.identity();
	//projection *= Morth;
	projection *= S;
	projection *= T;
	projection.dump();
}

void Camera::setPerspective(float fov, float aspectratio, float znear, float zfar)
{
    float r, l, t, b;

	near_dist = znear;
    far_dist = zfar;
    fovx = deg2rad(fov) / 2;
    fovy = fovx;//atan(tan(fovx) / aspectratio);

    r = znear * tan(fovx);
    l = -r;

    t = znear * tan(fovx)/aspectratio;///aspectratio;
    b = -t;

    projection.set(0,0,(2 * znear) / (r - l));
    projection.set(0,1,0);
    projection.set(0,2,0);
    projection.set(0,3,0);

    projection.set(1,0,0);
    projection.set(1,1,(2 * znear) / (t - b));
    projection.set(1,2,0);
    projection.set(1,3,0);

    projection.set(2,0,(r + l) / (r - l));
    projection.set(2,1,(t + b) / (t - b));
    projection.set(2,2,-(zfar + znear) / (zfar - znear));
    projection.set(2,3,-1);

    projection.set(3,0,0);
    projection.set(3,1,0);
    projection.set(3,2,-(2 * zfar * znear) / (zfar - znear));
    projection.set(3,3,0);
}



void Camera::getCameraSpaceVertex(float x, float y, float z, float *cx, float *cy, float *cz)
{
    float w = 1.0f;

    *cx = x*matrix(0,0)+y*matrix(1,0)+z*matrix(2,0)+w*matrix(3,0);
    *cy = x*matrix(0,1)+y*matrix(1,1)+z*matrix(2,1)+w*matrix(3,1);
    *cz = x*matrix(0,2)+y*matrix(1,2)+z*matrix(2,2)+w*matrix(3,2);
}

int Camera::projectVertex(float x, float y, float z, float *px, float *py)
{
    float cx,cy,cz;
    float pw;
    float w=1;

	cx = x*matrix(0,0)+y*matrix(1,0)+z*matrix(2,0)+w*matrix(3,0);
    cy = x*matrix(0,1)+y*matrix(1,1)+z*matrix(2,1)+w*matrix(3,1);
    cz = x*matrix(0,2)+y*matrix(1,2)+z*matrix(2,2)+w*matrix(3,2);

	if (cz >= -near_dist || cz < -far_dist) { *px = 0; *py = 0; return 0; }

    *px = cx*projection(0,0)+cy*projection(1,0)+cz*projection(2,0)+w*projection(3,0);
    *py = cx*projection(0,1)+cy*projection(1,1)+cz*projection(2,1)+w*projection(3,1);
    pw = cx*projection(0,3)+cy*projection(1,3)+cz*projection(2,3)+w*projection(3,3);

    if (pw < 0.0001) pw = float(0.0001);
    *px = *px/pw;
    *py = *py/pw;

    return 1;
}

void Camera::setPosition(const Vector &pos)
{
    this->pos = pos;
}

void Camera::getPosition(Vector &pos)
{
    //pos = this->pos;
	pos.set(this->pos.x,this->pos.y,this->pos.z);
}

void Camera::setPosition(float x, float y, float z)
{
    pos.set(x,y,z);
}

void Camera::getPosition(float *x, float *y, float *z)
{
    *x = pos.x; *y = pos.y; *z = pos.z;
}

void Camera::getDirection(Vector &v) const
{
	v.set(-rot(0,2),-rot(1,2),-rot(2,2));
}

void Camera::setRotationMatrix(const Matrix &m)
{
    rot = m;
}


void Camera::setupFrustum()
{
/*    int     i,j;
    float   c,s;
    Plane  *cPlane;
    float   *cNormal;
    Matrix  m;
    Vector dir;

    dir.set(-rot(0,2),-rot(1,2),-rot(2,2));

    // normals towards inner frustum!!!
    m = rot;

    m.transpose();

    c = cos(fovx);
    s = sin(fovx);

    Vector  &n1 = frustum.planes[LEFT_PLANE].normal;
    n1.x = c; n1.y = 0; n1.z = -s;
        n1 = n1 * m;
    //n->rotate(m);
    frustum.planes[LEFT_PLANE].d = n1 * pos;

    //right plane normal
    Vector &n2 = frustum.planes[RIGHT_PLANE].normal;
    n2.x =-c; n2.y = 0; n2.z = -s;
    n2 = n2 * m;
    //n->rotate(m);
    frustum.planes[RIGHT_PLANE].d = n2 * pos;
    //printf("right plane normal:%f %f %f\n",n->x,n->y,n->z);


    c = cos(fovy);
    s = sin(fovy);

    //top plane normal
    Vector &n3 = frustum.planes[TOP_PLANE].normal;
    n3.x = 0; n3.y = -c; n3.z = -s;
    n3 = n3 * m;
    //	n->rotate(m);
    frustum.planes[TOP_PLANE].d = n3 * pos;
    //  printf("top plane normal:%f %f %f\n",n->x,n->y,n->z);


    //bottom plane normal
    Vector &n4 = frustum.planes[BOTTOM_PLANE].normal;
    n4.x = 0; n4.y = c; n4.z = -s;
    //n->rotate(m);
        n4 = n4 * m;
    frustum.planes[BOTTOM_PLANE].d = n4 * pos;
    //	printf("bottom plane normal:%f %f %f\n",n->x,n->y,n->z);

    //near plane normal
    Vector &n5 = frustum.planes[NEAR_PLANE].normal;
    n5.x = 0; n5.y = 0; n5.z = -1;
    //n->rotate(m);
        n5 = n5 * m;
    

    Vector nearpos;

    nearpos = pos + dir * near_dist;
    frustum.planes[NEAR_PLANE].d = n5 * nearpos;
    //    printf("near plane normal:%f %f %f\n",n->x,n->y,n->z);

    //far plane normal
    Vector &n6 = frustum.planes[FAR_PLANE].normal;
    n6.x = 0; n6.y = 0; n6.z = 1;
    n6 = n6 * m;
    Vector farpos;
    farpos = pos + dir * far_dist;
    frustum.planes[FAR_PLANE].d = n6 * farpos;
    //   printf("far plane normal:%f %f %f\n",n->x,n->y,n->z);

    for( j = 0; j < 6; j++ )
    {
        cPlane = &frustum.planes[j];
        cNormal = (float*) &cPlane->normal;

        for( i = 0; i < 3; i++ )
        {
            //  if (i < 2) {
            if ( cNormal[i] > 0)    //indexlist[0..2] = max, indexlist[3..5] = min
            {
                cPlane->indexlist[i] = i+3;   //farthest point.x = max.x
                cPlane->indexlist[i+3] = i;   //closest  point.x = min.x
            } else {
                cPlane->indexlist[i] = i;     //farthest point.x = min.x
                cPlane->indexlist[i+3] = i+3; //closest  point.x = max.x
            }
        }
    }*/
}

int Camera::boxVisible(float *bbox)
{
/*    int i;
    Vector c;
    Plane *cplane;

    cplane = &frustum.planes[0];

    for (i = 1; i < 5; i++) //no near & far
    {
        c.x = bbox[cplane->indexlist[0]];
        c.y = bbox[cplane->indexlist[1]];
        c.z = bbox[cplane->indexlist[2]];

        if ( cplane->normal * c  - cplane->d  <= 0.0f ) //if farthest vertex in bbox isn't visible, bbox out
            return 0; // totally outside

        cplane++;
    }
*/
    return 1;
}

int Camera::vertexVisible(float x, float y, float z)
{
/*    Vector c;
    c.set(x,y,z);
    Plane *cplane;
    cplane = &frustum.planes[0];

    for (int i = 0; i < 5; i++)
    {
        if (cplane->normal * c - cplane->d  <= 0.0f)
            return 0;
        cplane++;
    }*/
    return 1;
}

void Camera::activate()
{
    memcpy(matrix.getData(),rot.getData(),16*sizeof(float));

    matrix.set(3,0,-(pos.x*rot(0,0) + pos.y*rot(1,0)+pos.z*rot(2,0)));
    matrix.set(3,1,-(pos.x*rot(0,1) + pos.y*rot(1,1)+pos.z*rot(2,1)));
    matrix.set(3,2,-(pos.x*rot(0,2) + pos.y*rot(1,2)+pos.z*rot(2,2)));
	
//	setProjectionMatrix();
//	setModelViewMatrix();
	//printf("cam pos : %f %f %f\n",pos.x,pos.y,pos.z);
}

void Camera::setDirection(float dx, float dy, float dz, float ux, float uy, float uz)
{
    Vector x,y,z;
    Vector tmpy;

    z.x = -dx; z.y = -dy, z.z = -dz;
    tmpy.x=ux; tmpy.y=uy; tmpy.z=uz;

    x.cross(tmpy,z);
    x.normalize();
    y.cross(z,x);
    y.normalize();

    rot.set(0,0,x.x);
    rot.set(1,0,x.y);
    rot.set(2,0,x.z);
    rot.set(0,1,y.x);
    rot.set(1,1,y.y);
    rot.set(2,1,y.z);
    rot.set(0,2,z.x);
    rot.set(1,2,z.y);
    rot.set(2,2,z.z);

//     z' = project z to (0,0,-1)
//     normalize(z').z gives cos(theta)
//     normalize(z').(0,0,-1) gives cos(phi)

}  


void Camera::setDirection(const Vector &d, const Vector &u)
{
  /*  Vector x,y,z;
  
     z = -1.0f*d;

    x.cross(u,z);
    x.normalize();
    y.cross(d,x);
    y.normalize();

	rot.set(0,0,x.x);
    rot.set(1,0,x.y);
    rot.set(2,0,x.z);
    rot.set(0,1,y.x);
    rot.set(1,1,y.y);
    rot.set(2,1,y.z);
    rot.set(0,2,z.x);
    rot.set(1,2,z.y);
    rot.set(2,2,z.z);
    
//     z' = project z to (0,0,-1)
//     normalize(z').z gives cos(theta)
//     normalize(z').(0,0,-1) gives cos(phi)
*/
}  


void Camera::getFOV(float *fovx_, float *fovy_)
{
	*fovx_ = fovx;
	*fovy_ = fovy;
	return;
}

float Camera::getZNear()
{
	return near_dist;
}


void Camera::getCameraMatrix(Matrix &m)
{
	m = matrix;
}

void Camera::getCameraRotationMatrix(Matrix &crot)
{
	crot = rot;
}
