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
#include "Quaternion.hpp"
#include "InteractiveCamera.hpp"

InteractiveCamera::InteractiveCamera() : Camera()
{
    strafe_speed = 0;
    speed = 0;
	cameraMaxSpeed = 1.0f;
	cameraFriction = CAMERA_FRICTION;
        cameraRotFriction = CAMERA_ROT_FRICTION;
//    speedY = 0;
//    speedZ = 0;
    rot_yaw_speed = 0;
    rot_pitch_speed = 0;
    angle_x = 0;
    angle_y = 0;
    rotating = 0;
}


float InteractiveCamera::getAngleX()
{
    return angle_x;
}

float InteractiveCamera::getAngleY()
{
    return angle_y;
}

void InteractiveCamera::setAngleX(float x)
{
    angle_x = x;
}

void InteractiveCamera::setAngleY(float y)
{
    angle_y = y;
}

void InteractiveCamera::addFriction()
{
    if (rot_yaw_speed > 0)
    {
        rot_yaw_speed -= cameraRotFriction;
        if (rot_yaw_speed < 0)
            rot_yaw_speed = 0;
    } else
    {
        rot_yaw_speed += cameraRotFriction;
        if (rot_yaw_speed > 0)
            rot_yaw_speed = 0;
    }

    if (rot_pitch_speed > 0)
    {
        rot_pitch_speed -= cameraRotFriction;
        if (rot_pitch_speed < 0)
            rot_pitch_speed = 0;
    } else
    {
        rot_pitch_speed += cameraRotFriction;
        if (rot_pitch_speed > 0)
            rot_pitch_speed = 0;
    }

    if (speed > 0)
    {
        speed -= cameraFriction;
        if (speed < 0)
            speed = 0;
    } else
    {
        speed += cameraFriction;
        if (speed > 0)
            speed = 0;
    }            /*
    if (speedY > 0)
    {
        speedY -= CAMERA_FRICTION;
        if (speedY < 0)
            speedY = 0;
    } else
    {
        speedY += CAMERA_FRICTION;
        if (speedY > 0)
            speedY = 0;
    }
    if (speedZ > 0)
    {
        speedZ -= CAMERA_FRICTION;
        if (speedZ < 0)
            speedZ = 0;
    } else
    {
        speedZ += CAMERA_FRICTION;
        if (speedZ > 0)
            speedZ = 0;
    }              */
    if (strafe_speed > 0)
    {
        strafe_speed -= cameraFriction;
        if (strafe_speed < 0)
            strafe_speed = 0;
    } else
    {
        strafe_speed += cameraFriction;
        if (strafe_speed > 0)
            strafe_speed = 0;
    }
}
/*
void quat_mult(float *q1, float *q2, float *_r)
{
    float r[4];

    r[0] = _r[0]; r[1] = _r[1]; r[2] = _r[2]; r[3] = _r[3];

    r[0] = q1[3]*q2[0] + q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1];
    r[1] = q1[3]*q2[1] - q1[0]*q2[2] + q1[1]*q2[3] + q1[2]*q2[0];
    r[2] = q1[3]*q2[2] + q1[0]*q2[1] - q1[1]*q2[0] + q1[2]*q2[3];
    r[3] = q1[3]*q2[3] - q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2];

    _r[0] = r[0]; _r[1] = r[1]; _r[2] = r[2]; _r[3] = r[3];
}*/

void InteractiveCamera::reset() {
    angle_y = 0;
    angle_x = 0;
	rot_yaw_speed = 0;
	rot_pitch_speed = 0;
	speed = 0;
	pos.x = 0;
    pos.y = 0;
    pos.z = 0;
	strafe_speed = 0;
	update();
	activate();
}

void InteractiveCamera::update()
{
    Quaternion dqx,dqy,q;

    angle_y += rot_yaw_speed;
    angle_x -= rot_pitch_speed;

    while (angle_x >= mPI*2.0f) angle_x -= mPI*2.0f;
    while (angle_x < 0) angle_x += mPI*2.0f;
/*    if (angle_y > mPI/2.0f+mPI/2.0f) angle_y = mPI/2.0f+mPI/2.0f;
    if (angle_y < -mPI/2.0f+mPI/2.0f) angle_y = -mPI/2.0f+mPI/2.0f;
  */
    
    dqy.loadAxisAngle(angle_y,1,0,0);
    dqx.loadAxisAngle(angle_x,0,1,0);
    q = dqx * dqy;
    rot.loadQuaternion(q);
   

    pos.x += -rot(0,2)*speed;
    pos.y += -rot(1,2)*speed;
    pos.z += -rot(2,2)*speed;

    pos.x += rot(0,0)*strafe_speed;
    pos.y += rot(1,0)*strafe_speed;
    pos.z += rot(2,0)*strafe_speed;

    setupFrustum();

    return;
}

void InteractiveCamera::loadCameraMatrix(const char *fn) {
    angle_x = 0;
    angle_y = 0;
    pos.x = 0;
    pos.y = 0;
    pos.z = 0;
    rot_yaw_speed = 0;
    rot_pitch_speed = 0;

    FILE *f = fopen(fn,"rb");
    fread(&pos.x, 1, sizeof(float)*1 , f);
    fread(&pos.y, 1, sizeof(float)*1 , f);
    fread(&pos.z, 1, sizeof(float)*1 , f);
    fread(&angle_x, 1, sizeof(float)*1 , f);
    fread(&angle_y, 1, sizeof(float)*1 , f);
    fclose(f);

    update();
    activate();
}


void InteractiveCamera::saveCameraMatrix(const char *fn) {

    FILE *f = fopen(fn,"wb");
    fwrite(&pos.x, 1, sizeof(float)*1 , f);
    fwrite(&pos.y, 1, sizeof(float)*1 , f);
    fwrite(&pos.z, 1, sizeof(float)*1 , f);
    fwrite(&angle_x, 1, sizeof(float)*1 , f);
    fwrite(&angle_y, 1, sizeof(float)*1 , f);
    fclose(f);
}

void InteractiveCamera::saveCameraMatrixASCII(const char *fn) {

    FILE *f = fopen(fn,"wb");
    fprintf(f, "%e %e %e %e %e %e\n",pos.x,pos.y,pos.z,angle_x,angle_y,angle_y);
    fclose(f);
}




void InteractiveCamera::updateSpeed(float amount)
{
    speed += amount;

    if (speed >= cameraMaxSpeed)
        speed = cameraMaxSpeed;
    if (speed <= -cameraMaxSpeed)
        speed = -cameraMaxSpeed;
}
/*
void Camera::updateSpeedY(float amount)
{
    speedY += amount;

    if (speedY >= CAMERA_MAX_SPEED)
        speedY = CAMERA_MAX_SPEED;
    if (speedY <= -CAMERA_MAX_SPEED)
        speedY = -CAMERA_MAX_SPEED;
}
void Camera::updateSpeedZ(float amount)
{
    speedZ += amount;

    if (speedZ >= CAMERA_MAX_SPEED)
        speedZ = CAMERA_MAX_SPEED;
    if (speedZ <= -CAMERA_MAX_SPEED)
        speedZ = -CAMERA_MAX_SPEED;
} */

void InteractiveCamera::updateStrafeSpeed(float amount)
{
    strafe_speed += amount;

    if (strafe_speed >= cameraMaxSpeed)
        strafe_speed = cameraMaxSpeed;
    if (strafe_speed <= -cameraMaxSpeed)
        strafe_speed = -cameraMaxSpeed;
}

void InteractiveCamera::updateYawSpeed(float amount)
{
    rot_yaw_speed += amount;
    rotating = 1;
    if (rot_yaw_speed >= CAMERA_MAX_ROT_SPEED)
        rot_yaw_speed = CAMERA_MAX_ROT_SPEED;
    if (rot_yaw_speed <= -CAMERA_MAX_ROT_SPEED)
        rot_yaw_speed = -CAMERA_MAX_ROT_SPEED;
}

void InteractiveCamera::updatePitchSpeed(float amount)
{
    rot_pitch_speed += amount;
    rotating = 1;
    if (rot_pitch_speed >= CAMERA_MAX_ROT_SPEED)
        rot_pitch_speed = CAMERA_MAX_ROT_SPEED;
    if (rot_pitch_speed <= -CAMERA_MAX_ROT_SPEED)
        rot_pitch_speed = -CAMERA_MAX_ROT_SPEED;
}
     /*
void InteractiveCamera::setDirection(float dx, float dy, float dz, float ux, float uy, float uz)
{
    Vector x,y,z;
    Vector tmpy;

    z.x = -dx; z.y = -dy, z.z = -dz;
    tmpy.x=ux; tmpy.y=uy; tmpy.z=uz;

    x.cross(tmpy,z);
    x.normalize();
    y.cross(z,x);
    y.normalize();

    rot.cell[0] = x.x;
    rot.cell[4] = x.y;
    rot.cell[8] = x.z;
    rot.cell[1] = y.x;
    rot.cell[5] = y.y;
    rot.cell[9] = y.z;
    rot.cell[2] = z.x;
    rot.cell[6] = z.y;
    rot.cell[10] = z.z;

    //z' = project z to (0,0,-1)
    // normalize(z').z gives cos(theta)
   //  normalize(z').(0,0,-1) gives cos(phi)

}      */


