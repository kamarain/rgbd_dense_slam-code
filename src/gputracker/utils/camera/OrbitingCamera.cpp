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
#include "OrbitingCamera.h"

OrbitingCamera::OrbitingCamera(float radius) : Camera()
{
    strafe_speed = 0;
    speed = 0;
	cameraMaxSpeed = 1.0f;
	cameraFriction = ORBITING_FRICTION;
	cameraRotFriction = ORBITING_ROT_FRICTION;
//    speedY = 0;
//    speedZ = 0;
	this->radius = radius;
	pitchSpeed = 0;
    yawSpeed = 0;

	yaw = 0;
    pitch = 0;
}
void OrbitingCamera::loadCameraMatrix(const char *fn) {
    FILE *f = fopen(fn,"rb");
    fread(&yaw, 1, sizeof(float)*1 , f);
    fread(&pitch, 1, sizeof(float)*1 , f);
    fread(&radius, 1, sizeof(float)*1 , f);
    fclose(f);
    update();
    activate();
}


void OrbitingCamera::saveCameraMatrix(const char *fn) {

    FILE *f = fopen(fn,"wb");
    fwrite(&yaw, 1, sizeof(float)*1 , f);
    fwrite(&pitch, 1, sizeof(float)*1 , f);
    fwrite(&radius, 1, sizeof(float)*1 , f);
    fclose(f);
}


float OrbitingCamera::getAngleX()
{
    return yaw;
}

float OrbitingCamera::getAngleY()
{
    return pitch;
}

void OrbitingCamera::setAngleX(float x)
{
    yaw = x;
}

void OrbitingCamera::setAngleY(float y)
{
    pitch = y;
}

void OrbitingCamera::addFriction()
{
	if (yawSpeed > 0) {
        yawSpeed -= cameraRotFriction;
        if (yawSpeed < 0)
            yawSpeed = 0;
    } else
    {
        yawSpeed += cameraRotFriction;
        if (yawSpeed > 0)
            yawSpeed = 0;
    }   

	if (pitchSpeed > 0) {
		pitchSpeed -= cameraRotFriction;
        if (pitchSpeed < 0)
            pitchSpeed = 0;
    } else
    {
        pitchSpeed += cameraRotFriction;
        if (pitchSpeed > 0)
            pitchSpeed = 0;
    }       
	yaw += yawSpeed;
	pitch += pitchSpeed;

	while (yaw >= mPI*2.0f) yaw -= mPI*2.0f;
    while (yaw < 0) yaw += mPI*2.0f;


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
    }   

	radius += speed;

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

void quat_mult(float *q1, float *q2, float *_r)
{
    float r[4];

    r[0] = _r[0]; r[1] = _r[1]; r[2] = _r[2]; r[3] = _r[3];

    r[0] = q1[3]*q2[0] + q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1];
    r[1] = q1[3]*q2[1] - q1[0]*q2[2] + q1[1]*q2[3] + q1[2]*q2[0];
    r[2] = q1[3]*q2[2] + q1[0]*q2[1] - q1[1]*q2[0] + q1[2]*q2[3];
    r[3] = q1[3]*q2[3] - q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2];

    _r[0] = r[0]; _r[1] = r[1]; _r[2] = r[2]; _r[3] = r[3];
}

void OrbitingCamera::reset() {
    pitch = 0;
    yaw = 0;
	pitchSpeed = 0;
	yawSpeed = 0;
	speed = 0;
	pos.x = 0;
    pos.y = 0;
    pos.z = 0;
	strafe_speed = 0;
	update();
	activate();
}

void OrbitingCamera::update()
{
    Quaternion dqx,dqy,q;
    
    dqy.loadAxisAngle(pitch,1,0,0);
    dqx.loadAxisAngle(yaw,0,1,0);
    q = dqx * dqy;
    rot.loadQuaternion(q);
   
    pos.x = -rot(0,2)*radius+ox;
    pos.y = -rot(1,2)*radius+oy;
    pos.z = -rot(2,2)*radius+oz;

	setupFrustum();

    return;
}

void OrbitingCamera::updateSpeed(float amount)
{
    speed += amount;

    if (speed >= cameraMaxSpeed)
        speed = cameraMaxSpeed;
    if (speed <= -cameraMaxSpeed)
        speed = -cameraMaxSpeed;
}

void OrbitingCamera::updateStrafeSpeed(float amount)
{
    strafe_speed += amount;

    if (strafe_speed >= cameraMaxSpeed)
        strafe_speed = cameraMaxSpeed;
    if (strafe_speed <= -cameraMaxSpeed)
        strafe_speed = -cameraMaxSpeed;
}

void OrbitingCamera::updatePitchSpeed(float amount)
{
    pitchSpeed += amount;
}

void OrbitingCamera::updateYawSpeed(float amount)
{
    yawSpeed += amount;
}
