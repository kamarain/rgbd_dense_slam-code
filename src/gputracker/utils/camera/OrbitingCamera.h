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

#define ORBITING_SPEED 0.06f
#define ORBITING_FRICTION 1e-2
/*
#define CAMERA_MAX_SPEED 0.3f
#define CAMERA_SPEED 0.03f
#define CAMERA_FRICTION 0.008f;
*/
#define ORBITING_MAX_ROT_SPEED 0.025f //2.50f
#define ORBITING_ROT_SPEED 0.012f //1.20f
#define ORBITING_ROT_FRICTION 3e-4

#include "Camera.hpp"

class OrbitingCamera : public customCameraTools::Camera
{
private:
    float speed;
//    float speedY;
//    float speedZ;
	float ox,oy,oz;
    float strafe_speed;
    float pitchSpeed;
    float yawSpeed;
    float yaw, pitch;
	float cameraMaxSpeed;
	float cameraFriction;
	float cameraRotFriction;
	float radius;
public:
    OrbitingCamera(float radius);

    float getAngleX();
    float getAngleY();
    void  setAngleX(float x);
    void  setAngleY(float y);
	void setRadius(float radius) { this->radius = radius;}
	void setOrbitingOrigin(float x, float y, float z) { ox = x; oy = y; oz = z; }
	void setFriction(float friction) {cameraFriction=friction;}
	void setRotFriction(float rotFriction) {cameraRotFriction=rotFriction;}
    void saveCameraMatrix(const char *fn);
    void loadCameraMatrix(const char *fn);
	void reset();
    void addFriction();
    void updateSpeed(float amountX);
//    void updateSpeedY(float amountY);
//    void updateSpeedZ(float amountZ);
    void updateStrafeSpeed(float amount);
    void updatePitchSpeed(float amount);
    void updateYawSpeed(float amount);
    void update();
	void setCameraMaxSpeed(float speed) { cameraMaxSpeed = speed; /*cameraFriction = speed/50.0f;*/ }
};
