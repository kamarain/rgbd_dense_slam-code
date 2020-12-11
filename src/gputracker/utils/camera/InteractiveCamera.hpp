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

#if !defined(__INTERACTIVE_CAMERA_H__)
#define __INTERACTIVE_CAMERA_H__

#include "commonmath.h"

#define CAMERA_SPEED 0.06f
#define CAMERA_FRICTION 0.0002f
/*
#define CAMERA_MAX_SPEED 0.3f
#define CAMERA_SPEED 0.03f
#define CAMERA_FRICTION 0.008f;
*/
//#define CAMERA_MAX_ROT_SPEED 0.004f//25f
#define CAMERA_MAX_ROT_SPEED 0.025f
#define CAMERA_ROT_SPEED 0.012f //1.20f
#define CAMERA_ROT_FRICTION 0.002f //0.20f

#include "Camera.hpp"

class InteractiveCamera : public customCameraTools::Camera
{
private:
    float speed;
//    float speedY;
//    float speedZ;
    float strafe_speed;
    float rot_yaw_speed;
    float rot_pitch_speed;
    float angle_x, angle_y;
    int rotating;
	float cameraMaxSpeed;
	float cameraFriction;
        float cameraRotFriction;
public:
    InteractiveCamera();

    float getAngleX();
    float getAngleY();
    void  setAngleX(float x);
    void  setAngleY(float y);
	void reset();
    void addFriction();
    void updateSpeed(float amountX);
    void saveCameraMatrix(const char *fn);
    void loadCameraMatrix(const char *fn);
    void saveCameraMatrixASCII(const char *fn);
    //void loadCameraMatrixASCII(const char *fn);

//    void updateSpeedY(float amountY);
//    void updateSpeedZ(float amountZ);
    void updateStrafeSpeed(float amount);
    void updateYawSpeed(float amount);
    void updatePitchSpeed(float amount);
    void setFriction(float friction) {cameraFriction=friction;}
    void setRotFriction(float rotFriction) {cameraRotFriction=rotFriction;}
    void update();
	void setCameraMaxSpeed(float speed) { cameraMaxSpeed = speed; /*cameraFriction = speed/50.0f;*/ }
};

#endif
