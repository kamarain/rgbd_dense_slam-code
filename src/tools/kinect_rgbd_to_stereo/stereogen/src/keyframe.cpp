/*
stereo-gen
Copyright (c) 2014, Tommi Tykkälä, All rights reserved.

This source code is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3.0 of the License, or (at your option) any later version.

This source code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this source code.
*/

#include "keyframe.h"
#include "basic_math.h"

Keyframe::Keyframe()
{
    identity4x4(&T[0]);
    identity4x4(&Tgl[0]);
    fovY = 45;
    aspectRatio = 4.0f/3.0f;
    cx = 0.5f; cy = 0.5f;
    for (int i = 0; i < 5; i++) kc[i] = 0;
}

Keyframe::~Keyframe()
{

}


void Keyframe::setPose(float *Tin, float fov, float aspect) {
    memcpy(&T[0],Tin,sizeof(float)*16);
    setFov(fov);
    setAspectRatio(aspect);
}
void Keyframe::setAspectRatio(float aspect) {
    aspectRatio = aspect;
}
float Keyframe::getAspectRatio() {
    return aspectRatio;
}
void Keyframe::setFov(float fovin) {
    fovY = fovin;
}
void Keyframe::setCenter(float cxin, float cyin) {
    cx = cxin; cy = cyin;
}
void Keyframe::setLensDistortion(float *kcin) {
    memcpy(&kc[0],kcin,sizeof(float)*5);
}
float *Keyframe::getPoseGL() { transpose4x4(&T[0],&Tgl[0]); return &Tgl[0];}
float *Keyframe::getPose() { return &T[0];}
float Keyframe::getFov() { return fovY; }
