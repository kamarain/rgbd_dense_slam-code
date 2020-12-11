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

#pragma once
#include <string.h>

class Keyframe
{
    public:
        Keyframe();
        ~Keyframe();
        void setPose(float *Tin, float fovXin, float aspect);
        void setFov(float fovXin);
        void setCenter(float cxin, float cyin);
        void setLensDistortion(float *kcin);
        float *getPoseGL();
        float *getPose();
        float getFov();
        void setAspectRatio(float aspect);
        float getAspectRatio();
    private:
        float T[16];
        float Tgl[16];
        float cx,cy;
        float kc[5];
        float fovY;
        float aspectRatio;
};
