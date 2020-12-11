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

#include "calib.h"
#include "multicore.h"

class ZConv {
private:
public:
        ZConv();
        ~ZConv();
        int convert(unsigned short *dptr, int dwidth, int dheight, float *zptr, int zwidth, int zheight, Calibration *calib);
        int d2z(unsigned short *dptr, int dwidth, int dheight, float *zptr, int zwidth, int zheight, Calibration *calib, bool bilateralFilter=false);
        int d2zHdr(float *dptr, int width, int height, float *zptr, int zwidth, int zheight, Calibration *calib, bool bilateralFiltering=false);
        int d2zGPU(unsigned short *dptr, int dwidth, int dheight, float *zptr, int zwidth, int zheight, Calibration *calib);
        void baselineTransform(float *depthImageL, float *depthImageR, int zwidth, int zheight, Calibration *calib);
        void baselineWarp(float *depthImageL,unsigned char *grayImageR, ProjectData *fullPointSet, int width, int height, Calibration *calib);
        void baselineWarpRGB(float *depthImageL,unsigned char *rgbDataR, ProjectData *fullPointSet, int width, int height, Calibration *calib);
        void undistortDisparityMap(unsigned short* disp16, float *udisp, int widht, int height, Calibration* calib);
        void dumpDepthRange(float *depthMap, int width, int height);
        void mapDisparityRange(unsigned short* ptr, int w, int h, int minD,int maxD);
        void increaseDynamics(float *dispImage,int w, int h, float scale);
        void setRange(float*ptr, int len, float minZ, float maxZ, float z);
};
