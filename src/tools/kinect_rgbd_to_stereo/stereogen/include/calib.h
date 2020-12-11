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

#define __CALIBRATION_H__

#define  KR_OFFSET   0
#define iKR_OFFSET   9
#define KcR_OFFSET  18
#define TLR_OFFSET  23
#define C1_OFFSET    39
#define KL_OFFSET   40
#define iKL_OFFSET  49
#define C0_OFFSET    58
#define MIND_OFFSET 59
#define MAXD_OFFSET 60
#define TRL_OFFSET  61
#define ALPHA0_OFFSET 77
#define ALPHA1_OFFSET 78
#define BETA_OFFSET 79 // 640*480 static disparity distortion function
#define CALIB_SIZE  307280

class Calibration {
private:
    double KL[9];
	double KR[9];
	double TLR[16];
	double TRL[16];
	double kcR[5];
	double kcL[5];
    //double B;
    //double b;
    float *beta;
    double c0,c1;
    double alpha0,alpha1;
    double minDist, maxDist;
    float *calibData; //KR,iKRL,kcR,TLR,B,KL,iKL,b,minDist,maxDist
    int width,height;
    bool useXYOffset; // are the 2d points defined based on IR or disparity image grid?
    void initOulu(const char *fileName, bool silent);
    bool fileExists(const char *fileName);
public:
    Calibration(const char *fn, bool silent = false);
    Calibration();
    ~Calibration();
    void copyCalib(Calibration *extCalib);
    double *getKR() { return &KR[0]; }
    double *getKL() { return &KL[0]; }
    double *getKcR() { return &kcR[0]; }
    double *getKcL() { return &kcL[0]; }
    double *getTLR() { return &TLR[0]; }
    //double  getB() { return B; }
    double getC0() { return c0; }
    double getC1() { return c1; }
    double getAlpha0() {return alpha0; }
    double getAlpha1() {return alpha1; }
    double getMinDist() { return minDist; }
    double getMaxDist() { return maxDist; }
    //double  getKinectBaseline() { return b; }
    void setMinDist(double minDist) { this->minDist = minDist; }
    void setMaxDist(double maxDist) { this->maxDist = maxDist; }
    void setupCalibDataBuffer(int width, int height);
    float *getCalibData() { return &calibData[0]; }
    float getFovX_R();
    float getFovY_R();
    float getFovX_L();
    float getFovY_L();
    void init(const char *fn, bool silent = true);
    int getWidth() { return width; }
    int getHeight() { return height; }
    bool isOffsetXY() { return useXYOffset; }
};
