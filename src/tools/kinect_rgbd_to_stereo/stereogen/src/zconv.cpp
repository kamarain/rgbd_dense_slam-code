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

#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>
#include  "calib.h"
#include "basic_math.h"
#include "multicore.h"
#include "zconv.h"

using namespace cv;

ZConv::ZConv() {

}

ZConv::~ZConv() {

}

void ZConv::mapDisparityRange(unsigned short* ptr, int w, int h, int minD2,int maxD2) {
    int len = w*h;
    unsigned short minD = 65535;
    unsigned short maxD = 0;
    int cnt = 0;
    for (int i = 0; i < len; i++) {
        if (ptr[i] > 0) {
            ptr[i] = 65535-ptr[i];
            if (ptr[i] < minD) minD = ptr[i];
            if (ptr[i] > maxD) maxD = ptr[i];
        }
    }

/*    unsigned short trans = minD2-minD;
    float scale = float(maxD2-minD2+1)/float(maxD-minD+1);
    for (int i = 0; i < len; i++) {
        if (ptr[i]>0)
            ptr[i] = (unsigned short)(float(ptr[i]+trans)*scale);
    }*/
}

void ZConv::dumpDepthRange(float *depthMap, int width, int height) {
    float *zptr = depthMap;
    float minZ = FLT_MAX;
    float maxZ = FLT_MIN;
    int len = width*height;
    int cnt = 0;
    float *tmp = new float[len];
    for (int i = 0; i < len; i++) {
        if (zptr[i] < minZ) minZ = zptr[i];
        if (zptr[i] > maxZ) maxZ = zptr[i];
        if (zptr[i] != 0.0f) { tmp[cnt] = zptr[i] ; cnt++; }
    }
    float medianZ = quickMedian(tmp,cnt);
    printf("minz: %f maxz:%f median:%f cnt: %d\n",minZ,maxZ,medianZ,cnt); fflush(stdin); fflush(stdout);
    delete[] tmp;
}

void ZConv::baselineTransform(float *zptrSrc, float *zptrDst, int width, int height, Calibration *calib)
{
    memset(zptrDst,0,sizeof(float)*width*height);

    int prevW = calib->getWidth(); int prevH = calib->getHeight();
    calib->setupCalibDataBuffer(width,height);

    float *KL = &calib->getCalibData()[KL_OFFSET];
    float *KR = &calib->getCalibData()[KR_OFFSET];
    float *TLR = &calib->getCalibData()[TLR_OFFSET];

    OMPFunctions *multicore = getMultiCoreDevice();
    Mat depthMapL(height, width, CV_32FC1, zptrSrc);
    Mat depthMapR(height, width, CV_32FC1, zptrDst);

    multicore->baselineTransform(depthMapL,depthMapR,KL, TLR, KR);

    calib->setupCalibDataBuffer(prevW,prevH);
}


void ZConv::baselineWarp(float *depthImageL,unsigned char *grayDataR, ProjectData *fullPointSet, int width, int height, Calibration *calib) {
    int prevW = calib->getWidth(); int prevH = calib->getHeight();
    calib->setupCalibDataBuffer(width,height);

    float *KL = &calib->getCalibData()[KL_OFFSET];
    float *KR = &calib->getCalibData()[KR_OFFSET];
    float *TLR = &calib->getCalibData()[TLR_OFFSET];
    float *kc = &calib->getCalibData()[KcR_OFFSET];

    OMPFunctions *multicore = getMultiCoreDevice();
    Mat depthMapL(height, width, CV_32FC1, depthImageL);
    Mat grayImageR(height, width, CV_8UC1, grayDataR);
    multicore->baselineWarp(depthMapL,grayImageR,KL, TLR, KR, kc, fullPointSet);
    calib->setupCalibDataBuffer(prevW,prevH);
}

void ZConv::baselineWarpRGB(float *depthImageL,unsigned char *rgbDataR, ProjectData *fullPointSet, int width, int height, Calibration *calib) {
    int prevW = calib->getWidth(); int prevH = calib->getHeight();
    calib->setupCalibDataBuffer(width,height);

    float *KL = &calib->getCalibData()[KL_OFFSET];
    float *KR = &calib->getCalibData()[KR_OFFSET];
    float *TLR = &calib->getCalibData()[TLR_OFFSET];
    float *kc = &calib->getCalibData()[KcR_OFFSET];

    OMPFunctions *multicore = getMultiCoreDevice();
    Mat depthMapL(height, width, CV_32FC1, depthImageL);
    Mat rgbImageR(height, width, CV_8UC3, rgbDataR);
    multicore->baselineWarpRGB(depthMapL,rgbImageR,KL, TLR, KR, kc, fullPointSet);
    calib->setupCalibDataBuffer(prevW,prevH);
}


void ZConv::undistortDisparityMap(unsigned short* disp16, float *udisp, int width, int height, Calibration* calib) {
    int prevW = calib->getWidth(); int prevH = calib->getHeight();
    calib->setupCalibDataBuffer(width,height);

    float alpha0 = calib->getCalibData()[ALPHA0_OFFSET];
    float alpha1 = calib->getCalibData()[ALPHA1_OFFSET];
    float *beta = &calib->getCalibData()[BETA_OFFSET];

    OMPFunctions *multicore = getMultiCoreDevice();
    Mat dispImage(height, width, CV_16UC1, disp16);
    Mat uDispImage(height, width, CV_32FC1, udisp);
    multicore->undistortDisparityMap(dispImage,uDispImage, alpha0, alpha1, beta);
    calib->setupCalibDataBuffer(prevW,prevH);
}


int ZConv::d2z(unsigned short *dptr, int width, int height, float *zptr, int zwidth, int zheight, Calibration *calib, bool bilateralFiltering) {
    int prevW = calib->getWidth(); int prevH = calib->getHeight();
    Mat dispImage(height, width, CV_16UC1, dptr);
    calib->setupCalibDataBuffer(width,height);
    //float  B =  calib->getCalibData()[B_OFFSET];
    //float *KL = &calib->getCalibData()[KL_OFFSET];
    //float  b =  calib->getCalibData()[b_OFFSET];
    float c0 = calib->getCalibData()[C0_OFFSET];
    float c1 = calib->getCalibData()[C1_OFFSET];
    float minDist = calib->getCalibData()[MIND_OFFSET];
    float maxDist = calib->getCalibData()[MAXD_OFFSET];

    Mat depthImage(zheight, zwidth, CV_32FC1,zptr);
    OMPFunctions *multicore = getMultiCoreDevice();

    float xOff = 0.0f;
    float yOff = 0.0f;

//    maxDist = 4000;

    if (calib->isOffsetXY()) {
        xOff = -4.0f; yOff = -3.0f;
    }

    if (bilateralFiltering) {
        Mat dispImageHdr(height, width, CV_32FC1);
        Mat dispImageHdr2(zheight, zwidth, CV_32FC1);
        multicore->replaceUShortRange(dispImage, 2047, 0xffff, 0xffff);
        multicore->convert2Float(dispImage,dispImageHdr);
        cv::bilateralFilter(dispImageHdr, dispImageHdr2, -1, 3.0, 3.0f);
        if (width == zwidth) {
            multicore->d2Z(dispImageHdr2,depthImage,c0,c1,minDist,maxDist,xOff,yOff);
        } else {
            multicore->d2ZLowHdr(dispImageHdr2,depthImage,c0,c1,minDist,maxDist,xOff,yOff);
        }
    } else {
        if (width == zwidth) {
//            printf("1converting input disparity map from %d x %d -> %d x %d!\n",width,height,zwidth,zheight);
            multicore->d2Z(dispImage,depthImage,c0,c1,minDist,maxDist,xOff,yOff);
        } else {
    //        printf("2converting input disparity map from %d x %d -> %d x %d!\n",width,height,zwidth,zheight);
            multicore->d2ZLow(dispImage,depthImage,c0,c1,minDist,maxDist,xOff,yOff);
        }
    }
    calib->setupCalibDataBuffer(prevW,prevH);
}

void ZConv::setRange(float*ptr, int len, float minZ, float maxZ, float z) {
    for (int i = 0; i < len; i++) if (ptr[i] >= minZ && ptr[i] <= maxZ) ptr[i] = z;
}

int ZConv::d2zHdr(float *dptr, int width, int height, float *zptr, int zwidth, int zheight, Calibration *calib, bool bilateralFiltering) {
    int prevW = calib->getWidth(); int prevH = calib->getHeight();
    Mat dispImage(height, width, CV_32FC1, dptr);
    calib->setupCalibDataBuffer(width,height);
    //float  B =  calib->getCalibData()[B_OFFSET];
    //float *KL = &calib->getCalibData()[KL_OFFSET];
    //float  b =  calib->getCalibData()[b_OFFSET];
    float c0 = calib->getCalibData()[C0_OFFSET];
    float c1 = calib->getCalibData()[C1_OFFSET];
    float minDist = calib->getCalibData()[MIND_OFFSET];
    float maxDist = calib->getCalibData()[MAXD_OFFSET];

    Mat depthImage(zheight, zwidth, CV_32FC1,zptr);
    OMPFunctions *multicore = getMultiCoreDevice();

    float xOff = 0.0f;
    float yOff = 0.0f;

    if (calib->isOffsetXY()) {
        xOff = -4.0f; yOff = -3.0f;
    }

    if (bilateralFiltering) {
        Mat dispImageHdr2(zheight, zwidth, CV_32FC1);
        cv::bilateralFilter(dispImage, dispImageHdr2, -1, 3.0, 3.0f);
        multicore->d2ZLowHdr(dispImageHdr2,depthImage,c0,c1,minDist,maxDist,xOff,yOff);
    } else {
        multicore->d2ZLowHdr(dispImage,depthImage,c0,c1,minDist,maxDist,xOff,yOff);
    }
    calib->setupCalibDataBuffer(prevW,prevH);
}

//NOTE: this routine normalizes depth map (gpu compatible form)!
int ZConv::d2zGPU(unsigned short *dptr, int width, int height, float *zptr, int zwidth, int zheight, Calibration *calib) {
    Mat dispImage(height, width, CV_16UC1, dptr);
    int prevW = calib->getWidth(); int prevH = calib->getHeight();
    calib->setupCalibDataBuffer(width,height);
    //float  B =  calib->getCalibData()[B_OFFSET];
    //float *KL = &calib->getCalibData()[KL_OFFSET];
    //float  b =  calib->getCalibData()[b_OFFSET];
    float c0 = calib->getCalibData()[C0_OFFSET];
    float c1 = calib->getCalibData()[C1_OFFSET];
    float minDist = calib->getCalibData()[MIND_OFFSET];
    float maxDist = calib->getCalibData()[MAXD_OFFSET];

    Mat depthImage(zheight, zwidth, CV_32FC1,zptr);
    OMPFunctions *multicore = getMultiCoreDevice();

    float xOff = 0.0f;
    float yOff = 0.0f;

    if (calib->isOffsetXY()) {
        xOff = -4.0f; yOff = -3.0f;
    }


    multicore->d2ZLowGPU(dispImage,depthImage,c0,c1,minDist,maxDist, xOff, yOff);
    calib->setupCalibDataBuffer(prevW,prevH);
}


//NOTE: this routine will not normalize depth map!
int ZConv::convert(unsigned short *dptr, int width, int height, float *zptr, int zwidth, int zheight, Calibration *calib) {
    assert(width == 640 && height == 480);
    assert(zwidth == 320 && zheight == 240);
    int prevW = calib->getWidth(); int prevH = calib->getHeight();
    calib->setupCalibDataBuffer(width,height);
    float *KR = &calib->getCalibData()[KR_OFFSET];
    float *kcR = &calib->getCalibData()[KcR_OFFSET];
    float *TLR = &calib->getCalibData()[TLR_OFFSET];
    float c0 = calib->getCalibData()[C0_OFFSET];
    float c1 = calib->getCalibData()[C1_OFFSET];

    float minDist = calib->getCalibData()[MIND_OFFSET];
    float maxDist = calib->getCalibData()[MAXD_OFFSET];

    Mat dispImage(height, width, CV_16UC1, dptr);
    Mat depthImageSmall(zheight,    zwidth, CV_32FC1,zptr);

    float xOff = 0.0f;
    float yOff = 0.0f;

    if (calib->isOffsetXY()) {
        xOff = -4.0f; yOff = -3.0f;
    }

    OMPFunctions *multicore = getMultiCoreDevice();   
    multicore->d2ZLow(dispImage, depthImageSmall, c0,c1,minDist,maxDist, xOff, yOff);
    calib->setupCalibDataBuffer(prevW,prevH);
}

