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

#include <opencv2/opencv.hpp>
#include <vector>
#include <calib/calib.h>
using namespace cv;
using namespace std;

typedef struct {
    float px,py;
    float rx,ry,rz;
    unsigned char magGrad;
} ProjectData;

class OMPFunctions
{
public:
    OMPFunctions();
    ~OMPFunctions();
    void convert2Gray(Mat &rgbImage, Mat &grayImage);
    void undistort(Mat &distorted, Mat &undistorted, float *K, float *iK, float *kc);
    void undistortLookup(Mat &distorted, Mat &undistorted);
    void undistortF(Mat &distorted, Mat &undistorted, float *K, float *iK, float *kc);
    void undistortLookupF(Mat &distorted, Mat &undistorted);
    void downSampleDepth(Mat &depthImage, Mat &depthImageSmall);
    void d2ZLow(Mat &dispImage, Mat &depthImageSmall, float c0, float c1, float minDist, float maxDist, float xOff, float yOff);
    void d2ZLowHdr(Mat &dispImage, Mat &depthImageSmall, float c0, float c1, float minDist, float maxDist, float xOff, float yOff);        
    void d2ZLowGPU(Mat &dispImage, Mat &depthImageSmall, float c0, float c1, float minDist, float maxDist, float xOff, float yOff); // gpu version has also normalization
    void solveZMap(Mat &dispImage, Mat &depthImage, float c0, float c1, float minDist, float maxDist);
    void z2Pts(Mat &depthMap, float *K, float *pts3);    
    void baselineTransform(Mat &depthImageL,Mat &depthImageR,float *KL, float *TLR, float *KR);
    void convert2Float(Mat &dispImage, Mat &hdrImage);
    void replaceUShortRange(Mat &dispImage, unsigned short valueStart, unsigned short valueEnd, unsigned short newValue);
    void downsampleDisparityMap(Mat &dispImage, Mat &dispImageLow);
    void baselineWarp(Mat &depthImageL,Mat &grayImageR,float *KL, float *TLR, float *KR, float *kc, ProjectData *fullPointSet);
    void generateDepthMap(ProjectData *fullPointSet, Mat &depthImageR);
    void undistortDisparityMap(Mat &dispImage, Mat &uDispImage, float alpha0, float alpha1, float *beta);
    void optimizePhotometrically(float *zmap, unsigned char *rgbMask, unsigned char *rgbReference, int w, int h, float *stdevImage, int nsamples, float *Kir, float *Krgb, float *kcRGB, std::vector<std::vector<float> >  &poseMat, std::vector<cv::Mat *> &neighborImage);
    void generateZArray(float *zmap,unsigned char *rgbMask,int width, int height,float *stdevImage,int nsamples,float *zArray);
    void generateCostVolume(float *zArray, int width, int height, int nsamples, unsigned char *mask, unsigned char *rgbReference, float *Kir, float *Krgb, float *kcRGB, std::vector<std::vector<float> >  &poseMat, std::vector<cv::Mat *> &neighborImage, float *costArray);
    void normalizeCosts(float *costArray, float *countArray, unsigned char *mask, int width, int height, int nsamples);
    void argMinCost(float *zArray, float *costArray, int width, int height, int nsamples, unsigned char *mask, float *zmap);
    double residualICP(Mat &xyzRef, Mat &maskRef, float *K, float *T, Mat &xyzCur, float *residual, float *jacobian, float scaleIn, float depthThreshold, int stride);
    double residualPhotometric(Mat &xyz,Mat &selection, int nPoints, float *kc, float *KR, float *TLR, float *T, cv::Mat &grayRef, float* residual, float *jacobian, float *wjacobian, int layer, float intensityThreshold, int stride);
    void refineDepthMap(Mat &xyzCur,Mat &weightsCur, float *K,  float *T, Mat &xyzRef,Mat &weightsRef, int stride, float depthThreshold=50.0f, float rayThreshold = 20.0f);
    void downSamplePointCloud(cv::Mat &hiresXYZ, cv::Mat &lowresXYZ, int stride);
    void generateNormals(cv::Mat &xyzImage, int stride);
    void downSampleMask(cv::Mat &hiresMask, cv::Mat &lowresMask);
    void downSampleHdrImage(cv::Mat &hiresImage, cv::Mat &lowresImage);
    void Jtresidual(float *jacobian, float *residual, int cnt, int rows, double *b);
    void AtA6(float *jacobian, int cnt, double *A);
    void generateOrientedPoints(cv::Mat &depthCPU, cv::Mat &xyzImage, float *KL, cv::Mat &normalStatus, float *kc, float *KR, float *TLR, cv::Mat &grayImage, int stride);
    void precomputePhotoJacobians(cv::Mat &xyzCur,float *kc, float *K, float *TLR, cv::Mat &gray, int nPoints, cv::Mat &photometricSelection, int stride, cv::Mat &photoJacobian, int layer, float scaleIn);
private:
    void init();
    int NCPU,NTHR;
    float *dxTable;
    bool mappingPrecomputed;
};

extern OMPFunctions *getMultiCoreDevice();
