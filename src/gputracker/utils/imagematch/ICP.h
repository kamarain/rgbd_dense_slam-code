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

#define MAX_PHOTOMETRIC_REFERENCES 7

class ICP {
private:
    cv::Mat depthMapCur;
    cv::Mat grayCur[3];
    cv::Mat grayCurPrev[MAX_PHOTOMETRIC_REFERENCES][3];
    //cv::Mat maskRef[3];
    //cv::Mat maskCur[3];
    cv::Mat icpSelectionMask[3];
    cv::Mat xyzRef[3]; cv::Mat weightsRef; cv::Mat xyzHiresRef;
    cv::Mat xyzCur[3]; cv::Mat weightsCur;
    cv::Mat residualImage;
    cv::Mat jacobianImage;
    cv::Mat photoJacobian[3];
    cv::Mat normalStatus;
    int nFramesReceived;
    int nFramesFused;
    float KL[9],KR[9],TLR[16];
    float origKL[9],origKR[9];
    float kc[5];
    float T[16],Tinc[16],TphotoBase[MAX_PHOTOMETRIC_REFERENCES][16];
    int xyzStride;
    // an array of indices which are used for photometric minimization
    cv::Mat photometricSelection;
    // number of photometric points in use
    int nPhotometric;
    int nPhotometricReferences;
    float scaleIn;
    float scaleOut;
    void generatePyramids(cv::Mat &depthMap, cv::Mat &grayImage, int nPhotometricPoints, int nPhotometricReferences);
    void updatePyramids(cv::Mat &depthMap, cv::Mat &grayImage);
    void scaleIntrinsic(int layer);
    void cgm(double *A, double *b, double *x);
    void generateMatrix(double *x, double *Tx);
    void generateMatrixLie(double *x, double *Tx, double scaleOut);
    //void selectUniformNormals(cv::Mat &mask, int selectNPoints, cv::Mat &normalStatus, cv::Mat &selectionMask);
    void selectPointsICP(cv::Mat &xyzImage, int selectNPoints, cv::Mat &normalStatus, cv::Mat &selectionImage, int stride);
    void selectPointsPhotometric(cv::Mat &xyzImage, int selectNPoints, cv::Mat &selectionImage, int stride);
    //void initializeWeights(cv::Mat &mask, cv::Mat &weights, float w=1.0f);
    void initializeWeights(cv::Mat &xyzImage, cv::Mat &weights, int stride, float w=1.0f);
    int getNumberOfPhotometricReferences();
    bool photosGotBetter(double *photoError, double tolerance, double *prevPhotoError,double minimumDecrese);
    void setData(float *dstData, float *srcData);
    void upsampleCloud(cv::Mat &src, cv::Mat &dst);
    bool usePhotometric;
    bool useICP;
    int referenceResolutionX;
    int referenceResolutionY;
public:
    ICP();
	~ICP();
    void setCalib(float *KL, float *KR, float *TLR, float *kc);
    void setDepthMap(cv::Mat &depthMap, cv::Mat &grayImage, int nPhotometricPoints, int nPhotometricReferences);
    void release();
    void optimize(int *nIterations, bool verbose=false);
    void optimizePhotometric(int *nIterations, bool verbose);
    void optimizeGeometric(int *nIterations, bool verbose);
    void optimizeDICP(int *nIterations, bool verbose);
    void setMode(bool usePhotometric, bool useICP);
    //Image2 &getDepthRefTex(int layer=0);
    float *getIncrement();
    float *getBaseTransform();
    void updateReference();
    void updatePhotoReference();
    void reset();
    void getReferenceCloud(cv::Mat **xyzImage, int *stride);
    void markPhotometricSelection(cv::Mat &rgbImage, int r, int g, int b);
    void setReferenceResolution(int icpReferenceResoX,int icpReferenceResoY);

    //void getReferenceCloud(cv::Mat **xyzImage, int *stride);
};
