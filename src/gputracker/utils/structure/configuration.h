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
#include <vector>
#include <opencv2/opencv.hpp>
using namespace cv;

class Image;
class ImagePyramid2;

class Configuration {
public:
	Configuration();
	~Configuration();
        //void generateFMatrix(int viewIndex0, int viewIndex1, float *F);
        //void project(int refIndex, int viewIndex, float *x3d, float *x2d);
        //void projectInit(int refIndex, int viewIndex, float *P);
    void warpToReference(float *curT, cv::Mat *curMap, float *TLR, float *Kir, float *Krgb, float *kc, cv::Mat *rgbMap, int colorThreshold = 15, float rayThreshold = 10);
    void init(float *refT, cv::Mat *refMap, float *TLR, float *refKir, float *refKrgb, float *kcRGB, int nBins, cv::Mat *rgbMap);
    void filterDepth(cv::Mat *resultMap, float *Kir, float *Krgb, float *kcRGB, float robustDistance=100.0f, int nMinimumSamples = 32);
private:
    int referenceFrame;
    float average(float *arr, int n);
    float robustAverage(float *arr, int n, float median, float robustDistance2, float *depthStdev);
    float robustInvAverage(float *arr, int n, float medianZ, float robustDistance2, float *depthStdev);
 //   float quickMedian(float *arr, int n);
    void clearNeighbors();
    void storeNeighbor(float *curT, float *TLR, cv::Mat *rgbMap);
    bool sanityCheckPoint(int refX, int refY, int width, unsigned char curR, unsigned char curG,unsigned char curB, float *iKir, float *p3, int colorThreshold, float rayThreshold2, int &dstOffset);
    void generatePixelSelectionMask(unsigned char *rgbData, float *refData, int width, int height, int pixelSelectionThreshold, unsigned char *mask);
    int nBins; // how many sample points all zmaps can produce per reference pixel?
    float *zAccumulator; // storage for z-samples
    // rgb images which can be used for photometrical refinement
    std::vector<cv::Mat *>           neighborImages;
    std::vector<std::vector<float> > neighborPoses;

    int *counterImage; // how many samples per pixel
    unsigned char *rgbReference; // reference colors
    unsigned char *rgbGradient;
    unsigned char *rgbMask; // pixelSelection mask for photometrical refinement
    float *stdevImage;
    // for zmap normalization:
    float refK[9];
    float refT[16];
};
