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
#include <capture/videoSource.h>
#include <calib/calib.h>
#include <multicore/multicore.h>

using namespace cv;

class VertexBuffer2;

class VideoPreProcessorCPU
{
public:
    VideoPreProcessorCPU(VideoSource *source, const int nLayers, Calibration *calib);
    ~VideoPreProcessorCPU();
    int preprocess();
    Mat &getRGBImage();
    Mat &getGrayImage();//int index);
    int getWidth();
    int getHeight();
    int getDepthWidth();
    int getDepthHeight();
    bool isPlaying();
    void setVideoSource(VideoSource *kinect);
    VideoSource *getVideoSource();
    void setBrightnessNormalization(bool flag);
    void updateCalibration();
    void setPixelSelectionAmount(int pixelAmount);
    float *getSelected2dPoints() { return selectedPoints2d; }
    float *getSelected3dPoints() { return selectedPoints3d; }
    void getPlane(float *mean,float *normal);
    Mat &getDepthImageR();
    Mat &getDepthImageL();
    cv::Mat depthMapR;
    void release();
    int getFrame();
    void setFrame(int frame);
    void pause();
    void reset();
    bool isPaused();
private:
    int frameIndex;
    int frameInc;
    bool loopFlag;

    VideoSource *device;
    int width, dwidth;
    int height, dheight;
    bool brightnessNormalizationFlag;
    const int nMultiResolutionLayers;
    void cpuPreProcess(unsigned char *rgb, unsigned short *disparity);
    void downSample2( Mat  &img, Mat &img2 );
    void normalizeBrightness( Mat &src, Mat &dst );
    void planeRegression(ProjectData *fullPointSet, int count, float *mean, float *normal);
    void fastImageMedian(Mat &src, int *medianVal);
    Calibration *calib;
    cv::Mat depthMapL;

    ProjectData *fullPointSet;
    int pixelSelectionAmount;
    float *selectedPoints3d;
    float *selectedPoints2d;
    float planeMean[3];
    float planeNormal[3];
    bool planeComputed;
};
