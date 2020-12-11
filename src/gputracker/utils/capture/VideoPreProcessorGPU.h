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
#include <image2/Image2.h>
#include <image2/ImagePyramid2.h>
#include <capture/videoSource.h>
#include <calib/calib.h>

using namespace cv;

class VertexBuffer2;

class VideoPreProcessorGPU
{
public:
    VideoPreProcessorGPU(VideoSource *source, const int nLayers, Calibration *calib);
    ~VideoPreProcessorGPU();
    int preprocess(ImagePyramid2 &imBW, Image2 &imRGB, VertexBuffer2 *vbuffer,  float *imDepthDevIR, Image2 &imDepth, bool keyframeMode=false);
    int getWidth();
    int getHeight();
    int getDepthWidth();
    int getDepthHeight();
    bool isPlaying();
    void setVideoSource(VideoSource *kinect);
    VideoSource *getVideoSource();
    void updateCalibration();
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
    const int nMultiResolutionLayers;
    void gpuPreProcess(unsigned char *rgbDev, unsigned short *disparityDev,ImagePyramid2 &imBW, Image2 &imRGB, Image2 &imDepthIR, Image2 &imDepth, VertexBuffer2 *vbuffer, bool keyframeMode);
    void uploadCalibData();
    Calibration *calib;
};
