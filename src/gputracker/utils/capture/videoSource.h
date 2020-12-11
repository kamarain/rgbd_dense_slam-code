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

#include <string>
using namespace std;

#define RGB_WIDTH 640
#define RGB_HEIGHT 480
#define RGB_WIDTH_SMALL 320
#define RGB_HEIGHT_SMALL 240
#define DISPARITY_WIDTH 640
#define DISPARITY_HEIGHT 480
#define COMPRESSED_DISPARITY_WIDTH 320
#define COMPRESSED_DISPARITY_HEIGHT 240
const int kinectRgbSize   = RGB_WIDTH*RGB_HEIGHT*3*sizeof(unsigned char);
const int kinectRgbSizeSmall   = RGB_WIDTH_SMALL*RGB_HEIGHT_SMALL*3*sizeof(unsigned char);
const int kinectBayerSize = RGB_WIDTH*RGB_HEIGHT*sizeof(unsigned char);
const int compressedKinectDepthSize = COMPRESSED_DISPARITY_WIDTH*COMPRESSED_DISPARITY_HEIGHT*sizeof(unsigned short);
const int kinectDepthSize = DISPARITY_WIDTH*DISPARITY_HEIGHT*sizeof(unsigned short);

class VideoSource {
private:
public:
    virtual int getWidth() = 0;
    virtual int getHeight() = 0;
    virtual int getDisparityWidth() = 0;
    virtual int getDisparityHeight() = 0;
    virtual int getFrame() { return 0; }
    virtual void setFrame(int frame) {};
    virtual int fetchRawImages(unsigned char **rgbCPU, unsigned short **depthCPU, int frameIndex) = 0;
    // dummy calls by default
    virtual void setRecording(const char *, bool, bool saveToDisk=false, int nFrames = 0, bool averageFrames = false, bool compressedDepthFrames=true) { };
    virtual bool isRecording() { return false; }
    virtual bool isPaused() { return false; }
    virtual float getSecondsRemaining() { return 0.0f; }
    virtual void pause() {};
    virtual void start() {};
    virtual void stop() {};
    virtual void record() {};
    virtual void reset() {};
    virtual void setExposure(float exposureVal)  {};
};
