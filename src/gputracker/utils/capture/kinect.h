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

#include "videoSource.h"

class Kinect : public VideoSource {
private:
	string recordingPathStr;
	void saveScratchBuffer();
	bool pauseFlag;
	bool initFailed;
	bool capturingFlag;
	bool saveToDisk;
	bool averageSavedFrames;
public:
    Kinect(const char *baseDir);
    ~Kinect();
    int getWidth();
    int getHeight();
    int getDisparityWidth();
    int getDisparityHeight();

    int fetchRawImages(unsigned char **rgbCPU, unsigned short **depthCPU, int frameIndex);
    // nFrames = 0 <-> max amount of frames
    void setRecording(const char *recordingPath, bool flag, bool saveToDisk=false, int nFrames=0, bool averageFrames = false, bool compressedDepthFrames=true);
    bool isRecording();
    bool isPaused();
    float getSecondsRemaining();
    void pause();
    void startKinect();
    void stopKinect();
    void start() { startKinect(); };
    void stop() { stopKinect(); };
    void record();
    void setExposure(float exposureVal);
};
