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

class FileSource : public VideoSource {
private:
    string loadingPathStr;
    int prevLoadIndex;
public:
    FileSource(const char *baseDir, bool flipY=false);
    ~FileSource();
    int getWidth();
    int getHeight();
    int getDisparityWidth();
    int getDisparityHeight();
    int fetchRawImages(unsigned char **rgbCPU, unsigned short **depthCPU, int frameIndex);
    void reset();
};
