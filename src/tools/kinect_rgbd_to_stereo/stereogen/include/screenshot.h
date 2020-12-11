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

class ScreenShot {
public:
    ScreenShot(const char *basepath, int x0, int y0, int width, int height);
    ~ScreenShot();
    void save();
private:
    int shotIndex;
    char basePath[512];
    unsigned char *screenShot;
    unsigned char *flippedImage;
    int x0,y0;
    int width,height;
    void flip(unsigned char *srcData, unsigned char *dstData);
    void writePPM(const char *fn, int dimx, int dimy, unsigned char *src);
//    cv::Mat flippedImage;
//    cv::Mat screenShot;
};
