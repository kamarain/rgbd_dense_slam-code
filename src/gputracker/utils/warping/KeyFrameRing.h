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

#ifndef KEYFRAMERING_H
#define KEYFRAMERING_H

#include <reconstruct/KeyFrame.h>
#include <image2/Image2.h>
#include <image2/ImagePyramid2.h>
#include <calib/calib.h>
#include <rendering/VertexBuffer2.h>


class KeyFrameRing {
public:
    KeyFrameRing();
    void init(int nRing, int width, int height, int nLayers, Calibration &calib);
    ~KeyFrameRing();
    void release();
    void updateSingleReference(int id, KeyFrame *kf,ImagePyramid2 &frame1C, Image2 &frame3C, VertexBuffer2 *vbuffer, float *imDepthDevIR, int pixelSelectionAmount);
    Image2 &getRGB(int index);
    VertexBuffer2 &getVertexBuffer(int index);
    ImagePyramid2 &getGray(int index);
    KeyFrame *getKeyFrame(int index);
    void resetTransforms();
    void updateCalibration();
    void renderBase();
    void renderSrcPoints(int cnt);
    void renderDstPoints(int cnt);
    void setTransforms(float *mtx16);
    void setPointSelectionAmount(int amount);
private:
   int findOldestSlot();
    int findNewSlot();
    KeyFrame **keyFrame;
    int keyFrameCount;
    float *calibDataDev;
//    unsigned int *histogramDev;
//    unsigned int *partialHistogramsDev;
    Calibration *calib;
    VertexBuffer2 *baseBuffer;
};


#endif // KEYFRAMERING_H
