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

class VertexBuffer2;

extern "C" void histogram256VBufGradMag(uint *d_Histogram, float *gradients, int nGradients, int pixelSelectionAmount);
extern "C" void generateWeights64(float *residualDev, int count, float *weightsDev, float *extWeightsDev,float *weightedResidual);
extern "C" void filterIndices3(VertexBuffer2 *vbuffer, float *gradientData, uint *histogramDev, int pixelSelectionAmount);
extern "C" void filterIndices4(VertexBuffer2 *vbuffer, float *gradientData, float *histogramDev, int pixelSelectionAmount, int nbins);

extern "C" void adaptedInitHistogram64(void);
extern "C" void adaptedCloseHistogram64(void);
extern "C" void adaptedInitHistogram256(void);
extern "C" void adaptedCloseHistogram256(void);

extern "C" void histogram256VBufGradMagRaw(
    uint *d_Histogram,
    void *d_Data,
    uint byteCount,
    int selectionAmount
);

//extern "C" void histogram256VBufGradMag(uint *d_Histogram, uint *partialHistogramsDev, float *gradients, int nGradients, int pixelSelectionAmount);
//extern "C" void generateWeights64(float *residualDev, int count, float *weightsDev, float *extWeightsDev,float *weightedResidual);
