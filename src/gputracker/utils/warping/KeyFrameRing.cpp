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


#include <GL/glew.h> // GLEW Library
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cudakernels/cuda_funcs.h>
#include <warping/KeyFrameRing.h>
#include <reconstruct/basic_math.h>
#include "timer/performanceCounter.h"

//#define PERFORMANCE_TEST

void KeyFrameRing::setTransforms(float *mtx16) {
    for (int i = 0; i < keyFrameCount; i++) {
        memcpy(keyFrame[i]->Tbase,mtx16,sizeof(float)*16);
        keyFrame[i]->setupCPUTransform();
    }
}

KeyFrameRing::KeyFrameRing() {
    keyFrame = NULL;
    keyFrameCount = 0;
//    histogramDev = NULL;
//    partialHistogramsDev = NULL;
    calib = NULL;
}

void KeyFrameRing::setPointSelectionAmount(int amount) {
    for (int i = 0; i < keyFrameCount; i++) {
        KeyFrame *kf = keyFrame[i];
        kf->reallocJacobian(amount);
    }
}

void KeyFrameRing::init(int nRing, int width, int height, int nLayers, Calibration &calib) {
    keyFrame = new KeyFrame*[nRing];
    keyFrameCount = nRing;
    this->calib = &calib;

    calib.setupCalibDataBuffer(width,height);
    cudaMalloc((void **)&calibDataDev, CALIB_SIZE * sizeof(float));
    cudaMemcpyAsync((void*)calibDataDev,calib.getCalibData(),CALIB_SIZE*sizeof(float),cudaMemcpyHostToDevice,0);

    char buf[512];
    for (int i = 0; i < keyFrameCount; i++) {
        keyFrame[i] = new KeyFrame(width,height);
        KeyFrame *kf = keyFrame[i];
        kf->id = -1;
        createHdrImage(NULL,width,height,3,&kf->rgbImage,ONLY_GPU_TEXTURE, false);
        kf->grayPyramid.createHdrPyramid(width,height,1,nLayers,false,ONLY_GPU_TEXTURE); kf->grayPyramid.setName("grayPyramid1C");
        sprintf(buf,"ringbuf%d",i);
//        kf->vbuffer.init(width*height,false,VERTEXBUFFER_STRIDE,buf);
        kf->vbuffer.init(width*height,false,COMPRESSED_STRIDE,buf);
        kf->setCalibDevPtr(calibDataDev,calib.getCalibData()[KR_OFFSET]/160.0f,calib.getCalibData()[KR_OFFSET+4]/120.0f);
    }

//    cudaMalloc((void **)&histogramDev, HISTOGRAM256_BIN_COUNT * sizeof(unsigned int));
//    cudaMalloc((void **)&partialHistogramsDev, PARTIAL_HISTOGRAM256_COUNT * HISTOGRAM256_BIN_COUNT * sizeof(int));
}

void KeyFrameRing::updateCalibration() {
    if (calib == NULL) return;
    cudaMemcpyAsync((void*)calibDataDev,calib->getCalibData(),CALIB_SIZE*sizeof(float),cudaMemcpyHostToDevice,0);
}

void KeyFrameRing::resetTransforms() {
    for (int i = 0; i < keyFrameCount; i++) {
        keyFrame[i]->resetTransform();
    }
}

KeyFrameRing::~KeyFrameRing() {
}

void KeyFrameRing::release() {
    if (keyFrame != NULL) {
        for (int i = 0; i < keyFrameCount; i++) {
            keyFrame[i]->release();
            delete keyFrame[i]; keyFrame[i] = NULL;
        }
        delete[] keyFrame;
        keyFrame = NULL;
    }
    if (calibDataDev != NULL) {
        cudaFree(calibDataDev);
        calibDataDev = NULL;
    }
//    if (histogramDev != NULL) cudaFree(histogramDev); histogramDev = NULL;
//    if (partialHistogramsDev != NULL) cudaFree(partialHistogramsDev); partialHistogramsDev = NULL;
}


int KeyFrameRing::findOldestSlot() {
    int oldestIndex = INT_MAX;
    int freeSlot = 0;
    for (int i = 0; i < keyFrameCount; i++) {
        if (keyFrame[i]->id < oldestIndex) {
            oldestIndex = keyFrame[i]->id;
            freeSlot = i;
        }
    }
    return freeSlot;
}

int KeyFrameRing::findNewSlot() {
    int newIndex = -1;
    int newSlot = 0;
    for (int i = 0; i < keyFrameCount; i++) {
        if (keyFrame[i]->id >= newIndex) {
            newIndex = keyFrame[i]->id;
            newSlot = i;
        }
    }
    return newSlot;
}

KeyFrame *KeyFrameRing::getKeyFrame(int index) {
    int newSlot = findNewSlot();
    int selectedSlot = (newSlot + keyFrameCount - index)%keyFrameCount;
    return keyFrame[selectedSlot];
}


Image2 &KeyFrameRing::getRGB(int index) {
    int newSlot = findNewSlot();
    int selectedSlot = (newSlot + keyFrameCount - index)%keyFrameCount;
    return keyFrame[selectedSlot]->rgbImage;
}

VertexBuffer2 &KeyFrameRing::getVertexBuffer(int index) {
    int newSlot = findNewSlot();
    int selectedSlot = (newSlot + keyFrameCount - index)%keyFrameCount;
    return keyFrame[selectedSlot]->vbuffer;
}

ImagePyramid2 &KeyFrameRing::getGray(int index) {
    int newSlot = findNewSlot();
    int selectedSlot = (newSlot + keyFrameCount - index)%keyFrameCount;
    return keyFrame[selectedSlot]->grayPyramid;
}

void KeyFrameRing::updateSingleReference(int id, KeyFrame *kf,ImagePyramid2 &frame1C, Image2 &frame3C, VertexBuffer2 *vbuffer, float *imDepthDevIR, int pixelSelectionAmount) {
    PerformanceCounter timer;

    kf->id = id;

    #if defined(PERFORMANCE_TEST)
        float delay = 0.0f;
        float delays[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        timer.StartCounter();
    #endif

    kf->lock();

    #if defined(PERFORMANCE_TEST)
        cudaThreadSynchronize(); timer.StopCounter(); delay = timer.GetElapsedTime()*1000.0f; delays[0] += delay; cudaEventRecord(start,0);
    #endif

    Image2 imDepth(imDepthDevIR,frame3C.width,frame3C.height,frame3C.width*sizeof(float),1,true);
    kf->selectPixelsGPUCompressedLock(*vbuffer,pixelSelectionAmount,&imDepth,&frame1C);
    kf->setBaseTransform(kf->getNextBaseDev());

    #if defined(PERFORMANCE_TEST)
        printf("keylock: %3.1f, vbuf+mipcopy: %3.1f, hist256: %3.1f, pack index: %3.1f, v-attrib: %3.1f, jpre: %3.1f ",delays[0],delays[1],delays[2],delays[3],delays[4],delays[5]);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
}
