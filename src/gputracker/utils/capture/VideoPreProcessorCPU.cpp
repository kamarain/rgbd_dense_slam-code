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
#include <VideoPreProcessorCPU.h>
#include <opencv2/opencv.hpp>
#include <libfreenect.h>
#include <pthread.h>
#include <stdio.h>
#include <string>
#include <inttypes.h>
#include "timer/performanceCounter.h"
#include <reconstruct/basic_math.h>
#include <rendering/VertexBuffer2.h>
#include <multicore/multicore.h>
#include <reconstruct/zconv.h>

using namespace std;
using namespace cv;
//using namespace omp_functions;

static float *calibDataDev = NULL;
static float *uDisparityDev = NULL;

//static unsigned int histogramCPU[HISTOGRAM256_BIN_COUNT];

// gray pyramid
static cv::Mat distortedGray;
static cv::Mat distortedGrayHDR;
static cv::Mat normalizedGray;
static cv::Mat grayImageHDR[16];
static cv::Mat rgbImage;

//#define PERFORMANCE_TEST

bool VideoPreProcessorCPU::isPlaying() {
    if (device == NULL) return false;
    if (device->isPaused()) return false;
    return true;
}

int VideoPreProcessorCPU::getFrame() {
    return frameIndex;
}
void VideoPreProcessorCPU::setFrame(int frame) {
    frameIndex = frame;
}
void VideoPreProcessorCPU::pause() {
    frameInc = !frameInc;
    if (device) device->reset();
}
void VideoPreProcessorCPU::reset() {
    frameIndex = 0; frameInc = 0;
    if (device) device->reset();
}
bool VideoPreProcessorCPU::isPaused() {
    if (frameInc == 0) return true; else return false;
}


VideoPreProcessorCPU::VideoPreProcessorCPU(VideoSource *kinect, const int nLayers, Calibration *calib) : nMultiResolutionLayers(nLayers)
{
    assert(kinect != NULL);
    this->calib = calib;

    setVideoSource(kinect);

    brightnessNormalizationFlag = false;

    depthMapR = Mat::zeros(height,width,CV_32FC1);
    depthMapL = Mat::zeros(height,width,CV_32FC1);

    fullPointSet = new ProjectData[width*height];
    memset(fullPointSet,0,sizeof(ProjectData)*width*height);
    pixelSelectionAmount = 0;
    selectedPoints3d = NULL;
    selectedPoints2d = NULL;
    planeComputed = false;
};

VideoPreProcessorCPU::~VideoPreProcessorCPU() {
}

void VideoPreProcessorCPU::release() {
    depthMapR.release();
    depthMapL.release();
    delete[] fullPointSet;
    if (selectedPoints3d != NULL) delete[] selectedPoints3d;
    if (selectedPoints2d != NULL) delete[] selectedPoints2d;
}

void VideoPreProcessorCPU::downSample2( Mat  &img, Mat &img2 )
{
        if (img.channels() != 1) assert(0);

        unsigned int newWidth  = (unsigned int)img.cols/2;
        unsigned int newHeight = (unsigned int)img.rows/2;

        if (img.type() == CV_8UC1) {
            unsigned char *srcPtr = img.ptr();
            unsigned char *dstPtr = img2.ptr();

            unsigned int offset = 0;
            for (unsigned int j = 0; j < newHeight; j++) {
                    for (unsigned int i = 0; i < newWidth; i++,offset++) {
                        int offset2 = (i<<1)+(j<<1)*img.cols;
                        dstPtr[offset] = (srcPtr[offset2] + srcPtr[offset2+1] + srcPtr[offset2+img.cols] + srcPtr[offset2+1+img.cols])>>2;
                }
            }
        } else if (img.type() == CV_32FC1) {
            float *srcPtr = (float*)img.ptr();
            float *dstPtr = (float*)img2.ptr();

            unsigned int offset = 0;
            for (unsigned int j = 0; j < newHeight; j++) {
                    for (unsigned int i = 0; i < newWidth; i++,offset++) {
                        int offset2 = (i<<1)+(j<<1)*img.cols;
                        dstPtr[offset] = (srcPtr[offset2] + srcPtr[offset2+1] + srcPtr[offset2+img.cols] + srcPtr[offset2+1+img.cols])/4.0f;
                }
            }
        }
}

void VideoPreProcessorCPU::setPixelSelectionAmount(int pixelAmount) {
    pixelSelectionAmount = pixelAmount;
    if (selectedPoints3d != NULL) delete[] selectedPoints3d;
    if (selectedPoints2d != NULL) delete[] selectedPoints2d;

    selectedPoints3d = new float[pixelAmount*3];
    selectedPoints2d = new float[pixelAmount*2];
}

void VideoPreProcessorCPU::fastImageMedian(Mat &src, int *medianVal) {
    unsigned char *srcData = src.ptr();

    unsigned int hist[256]; memset(hist,0,256*sizeof(int));
    unsigned int mass = 0;
    int offset = 0;
    for (int j  = 0; j < src.rows; j++) {
        for (int i  = 0; i < src.cols; i++,offset++) {
            unsigned char v0 = srcData[offset];
            hist[v0]++;
            mass++;
        }
    }
    // seek median value
    int desiredMass = mass/2;
    int currentMass = 0;
    int threshold = 0;
    for (int i = 0; i < 256; i++) {
        currentMass += hist[i];
        if (currentMass >= desiredMass) { threshold = i; break;}
    }
    *medianVal = threshold;
}

void VideoPreProcessorCPU::normalizeBrightness( Mat &src, Mat &dst )
{
    int medianVal = 0;
    fastImageMedian(src,&medianVal);

    unsigned char *srcData = src.ptr();
    unsigned char *dstData = dst.ptr();

    // set median to 128, this normalizes global brightness variations
    int offset = 0;
    for (int j  = 0; j < src.rows; j++) {
        for (int i  = 0; i < src.cols; i++,offset++) {
            int v0 = srcData[offset];
            int v1 = MIN(MAX(v0 + (128 - medianVal),0),255);
            dstData[offset] = (unsigned char)v1;
        }
    }
}


Mat &VideoPreProcessorCPU::getGrayImage() {
    return distortedGrayHDR;//grayImageHDR[index];
}

Mat &VideoPreProcessorCPU::getRGBImage() {
    return rgbImage;
}


Mat &VideoPreProcessorCPU::getDepthImageR() {
    return depthMapR;
}

Mat &VideoPreProcessorCPU::getDepthImageL() {
    return depthMapL;
}

void VideoPreProcessorCPU::planeRegression(ProjectData *fullPointSet, int count, float *mean, float *normal) {
    mean[0] = 0; mean[1] = 0; mean[2] = 0;
    float realcnt = 0;
    for (int i = 0; i < count; i++) {
        if (fullPointSet[i].magGrad > 10) {
            mean[0] += fullPointSet[i].rx;
            mean[1] += fullPointSet[i].ry;
            mean[2] += fullPointSet[i].rz;
            realcnt++;
        }
    }
    mean[0] /= realcnt; mean[1] /= realcnt; mean[2] /= realcnt;

    float mtx[9] = {0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < count; i++) {
        if (fullPointSet[i].magGrad > 10) {
            float n[3] = {0,0,0};
            n[0] = fullPointSet[i].rx-mean[0];
            n[1] = fullPointSet[i].ry-mean[1];
            n[2] = fullPointSet[i].rz-mean[2];
            mtx[0] += n[0]*n[0]; mtx[1] += n[0]*n[1]; mtx[2] += n[0]*n[2];
            mtx[3] += n[0]*n[1]; mtx[4] += n[1]*n[1]; mtx[5] += n[1]*n[2];
            mtx[6] += n[0]*n[2]; mtx[7] += n[1]*n[2]; mtx[8] += n[2]*n[2];
        }
    }
    cv::Mat E, V;
    cv::Mat M(3,3,CV_32FC1,mtx);
    cv::eigen(M,E,V);
    normal[0] = V.at<float>(2,0);
    normal[1] = V.at<float>(2,1);
    normal[2] = V.at<float>(2,2);
    normalize(&normal[0]);
//    printf("mean: %f %f %f\n",mean[0],mean[1],mean[2]);
//    printf("normal: %f %f %f\n",normal[0],normal[1],normal[2]);
}


void VideoPreProcessorCPU::getPlane(float *mean, float *normal) {
    memcpy(mean,&planeMean[0],sizeof(float)*3);
    memcpy(normal,&planeNormal[0],sizeof(float)*3);
}

// generate hdr grayscale pyramid with normalized brightness
void VideoPreProcessorCPU::cpuPreProcess(unsigned char *rgbCPU, unsigned short *disparityCPU) {
    ZConv zconv;
    zconv.d2z(disparityCPU,dwidth,dheight,(float*)depthMapL.ptr(),width,height,calib);
    zconv.baselineTransform((float*)depthMapL.ptr(),(float*)depthMapR.ptr(),width,height,calib);
    Mat rgbHeader(height, width, CV_8UC3, rgbCPU);
    if ( distortedGray.data == NULL) {
        distortedGray = Mat::zeros(height,width,CV_8UC1);
        distortedGrayHDR = Mat::zeros(height,width,CV_32FC1);
        rgbImage = rgbHeader.clone();
    }
    cvtColor(rgbHeader,distortedGray,CV_RGB2GRAY); // 0.5ms with 1 core
    distortedGray.convertTo(distortedGrayHDR,CV_32FC1);
    OMPFunctions *multicore = getMultiCoreDevice();

    float *ptr = NULL; calib->setupCalibDataBuffer(rgbHeader.cols,rgbHeader.rows);
    cv::Mat K(3,3,CV_32FC1); ptr = (float*)K.ptr(); for (int i = 0; i < 9; i++) ptr[i] = calib->getCalibData()[KR_OFFSET+i];
    cv::Mat kc(5,1,CV_32FC1); ptr = (float*)kc.ptr(); for (int i = 0; i < 5; i++) ptr[i] = calib->getCalibData()[KcR_OFFSET+i];
    undistort(rgbHeader,rgbImage,K,kc);
    //multicore->undistortF(rgbHeader,rgbImage,&calib->getCalibData()[KR_OFFSET],&calib->getCalibData()[iKR_OFFSET],&calib->getCalibData()[KcR_OFFSET]); //1-2ms (4 cores)

    //rgbHeader.copyTo(rgbImage);

    if (!planeComputed) {
        //    calib->setupCalibDataBuffer(width,height);
        //    multicore->undistortF(distortedGray,hdrGray,&calib->getCalibData()[KR_OFFSET],&calib->getCalibData()[iKR_OFFSET],&calib->getCalibData()[KcR_OFFSET]); //1-2ms (4 cores)
        zconv.baselineWarp((float*)depthMapL.ptr(),distortedGray.ptr(),fullPointSet,width,height,calib);
        // plane regression variables:
        planeRegression(fullPointSet,width*height,&planeMean[0],&planeNormal[0]);
        planeComputed = true;
    }
    // early exit:
    return;
    /*
    multicore->generateDepthMap(fullPointSet, depthMapR);


    if (pixelSelectionAmount <= 0) return;


    //static int counter = 0;
    //char buf[512];
    //sprintf(buf,"scratch/tmpimg%04d.ppm",counter++);
    //imwrite(buf,distortedGray);

    int histogram[256]; int size = width*height;
    memset(histogram,0,sizeof(int)*256);
    for (int i = 0; i < size; i++) {
        histogram[fullPointSet[i].magGrad]++;
    }

    int mass = 0; int thresholdBin = 255;
    for (int i = 255; i >= 0; i--) {
        mass += histogram[i];
        if (mass > pixelSelectionAmount) {
            thresholdBin = i;
            break;
        }
    }

    int numSelected = 0;
    for (int i = 0; i < size; i++) {
        if (fullPointSet[i].magGrad > thresholdBin) {
            selectedPoints3d[numSelected*3+0] = fullPointSet[i].rx;
            selectedPoints3d[numSelected*3+1] = fullPointSet[i].ry;
            selectedPoints3d[numSelected*3+2] = fullPointSet[i].rz;
            selectedPoints2d[numSelected*2+0] = fullPointSet[i].px;
            selectedPoints2d[numSelected*2+1] = fullPointSet[i].py;
            numSelected++;
        }
    }
    if (numSelected == pixelSelectionAmount) return;

    for (int i = 0; i < size; i++) {
        if (fullPointSet[i].magGrad == thresholdBin) {
            selectedPoints3d[numSelected*3+0] = fullPointSet[i].rx;
            selectedPoints3d[numSelected*3+1] = fullPointSet[i].ry;
            selectedPoints3d[numSelected*3+2] = fullPointSet[i].rz;
            selectedPoints2d[numSelected*2+0] = fullPointSet[i].px;
            selectedPoints2d[numSelected*2+1] = fullPointSet[i].py;
            numSelected++;
            if (numSelected == pixelSelectionAmount) return;
        }
    }
*/
    /*
    OMPFunctions *multicore = getMultiCoreDevice();

    for (int i = 0; i < nMultiResolutionLayers; i++) {
        if (grayImageHDR[i].data == NULL) {
            grayImageHDR[i] = Mat::zeros(height>>i,width>>i,CV_32FC1);
            printf("allocated gray %d\n",i);
        }
        if (i == 0) {
            // allocate storage for undistortion
            if ( distortedGray.data == NULL) {
                distortedGray = Mat::zeros(height,width,CV_8UC1);
                undistortedGray = Mat::zeros(height,width,CV_8UC1);
                normalizedGray = Mat::zeros(height,width,CV_8UC1);
            }
            cvtColor(rgbHeader,distortedGray,CV_RGB2GRAY); // 0.5ms with 1 core
            // normalize brightness?
            if (brightnessNormalizationFlag) {
                normalizeBrightness(distortedGray,normalizedGray);
                multicore->undistortF(normalizedGray,grayImageHDR[0],&calib->getCalibData()[KR_OFFSET],&calib->getCalibData()[iKR_OFFSET],&calib->getCalibData()[KcR_OFFSET]); //1-2ms (4 cores)
            } else {
               multicore->undistortF(distortedGray,grayImageHDR[0],&calib->getCalibData()[KR_OFFSET],&calib->getCalibData()[iKR_OFFSET],&calib->getCalibData()[KcR_OFFSET]); //1-2ms (4 cores)
            }
        } else {
            downSample2(grayImageHDR[i-1],grayImageHDR[i]);
        }
    }
*/
}


int VideoPreProcessorCPU::preprocess()
{
    assert(device != NULL);

    // rgbDev will hold 320x240 rgb image, disparityDev will be 640x480 disparity image
    unsigned char *rgbCPU = NULL; unsigned short *disparityCPU = NULL;
    int ret = device->fetchRawImages(&rgbCPU, &disparityCPU,frameIndex);
    if (ret) {        
        cpuPreProcess(rgbCPU,disparityCPU);        
        frameIndex+=frameInc;
    } else {
        reset();
    }
    return ret;
}

int VideoPreProcessorCPU::getWidth()
{
	return width;
}

int VideoPreProcessorCPU::getHeight()
{
	return height;
}

int VideoPreProcessorCPU::getDepthWidth()
{
	return width;

}

int VideoPreProcessorCPU::getDepthHeight()
{
	return height;
}

void VideoPreProcessorCPU::setBrightnessNormalization(bool flag) {
    brightnessNormalizationFlag = flag;
}

void VideoPreProcessorCPU::setVideoSource( VideoSource *kinect )
{
    width   = kinect->getWidth();
    height  = kinect->getHeight();
    dwidth  = kinect->getDisparityWidth();
    dheight = kinect->getDisparityHeight();
    frameInc = 1;
    loopFlag = false;
    frameIndex = 0;
    printf("preprocessor expects %d x %d rgb input and %d x %d disparity input\n",width,height,dwidth,dheight);

    device  = kinect;
    calib->setupCalibDataBuffer(width,height);
    planeComputed = false;
    //	printf("video source set!\n");
}

VideoSource *VideoPreProcessorCPU::getVideoSource() {
    return device;
}
