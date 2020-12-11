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
//#include <GL/gl.h>	// OpenGL32 Library
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <opencv2/opencv.hpp>
#include <libfreenect.h>
#include <pthread.h>
#include <stdio.h>
#include <string>
#include "KinectDisparityCompressor.h"
#include "fileSource.h"

using namespace std;
using namespace cv;

static Mat rgbMat(RGB_HEIGHT,RGB_WIDTH,CV_8UC3);
static Mat rgbMatSmall(RGB_HEIGHT_SMALL,RGB_WIDTH_SMALL,CV_8UC3);
static Mat depthMat(DISPARITY_HEIGHT,DISPARITY_WIDTH,CV_16UC1);

FileSource::FileSource(const char *baseDir, bool flipY)
{
	loadingPathStr = baseDir;
    prevLoadIndex = -1;
	printf("loading path set to %s\n",loadingPathStr.c_str());
}

FileSource::~FileSource()
{

}

void FileSource::reset() {
    prevLoadIndex = -1;
}

int FileSource::getWidth()
{
	return RGB_WIDTH_SMALL;
}

int FileSource::getHeight()
{
	return RGB_HEIGHT_SMALL;
}

int FileSource::getDisparityWidth() {
        return DISPARITY_WIDTH;
}

int FileSource::getDisparityHeight() {
    return DISPARITY_HEIGHT;
}


int FileSource::fetchRawImages(unsigned char **rgbCPU, unsigned short **depthCPU, int frameIndex)
{
    int loadIndex = frameIndex+1;
    char buf[512];
    *rgbCPU = rgbMatSmall.ptr();
    *depthCPU = (unsigned short*)depthMat.ptr();

//    if (prevLoadIndex == loadIndex) return 1;
//    prevLoadIndex = loadIndex;

    bool loadOk = true;

    sprintf(buf,"%s/bayer_rgbimage%04d.ppm",loadingPathStr.c_str(),loadIndex);
    Mat bayerHeader = imread(buf,0);
    if (bayerHeader.data!=NULL) {
        cvtColor(bayerHeader,rgbMat,CV_BayerGB2RGB);
        cv::pyrDown(rgbMat,rgbMatSmall);//,rgbMatSmall.size());
    } else {
        sprintf(buf,"%s/%04d.ppm",loadingPathStr.c_str(),loadIndex);
        Mat rgbHeader = imread(buf,-1);
        if (rgbHeader.data == NULL) { /*printf("file %s not valid!\n",buf);*/ loadOk =false; }
        else {
            cvtColor(rgbHeader,rgbMatSmall,CV_RGB2BGR);
        }
    }

    sprintf(buf,"%s/rawdepth%04d.ppm",loadingPathStr.c_str(),loadIndex);
    Mat depthHeader = imread(buf,-1);
    if (depthHeader.data == NULL) { /*printf("file %s not valid!\n",buf);*/ loadOk = false; }
    else {
        depthHeader.copyTo(depthMat);
    }

   // printf("frame:%d, loadok: %d\n",frameIndex,int(loadOk)); fflush(stdout);

    return loadOk;
}

