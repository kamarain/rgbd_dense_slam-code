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

#define __ImagePyramid_H__

#include "Image2.h"
/*#include <windows.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
*/
//#include <opencv2\opencv.hpp>
#include <assert.h>


#define MAX_PYRAMID_LEVELS 10

class ImagePyramid2 {
private:
public:
	int nLayers;
	Image2 pyramid[MAX_PYRAMID_LEVELS];
	ImagePyramid2();
	~ImagePyramid2();
	Image2 *getImagePtr(int index) { return &pyramid[index]; }
	Image2 &getImageRef(int index) { return pyramid[index]; }
	void releaseData();
	void createLayers(int gpuStatus);
	void lock();
	void unlock();
	void setName(const char *name);
	void setStream(cudaStream_t stream);
	virtual void gaussianBlur( Image2 *img, int size);
        void updateCudaArrays();
	void updateLayerHires(Image2 *hires, Image2 *target,int layer);

	virtual void updateLayers();
	//void updateLayersCuda();
	void updateLayersThreshold();

	void thresholdLayers(int threshold);

	int loadPyramid(char *fileName, int nLayers=1, int targetResoX=0, bool colorFlag = false, unsigned int gpuStatus=CREATE_GPU_TEXTURE);
	void correctLighting(Image2 &src, Image2 &dst);

	int updatePyramid(char *fileName, bool lightingCorrectionFlag=false, bool whiteFlag=false);

	int updatePyramid(void *data, bool lightingCorrectionFlag=false, bool whiteFlag=false);

    int createHdrPyramid(unsigned int width, unsigned int height, int nchannels, int nLayers=1, bool showDynamicRange=true,unsigned int gpuStatus=CREATE_GPU_TEXTURE, bool renderable=true);
    int createPyramid(unsigned int width, unsigned int height, int nChannels, int nLayers=1, unsigned int gpuStatus=CREATE_GPU_TEXTURE,bool writeDiscard=false, bool renderable=true);

	virtual int downSample2( Image2 *img );
	void downSampleThreshold2(Image2 *img, Image2 *targetImg);

	void copyPyramid(ImagePyramid2 *srcPyramid);

	void zeroPyramid();

	virtual int downSample2( Image2 *img, Image2 *img2);
};

class DisparityPyramid : public ImagePyramid2
{
public:
	DisparityPyramid() {};
	~DisparityPyramid() {};
	void updateLayers() {
		//for (int i = 1; i < nLayers; i++)
		//	downSample2(&pyramid[i-1],&pyramid[i]);
	}
	int downSample2( Image2 *img );

	int downSample2( Image2 *img, Image2 *img2);
};
