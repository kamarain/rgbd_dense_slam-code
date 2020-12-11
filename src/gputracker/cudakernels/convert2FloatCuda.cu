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

#include <image2/Image2.h>
//#include <types.h>
#include "hostUtils.h"

__global__ void convert2FloatKernel( unsigned char *srcPtr, float *dstPtr, unsigned int pitch)
{
	unsigned int xi = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int yi = blockIdx.y*blockDim.y+threadIdx.y;
	int rowOffset = yi*pitch;	
	int offsetR = 3*xi+0 + rowOffset;
	int offsetG = 3*xi+1 + rowOffset;
	int offsetB = 3*xi+2 + rowOffset;
	dstPtr[offsetR] = float(srcPtr[offsetR])/255.0f;
	dstPtr[offsetG] = float(srcPtr[offsetG])/255.0f;
	dstPtr[offsetB] = float(srcPtr[offsetB])/255.0f;
}

extern "C" void convert2FloatCuda(Image2 *rgbInput, Image2 *imRGB)
{
	if (rgbInput == 0 || rgbInput->devPtr == NULL || imRGB == 0 || imRGB->devPtr == NULL) return;
	unsigned char *srcPtr = (unsigned char*)rgbInput->devPtr;
	float *dstPtr= (float*)imRGB->devPtr;
	dim3 cudaBlockSize(32,30,1);
	dim3 cudaGridSize(rgbInput->width/cudaBlockSize.x,rgbInput->height/cudaBlockSize.y,1);
	convert2FloatKernel<<<cudaGridSize,cudaBlockSize,0,rgbInput->cudaStream>>>(srcPtr,dstPtr,(unsigned int)rgbInput->width*3);
}

