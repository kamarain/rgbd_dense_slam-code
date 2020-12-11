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
#include <cwchar>

// Kernel that executes on the CUDA device
/*
__global__ void rgb2GrayKernel( unsigned char *rgbPtr, unsigned char *grayPtr,int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	grayPtr[idx] = (rgbPtr[idx*3+0]*19588 + rgbPtr[idx*3+1]*38469 + rgbPtr[idx*3+2]*7471) >> 16; 	
}*/


__global__ void rgb2GrayHdrKernel( unsigned char *rgbPtr, float *grayPtr,int width)
{
	unsigned int xi = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int yi = blockIdx.y*blockDim.y+threadIdx.y;
	int offset = xi + yi*width;
	int offsetR = 3*xi + yi*width*3;
	int offsetG = offsetR+1;
	int offsetB = offsetR+2;	
	grayPtr[offset] = float((rgbPtr[offsetR]*19588 + rgbPtr[offsetG]*38469 + rgbPtr[offsetB]*7471) >> 16); 	
}


__global__ void rgbF2GrayHdrKernel( float *rgbPtr, float *grayPtr,int width)
{
	unsigned int xi = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int yi = blockIdx.y*blockDim.y+threadIdx.y;
	int offset = xi + yi*width;
	int offsetR = 3*xi + yi*width*3;
	int offsetG = offsetR+1;
	int offsetB = offsetR+2;	
	grayPtr[offset] = 0.3f*rgbPtr[offsetR] + 0.59f*rgbPtr[offsetG] + 0.11f*rgbPtr[offsetB]; 	
}

__global__ void convertHdrRGBKernel(unsigned char *srcPtr, float *dstPtr, unsigned int width, unsigned int height) {
    unsigned int xi = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int yi = blockIdx.y*blockDim.y+threadIdx.y;
    int pitch = width*3;
    int offsetR = 3*xi + yi*pitch;
    int offsetG = offsetR+1;
    int offsetB = offsetR+2;
    dstPtr[offsetR] = ((float)srcPtr[offsetR])/255.0f;
    dstPtr[offsetG] = ((float)srcPtr[offsetG])/255.0f;
    dstPtr[offsetB] = ((float)srcPtr[offsetB])/255.0f;
}

extern "C" void rgb2GrayCuda(Image2 *rgbImage, Image2 *grayImage)
{
	if (rgbImage == 0 || rgbImage->devPtr == NULL || grayImage == 0 || grayImage->devPtr == NULL) return;
	float *grayPtr = (float*)grayImage->devPtr;
	dim3 cudaBlockSize(32,30,1);
	dim3 cudaGridSize(grayImage->width/cudaBlockSize.x,grayImage->height/cudaBlockSize.y,1);

	if (!rgbImage->hdr) {
	     unsigned char *rgbPtr = (unsigned char*)rgbImage->devPtr;
	     rgb2GrayHdrKernel<<<cudaGridSize,cudaBlockSize,0,rgbImage->cudaStream>>>(rgbPtr,grayPtr,grayImage->width);
	} else {
	     float *rgbPtr = (float*)rgbImage->devPtr;
	     rgbF2GrayHdrKernel<<<cudaGridSize,cudaBlockSize,0,rgbImage->cudaStream>>>(rgbPtr,grayPtr,grayImage->width);
	}
}


extern "C" void convertToHDRCuda(Image2 *imRGB, Image2 *imRGBHDR)
{
    if (imRGB == 0 || imRGB->devPtr == NULL || imRGBHDR == 0 || imRGBHDR->devPtr == NULL) return;
    unsigned char *srcPtr = (unsigned char*)imRGB->devPtr;
    float *dstPtr= (float*)imRGBHDR->devPtr;
    dim3 cudaBlockSize(32,30,1);
    dim3 cudaGridSize(imRGB->width/cudaBlockSize.x,imRGB->height/cudaBlockSize.y,1);
    convertHdrRGBKernel<<<cudaGridSize,cudaBlockSize,0,imRGB->cudaStream>>>(srcPtr,dstPtr,(unsigned int)imRGB->width,(unsigned int)imRGB->height);
}
