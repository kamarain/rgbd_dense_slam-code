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

#include "KinectDisparityCompressor.h"

// compress 2x2 disparity blocks using max filter
// rationale: in run time 640->320 compression is done in anycase with 2x2 block max filter
// the 2x2 blocks here must match runtime 2x2 blocks!
void compressDisparity2(Mat &hiresD, Mat &loresD)
{
	unsigned short *dPtr = (unsigned short*)hiresD.ptr();
	unsigned short *dLowPtr = (unsigned short*)loresD.ptr();
	int widthLow = loresD.cols; int heightLow = loresD.rows;
	assert(widthLow*2 == hiresD.cols && heightLow*2 == hiresD.rows);

	int offset = 0;
	// fill first row of low res disparity by half 2x2 block maximum
	for (int i = 0; i < widthLow; i++, offset++) {
		int i2 = 2*i;
		dLowPtr[offset] = MAX(dPtr[i2],dPtr[i2+1]);
	}
	int width = 2*widthLow;
	// fill full 2x2 blocks by max value
	for (int j = 1; j < heightLow-1; j++) {
		for (int i = 0; i < widthLow; i++, offset++) {
			int i2 = 2*i; int j2 = 2*j-1;
			int offset2 = i2 + j2 * width;
			unsigned short m1 = MAX(dPtr[offset2],dPtr[offset2+1]);
			unsigned short m2 = MAX(dPtr[offset2+width],dPtr[offset2+1+width]);
			dLowPtr[offset] = MAX(m1,m2);
		}
	}
	// fill last row of low res disparity by half 2x2 block maximum
	for (int i = 0; i < widthLow; i++, offset++) {
		int offset2 = 2*i + (2*heightLow-1)*width;
		dLowPtr[offset] = MAX(dPtr[offset2],dPtr[offset2+1]);
	}
}


// decompress 2x2 disparity blocks using 2x2 replication
// rationale: in run time 640->320 compression is done in anycase with 2x2 block max filter
// the 2x2 blocks here must match runtime 2x2 blocks!
void decompressDisparity2(Mat &loresD, Mat &hiresD)
{
	unsigned short *dPtr = (unsigned short*)hiresD.ptr();
	unsigned short *dLowPtr = (unsigned short*)loresD.ptr();
	int widthLow = loresD.cols; int heightLow = loresD.rows;
	assert(widthLow*2 == hiresD.cols && heightLow*2 == hiresD.rows);

	int offset = 0;
	// fill first row of low res disparity by half 2x2 block maximum
	for (int i = 0; i < widthLow; i++, offset++) {
		int i2 = 2*i;
		dPtr[i2] = dLowPtr[offset];
		dPtr[i2+1] = dLowPtr[offset];
	}
	int width = 2*widthLow;
	// fill full 2x2 blocks by max value
	for (int j = 1; j < heightLow-1; j++) {
		for (int i = 0; i < widthLow; i++, offset++) {
			int i2 = 2*i; int j2 = 2*j-1;
			int offset2 = i2 + j2 * width;
			dPtr[offset2] = dLowPtr[offset];
			dPtr[offset2+1] = dLowPtr[offset];
			dPtr[offset2+width] = dLowPtr[offset];
			dPtr[offset2+width+1] = dLowPtr[offset];
		}
	}
	// fill last row of low res disparity by half 2x2 block maximum
	for (int i = 0; i < widthLow; i++, offset++) {
		int offset2 = 2*i + (2*heightLow-1)*width;
		dPtr[offset2] = dLowPtr[offset];
		dPtr[offset2+1] = dLowPtr[offset];
	}
}
