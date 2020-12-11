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

#include <GL/glew.h>
// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <helper_functions.h>
#include <rendercheck_gl.h>
#include <image2/Image2.h>
#include <image2/ImagePyramid2.h>
#include <rendering/VertexBuffer2.h>

#include "adapted_histograms.h"

extern "C" void rgb2GrayCuda(Image2 *rgbImage, Image2 *grayImage);
extern "C" void undistortCuda(Image2 *srcImage, Image2 *dstImage, float *calibDataDev);
extern "C" void undistortRGBCuda(Image2 *srcImage, Image2 *dstImage, float *calibDataDev);
extern "C" void d2ZCuda(unsigned short *disparity16U, Image2 *zImage, float *calibDataDev, float xOff, float yOff);
extern "C" void d2ZCudaHdr(float *disparityHdr, Image2 *zImage, float *calibDataDev, float xOff, float yOff);
extern "C" void z2CloudCuda(Image2 *zImageIR, float *calibDataDev, VertexBuffer2 *vbuffer, Image2 *rgbImage, ImagePyramid2 *grayPyramid, Image2 *zImage, bool computeGradient);
extern "C" void gradientCuda(ImagePyramid2 &pyramid, ImagePyramid2 &gradXPyramid, ImagePyramid2 &gradYPyramid, int baseLayer);
extern "C" void convert2FloatCuda(Image2 *rgbInput, Image2 *imRGB);
extern "C" void convertToHDRCuda(Image2 *imRGB, Image2 *imRGBHDR);
extern "C" void downSample2Pyramid(ImagePyramid2 &pyramid);
extern "C" void downSample2Cuda(Image2 *hires, Image2 *lowres);
extern "C" void undistortDisparityCuda(unsigned short *disparity16U, float *uPtr, float *calibDataDev, int width, int height, cudaStream_t stream = 0);
extern "C" void testFuncCuda(int n_blocks, int block_size, float *a, int N);
extern "C" void setNormalsCuda(VertexBuffer2 *vbuffer, float *normalImage, float scale);
extern "C" void extractGradientMagnitudes(VertexBuffer2 *vbuffer, float *gradientScratchDev);
extern "C" double cudaHista(float *src, float *hist, int length, int bins, float *d_hist);
extern "C" double cudaHistb(float *src, float *hist, int length, int bins);

extern "C" void addVertexAttributesCuda(Image2 *zImage, float *calibDataDev, VertexBuffer2 *vbuffer, ImagePyramid2 *grayPyramid);
//extern "C" void filterIndices(VertexBuffer2 *vbuffer,unsigned int *histogramDev, int pixelSelectionAmount);
//extern "C" void filterIndices2(float *vdata,int nVertices, unsigned int *histogramDev,int pixelSelectionAmount, int *indexBufferDev, int stride);
extern "C" void warpPoints(VertexBuffer2 *vbuffer, float *weightsDev, float *T, float *calibDataDev, VertexBuffer2 *baseBuf,ImagePyramid2 *grayPyramid);
extern "C" void interpolateResidual(VertexBuffer2 *vbuffer, VertexBuffer2 *vbufferCur, float *T, float *calibDataDev, ImagePyramid2 &grayPyramid, int layer, float *residual, float *zweightsDev);
extern "C" void precomputeJacobianCuda(VertexBuffer2 *vbuffer, float *calibDataDev, float *jacobianDev1, float *jacobianDev2, float *jacobianDev3, float optScaleIn);
//extern "C" void precomputeJacobianUncompressedCuda(VertexBuffer2 *vbuffer, float *calibDataDev, float *jacobianDev1, float *jacobianDev2, float *jacobianDev3);
extern "C" void dotProductCuda(float *vecA, float *vecB, int count, float *result);
extern "C" void JTJCuda(float *JT,int count, float *JtJDev);
extern "C" void JTresidualCuda(float *JT, float *residual, int len, float *result6);
extern "C" void initCudaDotProduct(int maxResidualLen);
extern "C" void releaseCudaDotProduct();
extern "C" void solveMotionCuda(float *JtJDev, float *x, float *b, float optScaleOut);
extern "C" void warpBase(VertexBuffer2 *vbuffer,float *T);
extern "C" void matrixMult4Cuda(float *A, float *B, float *C);
extern "C" void matrixMult4NormalizedCuda(float *TrelDev, float *TabsDev, float *TnextDev);
extern "C" void generateWeights64(float *residualDev, int count, float *weightsDev, float *extWeightsDev, float *weightedResidualDev);
extern "C" void weightJacobian(float *jacobianTDev, float *weights, int count, float *weightedJacobianTDev);
extern "C" void compressVertexBuffer(VertexBuffer2 *vbufferSrc, VertexBuffer2 *vbufferDst, bool rgbVisualization = false);
extern "C" void compressVertexBuffer2(int *indicesExt,float *verticesExt,int pixelSelectionAmount,int srcStride, VertexBuffer2 *vbuffer);
extern "C" void collectPointsCuda(VertexBuffer2 *vbufferSrc, float *Tsrc, int collectedPoints256, VertexBuffer2 *vbufferDst, float *Tdst);
extern "C" void collectPointsCuda2(VertexBuffer2 *vbufferSrc, float *Tsrc,  int collectedPoints256, float *vertexImageDev, float *Tdst);
extern "C" void setPointIntensityCuda(VertexBuffer2 *vbuffer, float *Tsrc,float *Tdst,ImagePyramid2 *grayPyramid);
extern "C" void collectPointsIntoImageCuda(VertexBuffer2 *vbufferSrc, float *Tsrc, int collectedPoints256, float *vertexImageDev, float *Tdst, int width, int height, float *calibDataDev);
extern "C" void invertPoseCuda(float *A, float *iA, int N);
extern "C" void convertMatrixToPosAxisAngleCuda(float *A, float *posAxisAngle, int N);
extern "C" void filterPoseCuda(float *posAxisAngle, float *weightsDev, int N, float *T);
extern "C" void filterDepthIIR(VertexBuffer2 *vbuffer, VertexBuffer2 *vbufferCur, float *T, float *calibDataDev, float *weightsDev, int width, int height, float weightThreshold);

extern "C" void cudaWarp(float *pR, float *pL, int nPoints, float *matrixData, float *outPoints);
extern "C" void	cudaPointLineWarp(float *pR, float *linesL0, int nPoints,float *matrixData,float *linesL1);

extern "C" void initCudaDotProduct(int maxResidualLen);
extern "C" void releaseCudaDotProduct();

#include "hostUtils.h"
