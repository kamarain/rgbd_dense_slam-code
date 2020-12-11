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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#include "adapted_histograms.h"
#include <rendering/VertexBuffer2.h>
#include <histogram_common.h>
namespace hist64 {
    #include <histogram64.cu>
}
namespace hist256 {
    #include <histogram256.cu>
}

__global__ void histogram256KernelVBufGradMag(uint *d_PartialHistograms, float *d_Data, uint dataCount) {
    //Per-warp subhistogram storage
    __shared__ uint s_Hist[HISTOGRAM256_THREADBLOCK_MEMORY];
    uint *s_WarpHist= s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;

    //Clear shared memory storage for current threadblock before processing
#pragma unroll
    for(uint i = 0; i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE); i++)
        s_Hist[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;

    //Cycle through the entire data set, update subhistograms for each warp
#ifndef __DEVICE_EMULATION__
    const uint tag = threadIdx.x << (UINT_BITS - LOG2_WARP_SIZE);
#else
    const uint tag = 0;
#endif

    __syncthreads();
    for(uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, gridDim.x)){
        // fabs(+-255) + fabs(+-255) = 510
        uint data = (uint)(d_Data[pos]*255.0);
        //if (data > 0)
        hist256::addWord(s_WarpHist, data, tag);
    }

    //Merge per-warp histograms into per-block and write to global memory
    __syncthreads();
    for(uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT; bin += HISTOGRAM256_THREADBLOCK_SIZE){
        uint sum = 0;

        for(uint i = 0; i < WARP_COUNT; i++)
            sum += s_Hist[bin + i * HISTOGRAM256_BIN_COUNT] & TAG_MASK;

        d_PartialHistograms[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum;
    }
}

// this kernel assumes only one thread!
__global__ void seekThreshold256Kernel(uint *d_Histogram, int pixelAmount) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        // return the bin where pixel amount is exceeded
        uint binThreshold = 0; int selectedPixelAmount = 0;
        for (int i = (HISTOGRAM256_BIN_COUNT-1); i >= 0; i--) {
            selectedPixelAmount += d_Histogram[i];
            if (selectedPixelAmount > pixelAmount) {
                binThreshold = i;
                break;
            }
        }
        d_Histogram[0] = binThreshold;
        // how many pixels too much?
        d_Histogram[1] = selectedPixelAmount-pixelAmount;
    }
}

// this kernel assumes only one thread!
__global__ void seekThreshold256Kernel2(float *d_Histogram, int pixelAmount, int nbins) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        // return the bin where pixel amount is exceeded
        uint binThreshold = 0; int selectedPixelAmount = 0;
        for (int i = (nbins-1); i >= 0; i--) {
            selectedPixelAmount += (uint)d_Histogram[i];
            if (selectedPixelAmount > pixelAmount) {
                binThreshold = i;
                break;
            }
        }
        d_Histogram[0] = (float)binThreshold;
        // how many pixels too much?
        d_Histogram[1] = (float)(selectedPixelAmount-pixelAmount);
    }
}

__global__ void filterIndexKernel3(float *gradientData, uint *hist, int *indexPointer, int nthreads, int pointsPerThread) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int startOffset = idx*pointsPerThread;
    int lastOffset = startOffset + (pointsPerThread-1);
    //   if (lastOffset >= 320*240) return;

    __shared__ int countInSegment[1024];
    __shared__ int startIndexOfSegment[1024];
    __shared__ int mediocreCountInSegment[1024];
    __shared__ int mediocreStartIndexOfSegment[1024];

    int packedSegment[1024];
    int mediocrePackedSegment[1024];

    // pack each segment in parallel and store count
    uint threshold = hist[0];

    int cnt = 0, mediocreCnt = 0;
    for (int i = startOffset; i <= lastOffset; i++) {
        uint vertexQuality = (uint)(gradientData[i]*255);
        if (vertexQuality > threshold) {
            packedSegment[cnt] = i; cnt++;
        } else if (vertexQuality == threshold) {
            mediocrePackedSegment[mediocreCnt] = i; mediocreCnt++;
        }
    }
    countInSegment[idx] = cnt; mediocreCountInSegment[idx] = mediocreCnt;
    __syncthreads();
    // compute start indices in the packed buffer:
    if (idx == 0) {
        int numberOfPixelsTooMuch = int(hist[1]);
        startIndexOfSegment[0] = 0;
        for (int i = 1; i < nthreads; i++) {
            startIndexOfSegment[i] = startIndexOfSegment[i-1] + countInSegment[i-1];
        }
        mediocreStartIndexOfSegment[0] = startIndexOfSegment[nthreads-1]+countInSegment[nthreads-1];
        for (int i = 1; i < nthreads; i++) {
            if (numberOfPixelsTooMuch > 0 && mediocreCountInSegment[i-1] > 0) {
                mediocreCountInSegment[i-1] -= numberOfPixelsTooMuch;
                if (mediocreCountInSegment[i-1] < 0) {
                    numberOfPixelsTooMuch += mediocreCountInSegment[i-1];
                    mediocreCountInSegment[i-1] = 0;
                } else {
                    numberOfPixelsTooMuch = 0;
                }
            }
            mediocreStartIndexOfSegment[i] = mediocreStartIndexOfSegment[i-1] + mediocreCountInSegment[i-1];
        }
    }
    __syncthreads();

    int startPackedOffset = startIndexOfSegment[idx];
    for (int i = 0; i < countInSegment[idx]; i++) {
        indexPointer[startPackedOffset+i] = packedSegment[i];
    }
    int mediocreStartPackedOffset = mediocreStartIndexOfSegment[idx];
    for (int i = 0; i < mediocreCountInSegment[idx]; i++) {
        indexPointer[mediocreStartPackedOffset+i] = mediocrePackedSegment[i];
    }
}

__global__ void filterIndexKernel4(float *gradientData, float *hist, int *indexPointer, int nthreads, int pointsPerThread, int nbins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int startOffset = idx*pointsPerThread;
    int lastOffset = startOffset + (pointsPerThread-1);
    //   if (lastOffset >= 320*240) return;

    __shared__ int countInSegment[1024];
    __shared__ int startIndexOfSegment[1024];
    __shared__ int mediocreCountInSegment[1024];
    __shared__ int mediocreStartIndexOfSegment[1024];

    int packedSegment[1024];
    int mediocrePackedSegment[1024];

    // pack each segment in parallel and store count
    uint threshold = (uint)hist[0];

    int cnt = 0, mediocreCnt = 0;
    for (int i = startOffset; i <= lastOffset; i++) {
        uint vertexQuality = (uint)(gradientData[i]*(nbins-1)+0.5f);
        if (vertexQuality > threshold) {
            packedSegment[cnt] = i; cnt++;
        } else if (vertexQuality == threshold) {
            mediocrePackedSegment[mediocreCnt] = i; mediocreCnt++;
        }
    }
    countInSegment[idx] = cnt; mediocreCountInSegment[idx] = mediocreCnt;
    __syncthreads();
    // compute start indices in the packed buffer:
    if (idx == 0) {
        int numberOfPixelsTooMuch = int(hist[1]);
        startIndexOfSegment[0] = 0;
        for (int i = 1; i < nthreads; i++) {
            startIndexOfSegment[i] = startIndexOfSegment[i-1] + countInSegment[i-1];
        }
        mediocreStartIndexOfSegment[0] = startIndexOfSegment[nthreads-1]+countInSegment[nthreads-1];
        for (int i = 1; i < nthreads; i++) {
            if (numberOfPixelsTooMuch > 0 && mediocreCountInSegment[i-1] > 0) {
                mediocreCountInSegment[i-1] -= numberOfPixelsTooMuch;
                if (mediocreCountInSegment[i-1] < 0) {
                    numberOfPixelsTooMuch += mediocreCountInSegment[i-1];
                    mediocreCountInSegment[i-1] = 0;
                } else {
                    numberOfPixelsTooMuch = 0;
                }
            }
            mediocreStartIndexOfSegment[i] = mediocreStartIndexOfSegment[i-1] + mediocreCountInSegment[i-1];
        }
    }
    __syncthreads();

    int startPackedOffset = startIndexOfSegment[idx];
    for (int i = 0; i < countInSegment[idx]; i++) {
        indexPointer[startPackedOffset+i] = packedSegment[i];
    }
    int mediocreStartPackedOffset = mediocreStartIndexOfSegment[idx];
    for (int i = 0; i < mediocreCountInSegment[idx]; i++) {
        indexPointer[mediocreStartPackedOffset+i] = mediocrePackedSegment[i];
    }
}
////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU histogram
////////////////////////////////////////////////////////////////////////////////

extern "C" void histogram256VBufGradMagRaw(uint *d_Histogram, void *d_Data, uint byteCount,int pixelSelectionAmount) {

    assert( byteCount % sizeof(float) == 0 );

    histogram256KernelVBufGradMag<<<hist256::PARTIAL_HISTOGRAM256_COUNT, HISTOGRAM256_THREADBLOCK_SIZE>>>(
                                                                                                   hist256::d_PartialHistograms,
                                                                                                   (float *)d_Data,
                                                                                                   byteCount / sizeof(float));

    char buf[512]; sprintf(buf,"histogram256KernelVBufGradMag() execution failed, arguments: byteCount:%d\n",byteCount);
    getLastCudaError(buf);

    hist256::mergeHistogram256Kernel<<<HISTOGRAM256_BIN_COUNT, MERGE_THREADBLOCK_SIZE>>>(
                                                                                  d_Histogram,
                                                                                  hist256::d_PartialHistograms,
                                                                                  hist256::PARTIAL_HISTOGRAM256_COUNT
                                                                                  );
    getLastCudaError("mergeHistogram256Kernel() execution failed\n");
    seekThreshold256Kernel<<<1, 1>>>(d_Histogram, pixelSelectionAmount);
    getLastCudaError("seekThresholdKernel execution failed\n");
}


extern "C" void histogram256VBufGradMag(uint *d_Histogram, float *gradients, int nGradients, int pixelSelectionAmount)
{
    float *d_Data = gradients;
    int nVertices = nGradients;
    int byteCount = nVertices*sizeof(float);
    histogram256VBufGradMagRaw(d_Histogram,(void*)d_Data,byteCount,pixelSelectionAmount);
}

extern "C" void filterIndices3(VertexBuffer2 *vbuffer, float *gradientData, uint *histogramDev, int pixelSelectionAmount) {
    assert(vbuffer != NULL && histogramDev != NULL);
    int *indexPointer = (int*)vbuffer->indexDevPtr;
    int nVertices = vbuffer->getVertexCount();

    if (vbuffer->getStride() != VERTEXBUFFER_STRIDE) {
        printf("filterIndices: vertexbuffer has illegal stride (%d), must be %d!\n",vbuffer->getStride(),VERTEXBUFFER_STRIDE);
        fflush(stdin); fflush(stdout);
        return;
    }
    int nthreads = 512;
    int pointsPerThread = nVertices/nthreads;
    dim3 cudaBlockSize(nthreads,1,1);
    dim3 cudaGridSize(1,1,1);
    filterIndexKernel3<<<cudaGridSize,cudaBlockSize,0>>>(gradientData,histogramDev,indexPointer,nthreads,pointsPerThread);
    vbuffer->setElementsCount(pixelSelectionAmount);
}

extern "C" void filterIndices4(VertexBuffer2 *vbuffer, float *gradientData, float *histogramDev, int pixelSelectionAmount, int nbins) {
    assert(vbuffer != NULL && histogramDev != NULL);
    int *indexPointer = (int*)vbuffer->indexDevPtr;
    int nVertices = vbuffer->getVertexCount();

    if (vbuffer->getStride() != VERTEXBUFFER_STRIDE) {
        printf("filterIndices4: vertexbuffer has illegal stride (%d), must be %d!\n",vbuffer->getStride(),VERTEXBUFFER_STRIDE);
        fflush(stdin); fflush(stdout);
        return;
    }

    seekThreshold256Kernel2<<<1, 1>>>(histogramDev, pixelSelectionAmount,nbins);
    char buf[512]; sprintf(buf,"seekThresholdKernel2 execution failed, arguments, ptr: %d, npixels: %d\n",(unsigned int)(histogramDev!=NULL),pixelSelectionAmount);
    getLastCudaError(buf);

    int nthreads = 512;
    int pointsPerThread = nVertices/nthreads;
    dim3 cudaBlockSize(nthreads,1,1);
    dim3 cudaGridSize(1,1,1);
    filterIndexKernel4<<<cudaGridSize,cudaBlockSize,0>>>(gradientData,histogramDev,indexPointer,nthreads,pointsPerThread,nbins);
    vbuffer->setElementsCount(pixelSelectionAmount);
}


__global__ void discretizeResidualKernel(float *residualDev, uint *discreteResidual) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idx4 = idx*4;
    uint v1 = (uint)(255.0f*fabs(residualDev[idx4+0]));
    uint v2 = (uint)(255.0f*fabs(residualDev[idx4+1]));
    uint v3 = (uint)(255.0f*fabs(residualDev[idx4+2]));
    uint v4 = (uint)(255.0f*fabs(residualDev[idx4+3]));
    discreteResidual[idx] = (v1<<24)+(v2<<16)+(v3<<8)+v4;
//    discreteResidual[idx] = (unsigned char)(255.0f*fabs(residualDev[idx]));
}


// this kernel assumes only one thread!
__global__ void seekThreshold64Kernel(uint *d_Histogram, int pixelAmount) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx != 0) return;

    uint binThreshold = 0; int selectedPixelAmount = 0;
    for (int i = 0; i < HISTOGRAM64_BIN_COUNT; i++) {
        selectedPixelAmount += d_Histogram[i];
        if (selectedPixelAmount > pixelAmount) {
            binThreshold = i;
            break;
        }
    }
    d_Histogram[0] = binThreshold;
}


__global__ void generateWeights64Kernel(float *residual, uint *median64, float *weightsDev, float *extWeightsDev, float *weightedResidual) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    float s = 1.4826f * float(4*median64[0]+3)/255.0f;
    float b = 4.6851f;
    float r = residual[idx];

    float u = fabs(r)/s;
    float w = 0.0f;
    if (u <= b) {
        float ub = u / b; float ub2 = ub*ub;
        w = ( 1.0f - ub2 )*extWeightsDev[idx];
    }
    weightedResidual[idx] = r*w;
    weightsDev[idx] = w;
}



////////////////////////////////////////////////////////////////////////////////
// CPU interface to GPU histogram calculator
////////////////////////////////////////////////////////////////////////////////
//histogram64kernel() intermediate results buffer
//MAX_PARTIAL_HISTOGRAM64_COUNT == 32768 and HISTOGRAM64_THREADBLOCK_SIZE == 64
//amounts to max. 480MB of input data
static uint *d_Histogram64 = NULL;
static uint *discreteResidual = NULL;

//Internal memory allocation
extern "C" void adaptedInitHistogram64(void){
    if (d_Histogram64 == NULL) {
        hist64::initHistogram64();
//        checkCudaErrors( cudaMalloc((void **)&d_PartialHistograms, MAX_PARTIAL_HISTOGRAM64_COUNT * HISTOGRAM64_BIN_COUNT * sizeof(uint)) );
        checkCudaErrors( cudaMalloc((void **)&d_Histogram64, HISTOGRAM64_BIN_COUNT * sizeof(uint)) );
        checkCudaErrors( cudaMalloc((void **)&discreteResidual, 320*240*sizeof(uint)) );
    }
}

//Internal memory deallocation
extern "C" void adaptedCloseHistogram64(void) {
    if (d_Histogram64 != NULL) {
        hist64::closeHistogram64();
        //checkCudaErrors( cudaFree(d_PartialHistograms) ); d_PartialHistograms = NULL;
        checkCudaErrors( cudaFree(d_Histogram64) ); d_Histogram64 = NULL;
        checkCudaErrors( cudaFree(discreteResidual ) ); discreteResidual = NULL;
    }
}

//Internal memory allocation
extern "C" void adaptedInitHistogram256(void){
    if (hist256::d_PartialHistograms == NULL) {
        hist256::initHistogram256();
    }
}

//Internal memory deallocation
extern "C" void adaptedCloseHistogram256(void) {
    if (hist256::d_PartialHistograms != NULL) {
        hist256::closeHistogram256();
    }
}

extern "C" void generateWeights64(float *residualDev, int count, float *weightsDev, float *extWeightsDev, float *weightedResidualDev) {
    if (residualDev == NULL || count < 1024 || weightsDev == NULL || weightedResidualDev == NULL) {
        printf("invalid args to generateWeights64 count == %d!\n",count);
        return;
    }
    // enforce multiple of 1024 for element count -> max performance
    if (count%1024 != 0) {
        printf("wrong number of selected pixels!\n");
        return;
    }

    int cnt4 = count/4;
    dim3 cudaBlockSize(1024,1,1);
    dim3 cudaGridSize(cnt4/cudaBlockSize.x,1,1);
    while (cudaGridSize.x == 0) {
        cudaBlockSize.x /= 2;
        cudaGridSize.x = cnt4/cudaBlockSize.x;
    }
    discretizeResidualKernel<<<cudaGridSize,cudaBlockSize,0,0>>>(residualDev,(uint*)discreteResidual);

    int byteCount = count;
    const uint histogramCount = hist64::iDivUp(byteCount, HISTOGRAM64_THREADBLOCK_SIZE * hist64::iSnapDown(255, sizeof(hist64::data_t)));

    if (byteCount % sizeof(hist64::data_t) != 0) {
        printf("wrong bytecount!\n");
        return;
    }
    if (histogramCount > hist64::MAX_PARTIAL_HISTOGRAM64_COUNT) {
        printf("wrong histogram count!\n");
        return;
    }
    hist64::histogram64Kernel<<<histogramCount, HISTOGRAM64_THREADBLOCK_SIZE>>>(hist64::d_PartialHistograms,(hist64::data_t *)discreteResidual,byteCount / sizeof(hist64::data_t));
    getLastCudaError("generateWeights64 -- histogram64Kernel() execution failed\n");
    hist64::mergeHistogram64Kernel<<<HISTOGRAM64_BIN_COUNT, MERGE_THREADBLOCK_SIZE>>>(d_Histogram64,hist64::d_PartialHistograms,histogramCount);
    getLastCudaError("generateWeights64 -- mergeHistogram64() execution failed\n");
    seekThreshold64Kernel<<<1, 1>>>(d_Histogram64, count/2);
    getLastCudaError("generateWeights64 -- seekThreshold64() execution failed\n");

    cudaBlockSize.x = 1024;
    cudaGridSize.x = count / cudaBlockSize.x;
    generateWeights64Kernel<<<cudaGridSize,cudaBlockSize,0,0>>>(residualDev,d_Histogram64,weightsDev,extWeightsDev,weightedResidualDev);
    getLastCudaError("generateWeights64 -- generateWeights64Kernel() execution failed\n");
}
