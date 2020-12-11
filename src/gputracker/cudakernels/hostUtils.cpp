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

#include <cudakernels/hostUtils.h>
#include <GL/glew.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
//#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
//#include <cutil_gl_inline.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h
#include <rendercheck_gl.h>


extern "C" void testFuncCuda(int nBlocks, int blockSize, float *a, int val);

void checkCudaError(const char *message)
{
	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess) {
		fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
	}                         
}

// note: this requires one texture allocation before gives values back!
#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

void printFreeDeviceMemory() {
/*    size_t freeMemoryCuda,totalMemoryCuda;
    cuMemGetInfo(&freeMemoryCuda,&totalMemoryCuda);
    printf("free memory on cuda : %d %d %f percent\n",freeMemoryCuda,totalMemoryCuda,100.0f*float(freeMemoryCuda)/float(totalMemoryCuda));
*/
    int total_mem_kb = 0;
    glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX,
              &total_mem_kb);

    int cur_avail_mem_kb = 0;
    glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX,
              &cur_avail_mem_kb);

    printf("free memory on opengl : %d %d %f percent\n",cur_avail_mem_kb,total_mem_kb,100.0f*float(cur_avail_mem_kb)/float(total_mem_kb));
}


void printDevProp( cudaDeviceProp devProp )
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %u\n",  (unsigned int)devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n",  (unsigned int)devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %u\n",  (unsigned int)devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %u\n",  (unsigned int)devProp.totalConstMem);
    printf("Texture alignment:             %u\n",  (unsigned int)devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multi-processors:    %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    printf("Can map host memory:           %s\n",  (devProp.canMapHostMemory ? "Yes" : "No"));
    return;
}

void cudaTest() {
    float *a_d; // Pointer to host & device arrays
    const int N = 10; // Number of elements in arrays
    size_t size = N * sizeof( float );
    float *a_h = new float[N]; // Allocate array on host
    cudaMalloc( (void **)&a_d, size ); // Allocate array on device
    // Initialize host array and copy it to CUDA device
    for ( int i = 0; i < N; i++ ) a_h[i] = (float)i;
    cudaMemcpy( a_d, a_h, size, cudaMemcpyHostToDevice );
    // Do calculation on device:
    int block_size = 4;
    int n_blocks   = N / block_size + ( N % block_size == 0 ? 0 : 1 );
    testFuncCuda(n_blocks,block_size,a_d,N);
    // Retrieve result from device and store it in host array
    cudaMemcpy( a_h, a_d, sizeof( float ) * N, cudaMemcpyDeviceToHost );
    // Print results
    for ( int i = 0; i < N; i++ ) printf( "%d %f\n", i, a_h[i] ); // Cleanup
    delete[] a_h;
    cudaFree( a_d );
}

extern "C"
bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}
