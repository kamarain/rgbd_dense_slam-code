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

//#include <GL/glew.h>
//#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
//#include <cutil_gl_inline.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h
//#include <rendercheck_gl.h>
struct cudaDeviceProp;

void checkCudaError(const char *message);
void printFreeDeviceMemory();
void printDevProp( cudaDeviceProp devProp );
void cudaTest();
extern "C" bool isPow2(unsigned int x);
