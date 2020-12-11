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

#define __LINE_BUFFER_H__

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#if defined(__APPLE__) && defined(__MACH__)
#include <OpenGL/gl.h>	// Header File For The OpenGL32 Library
#include <OpenGL/glu.h>	// Header File For The GLu32 Library
#else
#include <GL/glew.h> // GLEW Library
#include <GL/gl.h>	// OpenGL32 Library
#include <GL/glu.h>	// GLU32 Library
#endif
//#include "RenderBuffer.h"

static const int lbufferSize = 2;

class LineBuffer {
private:
	enum {
		VERTEX_BUFFER = 0,
		COLOR_BUFFER1 = 1
	};
public:
	LineBuffer(int size);
	~LineBuffer();
	void addLine(float x, float y, float z, float x2, float y2, float z2, unsigned char r, unsigned char g, unsigned char b);
    void render();
    void render(int nSegments);
	void upload();
	void reset();
	void getPoint(unsigned int index, float *x, float *y, float *z);
	float *getPoint(int index);
	unsigned char *getColor(int index);
	int getPointCount();
    int getMaxPointCount();
	int nPoints;
	int maxPoints;
	float *xyz;
	unsigned char *rgb1;
	unsigned int buffers[lbufferSize];
	unsigned int newVerticeCount;
};	
