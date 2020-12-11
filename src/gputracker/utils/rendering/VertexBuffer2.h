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

// forward declare cuda stream type ( no need for headers)
struct CUstream_st;
typedef CUstream_st *cudaStream_t;
class Shader;
//VERTEXBUFFER STRIDE MUST BE DIVISIBLE BY 3 (normal rendering requires)
#define VERTEXBUFFER_STRIDE 21 // x, y, z, nx, ny, nz, x2d_1, y2d_1, r, g, b, gradX_1, gradY_1, gradMag_1, gray_1, gradX_2, gradY_2, gray_2, gradX_3, gradY_3, gray_3
#define BASEBUFFER_STRIDE 6 // x,y,z,r,g,b
#define COMPRESSED_STRIDE 9 // x,y,z,nx,ny,nz,gray_1,gray_2,gray_3

class VertexBuffer2 {
private:
    unsigned int vbo;
    unsigned int ibo;
    unsigned int maxNumVertices;
    unsigned int numVertices;
    unsigned int maxNumElements;
    unsigned int numElements;
    float *vertexDataHost;
    unsigned int *indexDataHost;
    bool m_renderableBufferFlag;
    int stride;
public:
    char name[512];
    void *devPtr;
    void *indexDevPtr;
    cudaStream_t cudaStream;
    VertexBuffer2();
    VertexBuffer2(int numVertices, bool renderableBuffer=true, int stride=VERTEXBUFFER_STRIDE, const char *vbufferName=NULL);
    void init(int numVertices, bool renderableBuffer=true,int stride=VERTEXBUFFER_STRIDE,const char *vbufferName=NULL); // for late init
    void release();
    ~VertexBuffer2();
    void *lock();
    void unlock();
    void *lockIndex();
    void unlockIndex();
    void setStream(cudaStream_t stream);
    void addVertex(float x, float y, float z, float r, float g, float b);
    void addVertexWithoutElement(float x, float y, float z, float r, float g, float b);
    void addLine(float x1, float y1, float z1, float x2, float y2, float z2, float r, float g, float b);
    void upload();
    void render();
    void render(Shader *shader, float *lightPos);
    void renderColor(float r, float g, float b, float a);
    // this is a hack routine to be able to render vertex buffer contents correctly, which are written by baseBuffer2.
    void renderBaseBufferLines();
    void renderAll();
    void renderPointRange2d(int start, int end);
    void setVertexAmount(int num);
    void copyTo(VertexBuffer2 &vbuf);
    // this is a hack routine to be able to render vertex normals as line segments from COMPRESSED_STRIDE buffer
    void renderPackedLines(int iboNum = 1);
    unsigned int getVertexCount() { return numVertices; }
    unsigned int getElementsCount() { return numElements; }
    unsigned int getMaxVertexCount() { return maxNumVertices; }
    unsigned int getMaxElementsCount() { return maxNumElements; }
    // this is for indicating how many valid indices exist in ibo
    void setElementsCount(int newCount) { if (newCount <= maxNumElements) { numElements = newCount; } else { printf("illegal element size! (%d,%d)\n",newCount,maxNumElements); fflush(stdin); fflush(stdout);}}
    unsigned int *getIndexBufferCPU() { return indexDataHost; }
    int getStride() { return stride; }
};
