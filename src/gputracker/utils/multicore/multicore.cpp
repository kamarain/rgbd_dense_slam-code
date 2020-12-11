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

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "multicore.h"
#include <omp.h>
#include <assert.h>
#include <reconstruct/basic_math.h>
//#include <image2/Image2.h>

static OMPFunctions multicore;

OMPFunctions::OMPFunctions() {
    init();
}

void OMPFunctions::init()
{
    /* get the total number of CPUs/cores available for OpenMP */
    NCPU = omp_get_num_procs();
    /* get the total number of threads requested */
    NTHR = omp_get_max_threads();

    printf("OpenMP cores: %i, threads: %i\n",NCPU,NTHR);

#pragma omp parallel
  {
        /* get the current thread ID in the parallel region */
        int tid = omp_get_thread_num();
        /* get the total number of threads available in this parallel region */
        int NPR = omp_get_num_threads();
        printf("Hyper thread %i/%i says hello!\n",tid,NPR);
  }
    dxTable = new float[640*480*2];
    mappingPrecomputed = false;
}


OMPFunctions::~OMPFunctions() {
    delete[] dxTable;
}

OMPFunctions *getMultiCoreDevice() {
    return &multicore;
}

void OMPFunctions::convert2Float(Mat &dispImage, Mat &hdrImage) {
    int width  = dispImage.cols;
    int height = dispImage.rows;

    unsigned short *dPtr = (unsigned short*)dispImage.ptr();
    float *hdrPtr = (float*)hdrImage.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (int xi = 0; xi < width; xi++,offset++) {
                hdrPtr[offset] = (float)dPtr[offset];
            }
        }
    }
}

void OMPFunctions::d2ZLowHdr(Mat &dispImage, Mat &depthImageSmall, float c0, float c1, float minDist, float maxDist, float xOff, float yOff) {
    int width  = depthImageSmall.cols;
    int height = depthImageSmall.rows;
    int width2  = width*2;
    int height2 = height*2;

    float *srcPtr = (float*)dispImage.ptr();
    float *dstPtr = (float*)depthImageSmall.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (int xi = 0; xi < width; xi++,offset++) {
                // IR image -> disparity image has constant offset (Konolige's tech guide)
                // http://www.ros.org/wiki/kinect_calibration/technical
                unsigned int sxi = 2*xi + xOff;
                unsigned int syi = 2*yi + yOff;
                if (sxi < (width2-1) && syi < (height2-1)) {
                    int srcIdx1 = sxi + 0 + (syi + 0) * width2;
                    int srcIdx2 = sxi + 1 + (syi + 0) * width2;
                    int srcIdx3 = sxi + 1 + (syi + 1) * width2;
                    int srcIdx4 = sxi + 0 + (syi + 1) * width2;
                    float d1 = (float)srcPtr[srcIdx1];
                    float d2 = (float)srcPtr[srcIdx2];
                    float d3 = (float)srcPtr[srcIdx3];
                    float d4 = (float)srcPtr[srcIdx4];
                    // prefer points close to sensor
                    float d = d1;
                    if (d2 < d) d = d2;
                    if (d3 < d) d = d3;
                    if (d4 < d) d = d4;

                    float z = 0.0f;
                    //if (d < 2047)
                    {
                        z = fabs(1.0f/(c0+c1*d));
                        //float z = fabs(8.0f*b*fx/(B-d));
                        if (z > maxDist || z < minDist) z = 0.0f;
                    }
                    dstPtr[offset] = z;
                    continue;
                }
                dstPtr[offset] = 0.0f;
            }
        }
    }
}


 void OMPFunctions::undistortDisparityMap(Mat &dispImage, Mat &uDispImage, float alpha0, float alpha1, float *beta) {
     int width  = dispImage.cols;
     int height = dispImage.rows;

     unsigned short *dPtr = (unsigned short*)dispImage.ptr();
     float *hdrPtr = (float*)uDispImage.ptr();

     int nBlocks = 4;
     int blockSize = height/nBlocks;
     int blockID = 0;
     #pragma omp parallel for private(blockID)
     for (blockID = 0; blockID < nBlocks; blockID++) {
         int offset = blockID*blockSize*width;
         for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
             for (int xi = 0; xi < width; xi++,offset++) {
                 float d = (float)dPtr[offset];
                 if (beta[offset]>0.0f) {
                    hdrPtr[offset] = d + beta[offset]*expf(alpha0-alpha1*d);
                    //return disp + dc_beta(v,u)*std::expf(dc_alpha[0] - dc_alpha[1]*disp);
                 } else {
                    hdrPtr[offset] = d;
                 }
             }
         }
     }
 }

 void OMPFunctions::z2Pts(Mat &depthMap, float *K, float *pts3) {
     int width  = depthMap.cols;
     int height = depthMap.rows;

     float *zPtr = (float*)depthMap.ptr();
     float iK[9]; inverse3x3(K,&iK[0]);

     int nBlocks = 4;
     int blockSize = height/nBlocks;
     int blockID = 0;
     #pragma omp parallel for private(blockID)
     for (blockID = 0; blockID < nBlocks; blockID++) {
         int offset = blockID*blockSize*width;
         int offset3 = offset*3;
         for (float yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
             for (float xi = 0; xi < width; xi++,offset++,offset3+=3) {
                 get3DPoint(xi,yi,zPtr[offset],iK,&pts3[offset3+0],&pts3[offset3+1],&pts3[offset3+2]);
             }
         }
     }
 }

/*
// fx and B manually *2 because they RGB_WIDTH_SMALL*2 = DEPTH_WIDTH (set in the main program according to rgb)
float fx = calibDataDev[KL_OFFSET]*2;
float B = calibDataDev[B_OFFSET]*2;
// the rest values read normally as they are reso invariant
float b = calibDataDev[b_OFFSET];
float minDist = calibDataDev[MIND_OFFSET];
float maxDist = calibDataDev[MAXD_OFFSET];
*/


void OMPFunctions::d2ZLow(Mat &dispImage, Mat &depthImageSmall, float c0, float c1, float minDist, float maxDist, float xOff, float yOff) {
    int width  = depthImageSmall.cols;
    int height = depthImageSmall.rows;
    int width2  = width*2;
    int height2 = height*2;


    unsigned short *dPtr = (unsigned short*)dispImage.ptr();
    float *zPtr = (float*)depthImageSmall.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (int xi = 0; xi < width; xi++,offset++) {
                // IR image -> disparity image has constant offset (Konolige's tech guide)
                // http://www.ros.org/wiki/kinect_calibration/technical
                unsigned int sxi = 2*xi + xOff;
                unsigned int syi = 2*yi + yOff;

                if (sxi < (width2-1) && syi < (height2-1)) {
                        int srcIdx1 = sxi + 0 + (syi + 0) * width2;
                        int srcIdx2 = sxi + 1 + (syi + 0) * width2;
                        int srcIdx3 = sxi + 1 + (syi + 1) * width2;
                        int srcIdx4 = sxi + 0 + (syi + 1) * width2;
                        float d1 = (float)dPtr[srcIdx1];
                        float d2 = (float)dPtr[srcIdx2];
                        float d3 = (float)dPtr[srcIdx3];
                        float d4 = (float)dPtr[srcIdx4];
                        if ((d1 < 2047) && (d2 < 2047) && (d3 < 2047) && (d4 < 2047)) {
                                float d = d1;
                                if (d2 < d) d = d2;
                                if (d3 < d) d = d3;
                                if (d4 < d) d = d4;
                                float z = fabs(1.0f/(c0+c1*d));
                                //float z = fabs(8.0f*b*fx/(B-d));
                                if (z > maxDist || z < minDist) z = 0.0f;
                                zPtr[offset] = z;//(z-minDist)/(maxDist-minDist);
                                continue;
                        }
                }
                zPtr[offset] = 0.0f;
            }
        }
    }
}


void OMPFunctions::replaceUShortRange(Mat &dispImage, unsigned short valueStart, unsigned short valueEnd, unsigned short newValue) {
    int width  = dispImage.cols;
    int height = dispImage.rows;

    unsigned short *dPtr = (unsigned short*)dispImage.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (int xi = 0; xi < width; xi++,offset++) {
                if (dPtr[offset] >= valueStart && dPtr[offset] <= valueEnd) dPtr[offset] = newValue;
            }
        }
    }
}


void OMPFunctions::d2ZLowGPU(Mat &dispImage, Mat &depthImageSmall, float c0, float c1, float minDist, float maxDist, float xOff, float yOff) {
    int width  = depthImageSmall.cols;
    int height = depthImageSmall.rows;
    int width2  = width*2;
    int height2 = height*2;


    unsigned short *dPtr = (unsigned short*)dispImage.ptr();
    float *zPtr = (float*)depthImageSmall.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (int xi = 0; xi < width; xi++,offset++) {
                // IR image -> disparity image has constant offset (Konolige's tech guide)
                // http://www.ros.org/wiki/kinect_calibration/technical
                unsigned int sxi = 2*xi + xOff;
                unsigned int syi = 2*yi + yOff;

                if (sxi < (width2-1) && syi < (height2-1)) {
                        int srcIdx1 = sxi + 0 + (syi + 0) * width2;
                        int srcIdx2 = sxi + 1 + (syi + 0) * width2;
                        int srcIdx3 = sxi + 1 + (syi + 1) * width2;
                        int srcIdx4 = sxi + 0 + (syi + 1) * width2;
                        float d1 = (float)dPtr[srcIdx1];
                        float d2 = (float)dPtr[srcIdx2];
                        float d3 = (float)dPtr[srcIdx3];
                        float d4 = (float)dPtr[srcIdx4];
                        if ((d1 < 2047) && (d2 < 2047) && (d3 < 2047) && (d4 < 2047)) {
                                float d = d1;
                                if (d2 < d) d = d2;
                                if (d3 < d) d = d3;
                                if (d4 < d) d = d4;
                                 float z = fabs(1.0f/(c0+c1*d));
                                //float z = fabs(8.0f*b*fx/(B-d));
                                if (z > maxDist || z < minDist) z = 0.0f;
                                zPtr[offset] = (z-minDist)/(maxDist-minDist);
                                continue;
                        }
                }
                zPtr[offset] = 0.0f;
            }
        }
    }
}


void OMPFunctions::downSampleDepth(Mat &depthImage, Mat &depthImageSmall) {
    int width = depthImageSmall.cols;
    int height = depthImageSmall.rows;

    float *dstPtr = (float*)depthImageSmall.ptr();
    float *srcPtr = (float*)depthImage.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int j = blockID*blockSize; j < (blockID+1)*blockSize; j++) {
            for (int i = 0; i < width; i++,offset++) {
               int offset2 = i*2+j*2*width*2;
               float z1 = srcPtr[offset2];
               float z2 = srcPtr[offset2+1];
               float z3 = srcPtr[offset2+width*2];
               float z4 = srcPtr[offset2+width*2+1];
               // TODO: bilateraalisuodatus?
               dstPtr[offset] = MIN(MIN(MIN(z1,z2),z3),z4);
            }
        }
    }
}

void OMPFunctions::solveZMap(Mat &dispImage, Mat &depthImage, float c0, float c1, float minDist, float maxDist) {
    unsigned short *dptr = (unsigned short*)dispImage.ptr();
    float *zptr = (float*)depthImage.ptr();


    int nBlocks = 4;
    int blockSize = dispImage.rows/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*dispImage.cols;
        for (int j = blockID*blockSize; j < (blockID+1)*blockSize; j++) {
            for (int i = 0; i < dispImage.cols; i++,offset++) {
                unsigned short d = dptr[offset];
                if (d < 2047) {
                    float z = fabs(1.0f/(c0+c1*d));
                    //float z = fabs(8.0f*b*fx/(B-d));
                    if (z > maxDist || z < minDist) z = 0.0f;
                    zptr[offset] = z;
                } else
                    zptr[offset] = 0;
            }
        }
    }
}


void OMPFunctions::baselineTransform(Mat &depthImageL,Mat &depthImageR,float *KL, float *TLR, float *KR) {
    int width  = depthImageL.cols;
    int height = depthImageL.rows;
    float *zptrSrc = (float*)depthImageL.ptr();
    float *zptrDst = (float*)depthImageR.ptr();
    for (int i = 0; i < width*height; i++) zptrDst[i] = FLT_MAX;

    float fx = KL[0];
    float fy = KL[4];
    float cx = KL[2];
    float cy = KL[5];

    // if multiple hits inside rgb image pixel, pick the one with minimum z
    // this prevents occluded pixels to interfere zmap
    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int j = blockID*blockSize; j < (blockID+1)*blockSize; j++) {
            for (int i = 0; i < width; i++,offset++) {
                float z = zptrSrc[offset];
                float p3[3],r3[3],p2[3];
                p3[0] = -(float(i) - cx) * z / fx;
                p3[1] = -(float(j) - cy) * z / fy;
                p3[2] = -z;
                transformRT3(TLR, p3, r3);
                matrixMultVec3(KR, r3, p2); p2[0] /= p2[2]; p2[1] /= p2[2];
                int xi = (int)p2[0];
                int yi = (int)p2[1];

               if (xi >= 0 && yi >= 0 && xi < width && yi < height) {
                    int offset = xi + yi * width;
                    float prevZ =  zptrDst[offset];
                    float newZ  = fabs(r3[2]);
                    if (newZ > 0 && newZ < prevZ) zptrDst[offset] = newZ;
                }
            }
        }
    }
    for (int i = 0; i < width*height; i++) if (zptrDst[i] == FLT_MAX) zptrDst[i] = 0.0f;
    return;
}

void OMPFunctions::baselineWarp(Mat &depthImageL,Mat &grayImageR,float *KL, float *TLR, float *KR, float *kc, ProjectData *fullPointSet) {
    int width  = depthImageL.cols;
    int height = depthImageL.rows;
    float *zptrSrc = (float*)depthImageL.ptr();
    unsigned char *grayPtrDst = (unsigned char*)grayImageR.ptr();

    float fx = KL[0];
    float fy = KL[4];
    float cx = KL[2];
    float cy = KL[5];

    // if multiple hits inside rgb image pixel, pick the one with minimum z
    // this prevents occluded pixels to interfere zmap
    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int j = blockID*blockSize; j < (blockID+1)*blockSize; j++) {
            for (int i = 0; i < width; i++,offset++) {
                float z = zptrSrc[offset];
                float p3[3],r3[3],r3n[3],p2[3],p2n[3];
                p3[0] = -(float(i) - cx) * z / fx;
                p3[1] = -(float(j) - cy) * z / fy;
                p3[2] = -z;
                transformRT3(TLR, p3, r3); r3n[0] = r3[0]/r3[2]; r3n[1] = r3[1]/r3[2]; r3n[2] = 1.0f;
                // p2: distorted point
                distortPointCPU(r3n,kc,KR,p2);
                // p2n: undistorted point
                //matrixMultVec3(KR, r3n, p2n);
                int xi = (int)p2[0];
                int yi = (int)p2[1];
                fullPointSet[offset].px      = p2[0];
                fullPointSet[offset].py      = p2[1];
                fullPointSet[offset].rx      = r3[0];
                fullPointSet[offset].ry      = r3[1];
                fullPointSet[offset].rz      = r3[2];
                fullPointSet[offset].magGrad = 0;
                if (xi > 0 && yi > 0 && xi < (width-1) && yi < (height-1)) {
                    int offset2 = xi + yi * width;
                    unsigned char gx0 =  grayPtrDst[offset2-1];
                    unsigned char gx1 =  grayPtrDst[offset2+1];
                    unsigned char gy0 =  grayPtrDst[offset2-width];
                    unsigned char gy1 =  grayPtrDst[offset2+width];
                    int dx = gx1-gx0;
                    int dy = gy1-gy0;
                    fullPointSet[offset].magGrad  = (abs(dx) + abs(dy))/2;
                }
            }
        }
    }
}

void OMPFunctions::generateDepthMap(ProjectData *fullPointSet, Mat &depthImageR) {
    int width  = depthImageR.cols;
    int height = depthImageR.rows;
    int size = width*height;
    float *zptrDst = (float*)depthImageR.ptr();

    for (int i = 0; i < size; i++) zptrDst[i] = FLT_MAX;

    // if multiple hits inside rgb image pixel, pick the one with minimum z
    // this prevents occluded pixels to interfere zmap
    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int j = blockID*blockSize; j < (blockID+1)*blockSize; j++) {
            for (int i = 0; i < width; i++,offset++) {
                float px = fullPointSet[offset].px;
                float py = fullPointSet[offset].py;
                float z = -fullPointSet[offset].rz;
                int xi = (int)(px+0.5f);
                int yi = (int)(py+0.5f);
                if (xi >= 0 && yi >= 0 && xi <= (width-1) && yi <= (height-1)) {
                    int offset2 = xi + yi * width;
                    float cz = zptrDst[offset2];
                    if (z < cz) zptrDst[offset2] = z;
                }
            }
        }
    }

    float minVal = FLT_MAX;
    float maxVal = 0;
    for (int i = 0; i < size; i++) {
        if (zptrDst[i] < minVal) minVal = zptrDst[i];
        if (zptrDst[i] > maxVal) maxVal = zptrDst[i];
    }
    for (int i = 0; i < size; i++) {
        zptrDst[i] = 0.5f;//255.0f*(zptrDst[i]-minVal)/(maxVal-minVal);
    }
    //printf("jeap\n"); fflush(stdin); fflush(stdout);
}



void OMPFunctions::convert2Gray(Mat &rgbImage, Mat &grayImage) {
    assert(rgbImage.rows == grayImage.rows && rgbImage.cols == grayImage.cols);
    int nBlocks = 4;

    assert(NTHR >= nBlocks);

    int blockSize = rgbImage.cols*(rgbImage.rows/nBlocks);
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        unsigned char *srcPtr = rgbImage.ptr()  + blockID*blockSize*3;
        unsigned char *dstPtr = grayImage.ptr() + blockID*blockSize;

        for (int i = 0; i < blockSize; i++) {
            *dstPtr = (srcPtr[0]*19588 + srcPtr[1]*38469 + srcPtr[2]*7471)>>16;
            dstPtr++; srcPtr+=3;
        }
    }
}

void OMPFunctions::undistort(Mat &src, Mat &dst, float *K, float *iK, float *kc) {
    assert(src.ptr() != NULL && dst.ptr() != NULL && src.rows == dst.rows && src.cols == dst.cols);
    int nBlocks = 6;

    // mark zero into upper-left corner
    unsigned char *ptr = src.ptr();
    ptr[0] = 0;
    ptr[1] = 0;
    ptr[src.cols] = 0;
    ptr[src.cols+1] = 0;

    if (mappingPrecomputed) { undistortLookup(src,dst); return; }

    assert(NTHR >= nBlocks);

    int blockSize = src.cols*(src.rows/nBlocks);
    int width = src.cols;
    int height = src.rows;
    int blockHeight = height/nBlocks;

    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++)  {
   //     printf("blockID: %d\n",blockID);
        int offset = blockID*blockSize;
        unsigned char *srcPtr = src.ptr();
        unsigned char *dstPtr = dst.ptr();
        int y0 = blockID*blockHeight;
        float pu[2],pd[2],p[2],r2,r4,r6,radialDist;
        for (int yi = y0; yi < y0+blockHeight; yi++) {
           for (int xi = 0; xi < width; xi++,offset++) {
               pu[0] = float(xi); pu[1] = float(yi);
               // normalize point
               pd[0] = iK[0]*pu[0] + iK[1]*pu[1] + iK[2];
               pd[1] = iK[3]*pu[0] + iK[4]*pu[1] + iK[5];
               // define radial displacement
               r2 = (pd[0]*pd[0])+(pd[1]*pd[1]); r4 = r2*r2; r6 = r4 * r2;
               radialDist = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
               pd[0] *= radialDist;
               pd[1] *= radialDist;
               // define sampling point in distorted image
               p[0] = K[0]*pd[0] + K[1]*pd[1] + K[2];
               p[1] = K[3]*pd[0] + K[4]*pd[1] + K[5];

               int xdi = (int)p[0];
               int ydi = (int)p[1];
               if (xdi >= 0 && ydi >= 0 && xdi < width-2 && ydi < height-2) {
                       int srcOffset = xdi+ydi*width;
                       unsigned int fracX = (unsigned int)((p[0]-xdi)*256.0f);
                       unsigned int fracY = (unsigned int)((p[1]-ydi)*256.0f);

                       unsigned char *ptr = &srcPtr[srcOffset];
                       unsigned char i1 = ptr[0]; unsigned char i2 = ptr[1]; ptr += width;
                       unsigned char i4 = ptr[0]; unsigned char i3 = ptr[1];

                       const unsigned int c = fracX * fracY;
                       const unsigned int a = 65536 - ((fracY+fracX)<<8)+c;
                       const unsigned int b = (fracX<<8) - c;
                       const unsigned int d = 65536 - a - b - c;

                       dstPtr[offset] = (a*i1 + b*i2 + c*i3 + d*i4)>>16;
                       dxTable[offset*2+0] = p[0]-xi;
                       dxTable[offset*2+1] = p[1]-yi;
               } else {
                       dstPtr[offset]  = 0;
                       dxTable[offset*2+0] = 0.0f-xi;
                       dxTable[offset*2+1] = 0.0f-yi;
               }
            }
        }
    }
    mappingPrecomputed = true;
}

void OMPFunctions::undistortLookup(Mat &src, Mat &dst)
{
    assert(src.rows == dst.rows && src.cols == dst.cols);
    int nBlocks = 6;
    assert(NTHR >= nBlocks);

    int blockSize = src.cols*(src.rows/nBlocks);
    int width = src.cols;
    int height = src.rows;
    int blockHeight = height/nBlocks;

    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize;
        unsigned char *srcPtr = src.ptr();
        unsigned char *dstPtr = dst.ptr();
        int y0 = blockID*blockHeight;
        float p[2];
        for (int yi = y0; yi < y0+blockHeight; yi++) {
           for (int xi = 0; xi < width; xi++,offset++) {
               p[0] = dxTable[offset*2+0]+xi;
               p[1] = dxTable[offset*2+1]+yi;
               int xdi = int(p[0]);
               int ydi = int(p[1]);
               int srcOffset = xdi+ydi*width;
               unsigned int fracX = (unsigned int)((p[0]-xdi)*256.0f);
               unsigned int fracY = (unsigned int)((p[1]-ydi)*256.0f);
               unsigned char *ptr = &srcPtr[srcOffset];
               unsigned char i1 = ptr[0]; unsigned char i2 = ptr[1]; ptr += width;
               unsigned char i4 = ptr[0]; unsigned char i3 = ptr[1];

               const unsigned int c = fracX * fracY;
               const unsigned int a = 65536 - ((fracY+fracX)<<8)+c;
               const unsigned int b = (fracX<<8) - c;
               const unsigned int d = 65536 - a - b - c;

               dstPtr[offset] = (a*i1 + b*i2 + c*i3 + d*i4)>>16;
            }
        }
    }
}

void OMPFunctions::undistortF(Mat &src, Mat &dst, float *K, float *iK, float *kc) {
    assert(src.ptr() != NULL && dst.ptr() != NULL && src.rows == dst.rows && src.cols == dst.cols);
    int nBlocks = 6;

    // mark zero into upper-left corner
    unsigned char *ptr = src.ptr();
    ptr[0] = 0;
    ptr[1] = 0;
    ptr[src.cols] = 0;
    ptr[src.cols+1] = 0;

    if (mappingPrecomputed) { undistortLookupF(src,dst); return; }

    assert(NTHR >= nBlocks);

    int blockSize = src.cols*(src.rows/nBlocks);
    int width = src.cols;
    int height = src.rows;
    int blockHeight = height/nBlocks;

    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
   //     printf("blockID: %d\n",blockID);
        int offset = blockID*blockSize;
        unsigned char *srcPtr = src.ptr();
        float *dstPtr = (float*)dst.ptr();
        int y0 = blockID*blockHeight;
        float pu[2],pd[2],p[2],r2,r4,r6,radialDist;
        for (int yi = y0; yi < y0+blockHeight; yi++) {
           for (int xi = 0; xi < width; xi++,offset++) {
               pu[0] = float(xi); pu[1] = float(yi);
               // normalize point
               pd[0] = iK[0]*pu[0] + iK[1]*pu[1] + iK[2];
               pd[1] = iK[3]*pu[0] + iK[4]*pu[1] + iK[5];
               // define radial displacement
               r2 = (pd[0]*pd[0])+(pd[1]*pd[1]); r4 = r2*r2; r6 = r4 * r2;
               radialDist = 1 + kc[0]*r2 + kc[1]*r4 + kc[4]*r6;
               pd[0] *= radialDist;
               pd[1] *= radialDist;
               // define sampling point in distorted image
               p[0] = K[0]*pd[0] + K[1]*pd[1] + K[2];
               p[1] = K[3]*pd[0] + K[4]*pd[1] + K[5];

               int xdi = (int)p[0];
               int ydi = (int)p[1];
               if (xdi >= 0 && ydi >= 0 && xdi < width-2 && ydi < height-2) {
                       int srcOffset = xdi+ydi*width;
                       float fracX = p[0]-xdi;
                       float fracY = p[1]-ydi;

                       unsigned char *ptr = &srcPtr[srcOffset];
                       float i1 = float(ptr[0]); float i2 = float(ptr[1]); ptr += width;
                       float i4 = float(ptr[0]); float i3 = float(ptr[1]);

                       const float c = fracX * fracY;
                       const float a = 1-fracY-fracX-c;
                       const float b = fracX-c;
                       const float d = 1-a-b-c;

                       dstPtr[offset] = a*i1 + b*i2 + c*i3 + d*i4;

                       dxTable[offset*2+0] = p[0]-xi;
                       dxTable[offset*2+1] = p[1]-yi;
               } else {
                       dstPtr[offset]  = 0.0f;
                       dxTable[offset*2+0] = 0.0f-xi;
                       dxTable[offset*2+1] = 0.0f-yi;
               }
            }
        }
    }
    mappingPrecomputed = true;
}

void OMPFunctions::undistortLookupF(Mat &src, Mat &dst)
{
    assert(src.rows == dst.rows && src.cols == dst.cols);
    int nBlocks = 6;
    assert(NTHR >= nBlocks);

    int blockSize = src.cols*(src.rows/nBlocks);
    int width = src.cols;
    int height = src.rows;
    int blockHeight = height/nBlocks;

    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize;
        unsigned char *srcPtr = (unsigned char*)src.ptr();
        float *dstPtr = (float*)dst.ptr();
        int y0 = blockID*blockHeight;
        float p[2];
        for (int yi = y0; yi < y0+blockHeight; yi++) {
           for (int xi = 0; xi < width; xi++,offset++) {
               p[0] = dxTable[offset*2+0]+xi;
               p[1] = dxTable[offset*2+1]+yi;
               int xdi = int(p[0]);
               int ydi = int(p[1]);
               int srcOffset = xdi+ydi*width;
               float fracX = p[0]-xdi;
               float fracY = p[1]-ydi;
               unsigned char *ptr = &srcPtr[srcOffset];
               float i1 = float(ptr[0]); float i2 = float(ptr[1]); ptr += width;
               float i4 = float(ptr[0]); float i3 = float(ptr[1]);

               const float c = fracX * fracY;
               const float a = 1-fracY-fracX-c;
               const float b = fracX-c;
               const float d = 1-a-b-c;

               dstPtr[offset] = a*i1 + b*i2 + c*i3 + d*i4;
            }
        }
    }
}

void OMPFunctions::generateZArray(float *zmap, unsigned char *rgbMask,int width, int height,float *stdevImage,int nsamples,float *zArray) {
    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
#pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (int xi = 0; xi < width; xi++,offset++) {
                if (!rgbMask[offset]) continue;
                float z = zmap[offset];
                float distance = stdevImage[offset];
                float zmin = z - distance; if (zmin < 0) zmin = 0;
                float zmax = z + distance;
                float zdelta = (zmax-zmin)/float(nsamples);
                for (int zi = 0; zi < nsamples; zi++) {
                    zArray[offset*nsamples+zi] = zmin+float(zi)*zdelta;
                }
            }
        }
    }
}

inline void extractPose(vector<float> &pose, float *T) {
    for (int i = 0; i < 16; i++) T[i] = pose[i];
}

void bilinearInterpolateCPU(float x, float y, int width, int height, unsigned char *srcPtr, float &colorR, float &colorG, float &colorB)
{
    unsigned int xdi = (unsigned int)x; int ydi = (unsigned int)y;
    if (xdi > width-2 || ydi > height-2) return;
    float fx = x-xdi; float fy = y-ydi;

    int pitch = width*3;
    int offsetR2 = 3*xdi+ydi*pitch;
    int offsetG2 = offsetR2+1;
    int offsetB2 = offsetR2+2;

    float a = (1-fx)*(1-fy);
    float b = fx*(1-fy);
    float c = (1-fx)*fy;
    float d = fx*fy;

    float v0 = (float)srcPtr[offsetR2];       float v1 = (float)srcPtr[offsetR2+3];
    float v2 = (float)srcPtr[offsetR2+pitch]; float v3 = (float)srcPtr[offsetR2+pitch+3];
    colorR = a*v0 + b*v1 + c*v2 + d*v3;

    v0 = (float)srcPtr[offsetG2];       v1 = (float)srcPtr[offsetG2+3];
    v2 = (float)srcPtr[offsetG2+pitch]; v3 = (float)srcPtr[offsetG2+pitch+3];
    colorG = a*v0 + b*v1 + c*v2 + d*v3;

    v0 = (float)srcPtr[offsetB2];       v1 = (float)srcPtr[offsetB2+3];
    v2 = (float)srcPtr[offsetB2+pitch]; v3 = (float)srcPtr[offsetB2+pitch+3];
    colorB = a*v0 + b*v1 + c*v2 + d*v3;
}


void OMPFunctions::normalizeCosts(float *costArray, float *countArray, unsigned char *mask, int width, int height, int nsamples) {
    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
#pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int offset = blockID*blockSize*width;
        for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (int xi = 0; xi < width; xi++,offset++) {
                // skip masked pixels
                if (!mask[offset]) continue;
                // normalize cost volume (some points might have been outside image)
                for (int zi = 0; zi < nsamples; zi++) {
                    int costIndex = offset*nsamples+zi;
                    float N = countArray[costIndex];
                    if (N > 0) {
                        costArray[costIndex] /= N;
                    }
                }
            }
        }
    }
}


void OMPFunctions::generateCostVolume(float *zArray, int width, int height, int nsamples, unsigned char *mask, unsigned char *rgbReference, float *Kir, float *Krgb, float *kcRGB, std::vector<std::vector<float> >  &poseMat, std::vector<cv::Mat *> &neighborImage, float *costArray) {
    float iKir[9]; inverse3x3(Kir,iKir);
    float iKrgb[9];inverse3x3(Krgb,iKrgb);


    float *countArray = new float[width*height*nsamples]; memset(countArray,0,sizeof(float)*width*height*nsamples);
  /*  static int joo = 0;

    //cv::Mat argMinImage(height,width,CV_8UC1);
    cv::Mat argMinImage = imread("scratch/argmin-pattern.png",0); unsigned char *ptr = argMinImage.ptr();

*/
    for (size_t vi = 0; vi != neighborImage.size(); vi++ ) {
        cv::Mat *rgbMat = neighborImage[vi];
        unsigned char *rgb = (unsigned char*)rgbMat->ptr();
        float dstT[16];
        extractPose(poseMat[vi],&dstT[0]);
        //projectInit(Krgb, dstT, P);
/*
        cv::Mat outputImage(height,width,CV_8UC3);
        memcpy(outputImage.ptr(),neighborImage[vi]->ptr(),width*height*3);
*/

        int nBlocks = 4;
        int blockSize = height/nBlocks;
        int blockID = 0;
#pragma omp parallel for private(blockID)
        for (blockID = 0; blockID < nBlocks; blockID++) {
            int offset = blockID*blockSize*width;
            for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
                for (int xi = 0; xi < width; xi++,offset++) {
                    if (!mask[offset]) continue;

                    /*bool printFlag = true;//(xi % 2 == 0) && (yi % 2 == 0);
                    float prevX=0,prevY = 0;
                    if (printFlag) {
                        float v[3],w[3],p2[3];
                        get3DPoint(float(xi),float(yi),zArray[offset*nsamples], iKir, &v[0], &v[1], &v[2]);
                        transformRT3(dstT,v,w); w[0] /= w[2]; w[1] /= w[2]; w[2] = 1;
                        distortPointCPU(w,kcRGB,Krgb,p2); prevX = p2[0]; prevY = p2[1];
                    }*/

                    int rgbOffset = offset*3;
                    float refR = rgbReference[rgbOffset+0];
                    float refG = rgbReference[rgbOffset+1];
                    float refB = rgbReference[rgbOffset+2];

                    for (int zi = 0; zi < nsamples; zi++) {
                        int costIndex = offset*nsamples+zi;
                        float v[3],w[3],p2[3];
                        get3DPoint(float(xi),float(yi),zArray[costIndex], iKir, &v[0], &v[1], &v[2]);
                        transformRT3(dstT,v,w); w[0] /= w[2]; w[1] /= w[2]; w[2] = 1;
                        distortPointCPU(w,kcRGB,Krgb,p2);

                        if (p2[0] >= 0.0f && p2[0] <= (width-1) && p2[1] >= 0.0f && p2[1] <= (height-1)) {
                            float r=0,g=0,b=0;
                            bilinearInterpolateCPU(p2[0], p2[1], width, height,rgb, r,g,b);
                            //if (printFlag && (ptr[offset]!=0)) {

                            //    float dist = fabs(zi-ptr[offset])/float(nsamples/2);
                            //    if (dist > 1.0f) dist = 1.0f; dist = 1.0f-dist;  dist *= dist; dist *= 255;
                            //    cv::line(outputImage, cv::Point(prevX,prevY), cv::Point(p2[0],p2[1]), CV_RGB(dist,dist,dist));
                            //    prevX = p2[0]; prevY = p2[1];
                            //}
                            costArray[costIndex] += fabs(r-refR)+fabs(g-refG)+fabs(b-refB);
                            countArray[costIndex]++;
                        }
                    }

                    //printFlag = false;
               }
            }
        }


/*
        if (joo == 13) {
            char buf[512];
            sprintf(buf,"scratch/sample-pattern%d.png",int(vi));
            imwrite(buf,outputImage);
        }*/

    }

    normalizeCosts(costArray,countArray,mask,width,height,nsamples);

    //joo++;
    delete[] countArray;
}

void OMPFunctions::argMinCost(float *zArray, float *costArray, int width, int height, int nsamples, unsigned char *mask, float *zmap) {
    const int nBlocks = 4;
    int nOpt[nBlocks];
    int nTry[nBlocks];
    int blockSize = height/nBlocks;
    int blockID = 0;
/*
      static int joo = 0;
      cv::Mat outputImage(height,width,CV_8UC1); memset(outputImage.ptr(), 0, width*height); unsigned char *ptr = outputImage.ptr();
*/

#pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        nOpt[blockID] = 0; nTry[blockID] = 0;
        int offset = blockID*blockSize*width;
        for (int yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (int xi = 0; xi < width; xi++,offset++) {
                if (!mask[offset]) { continue; }
                nTry[blockID]++;
                int costIndexBase = offset*nsamples;
                int argMin = 0; float minCost = FLT_MAX;
                for (int i = 0; i < nsamples; i++) {
                    int ci = costIndexBase + i;
                    if (costArray[ci] < minCost) {
                        minCost = costArray[ci];
                        argMin = i;
                    }
                }

                // make sure the optimum is not at the bound (true bound could be outside bounds)
                if (argMin != 0 && argMin != (nsamples-1)) {
                    zmap[offset] = zArray[costIndexBase+argMin];
                    nOpt[blockID]++;
                    //ptr[offset] = argMin;
                }
            }
        }
    }
    int opt = 0, ntry = 0;
    for (int i = 0; i < nBlocks; i++) {opt += nOpt[i]; ntry += nTry[i]; }
    if (ntry > 0) {
        float percent = float(opt)/float(ntry)*100.0f;
        printf("photometric optimization success percent: %3.1f\n",percent);
    }

  /*  if (joo == 13) {
        char buf[512];
        sprintf(buf,"scratch/argmin-pattern.png");
        imwrite(buf,outputImage);
    }
    joo++;*/

}


//rgbReference must be given as argument!
void OMPFunctions::optimizePhotometrically(float *zmap, unsigned char *rgbMask, unsigned char *rgbReference, int width, int height, float *stdevImage, int nsamples, float *Kir, float *Krgb, float *kcRGB, std::vector<std::vector<float> >  &poseMat, std::vector<cv::Mat *> &neighborImage) {

    if (zmap == NULL) { printf("zmap null!\n"); return; }
    if (rgbMask == NULL) { printf("rgb mask null!\n"); return; }
    if (rgbReference == NULL) { printf("rgbReference null!\n"); return; }
    if (stdevImage == NULL) { printf("stdevImage null!\n"); return; }
    if (Kir == NULL) { printf("Kir null!\n"); return; }
    if (Krgb == NULL) { printf("Krgb null!\n"); return; }
    if (kcRGB == NULL) { printf("kcRGB null!\n"); return; }

   // printf("nposes : %d, nneigh: %d\n",int(poseMat.size()), int(neighborImage.size()));
   // printf("w: %d, h: %d, nsamples: %d\n",width,height,nsamples);

    float *zArray    = new float[width*height*nsamples];
    float *costArray = new float[width*height*nsamples];  memset(costArray,0,sizeof(float)*width*height*nsamples);

    generateZArray(zmap,rgbMask,width,height,stdevImage,nsamples,zArray);
    generateCostVolume(zArray,width,height,nsamples,rgbMask,rgbReference,Kir,Krgb, kcRGB, poseMat,neighborImage,costArray);
    argMinCost(zArray,costArray,width,height,nsamples,rgbMask,zmap);

    delete[] zArray;
    delete[] costArray;
}

void OMPFunctions::Jtresidual(float *jacobian, float *residual, int width, int rows, double *b) {
    int nBlocks = 3;
    int blockSize = rows/nBlocks;
    int blockID = 0;
    double dblCnt = (double)width;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int rowOffset = blockID*blockSize*width;
        for (int row = blockID*blockSize; row < (blockID+1)*blockSize; row++) {
            double dot = 0.0f;
            for (int xi = 0; xi < width; xi++,rowOffset++) {
                dot += jacobian[rowOffset] * residual[xi];
            }
            b[row] = dot/dblCnt;
        }
    }
}

void OMPFunctions::AtA6(float *jacobian, int count, double *A) {
    double dblCnt = double(count);
    int xiyi[21*2];
    // enumarate 21 dot product combinations:
    int off = 0;
    for (int j = 0; j < 6; j++) {
        for (int i = j; i < 6; i++,off+=2) {
            xiyi[off+0] = i; xiyi[off+1] = j;
        }
    }

    int nBlocks = 4;
    int blockSize = 20/nBlocks; // remainder 1 is added to last block
    int blockID = 0;

#pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        int remainder = 0;
        if (blockID == nBlocks-1) remainder = 1;
        for (int i = blockID*blockSize; i < (blockID+1)*blockSize+remainder; i++) {
            int xi = xiyi[i*2+0];
            int yi = xiyi[i*2+1];

            float *u = jacobian+yi*count;
            float *v = jacobian+xi*count;
            double dot = 0.0;
            for (int k = 0; k < count; k++) {
                dot += u[k] * v[k];
            }
            dot /= dblCnt;
            A[xi+yi*6] = dot;
            A[xi*6+yi] = dot;
        }
    }
/*
    for (int j = 0; j < rows; j++) {
        for (int i = j; i < rows; i++) {
            dotProduct4(jacobian+j*count, jacobian+i*count, count, A+i+j*6, A+i*6+j);
        }
    }
*/
}

void normalizePointCloudDepths(float *pts, float *ptsWeights, float *sumDepths, float *sumWeights, int totalAmount, int stride) {
    for (int i = 0; i < totalAmount; i++) {
        if (sumWeights[i]>1e-3f) {
            pts[i*stride+2] = (ptsWeights[i]*pts[i*stride+2]+sumDepths[i])/sumWeights[i];
        }
    }
}

inline float dist3(float *p1, float *p2) {
    float dX = (p1[0]-p2[0]); dX *= dX;
    float dY = (p1[1]-p2[1]); dY *= dY;
    float dZ = (p1[2]-p2[2]); dZ *= dZ;
    return dX+dY+dZ;
}

inline float robustweight(float *p1, float *n1, float *p2, float *n2, float depthDistance, float distance2) {
    if (fabs(p1[2]-p2[2]) > depthDistance) return 0.0f;
    float ray[3],dev[3],rayComponent;
    float len = sqrtf(p2[0]*p2[0]+p2[1]*p2[1]+p2[2]*p2[2]);
    ray[0] = p2[0]/len;
    ray[1] = p2[1]/len;
    ray[2] = p2[2]/len;
    rayComponent = ray[0]*p1[0]+ray[1]*p1[1]+ray[2]*p1[2];
    ray[0] *= rayComponent;
    ray[1] *= rayComponent;
    ray[2] *= rayComponent;
    dev[0] = p1[0] - ray[0];
    dev[1] = p1[1] - ray[1];
    dev[2] = p1[2] - ray[2];
    float deviation = dev[0]*dev[0]+dev[1]*dev[1]+dev[2]*dev[2];
    if (deviation > distance2) return 0.0f;
    float dot = n1[0]*n2[0]+n1[1]*n2[1]+n1[2]*n2[2];
    return (1.0f - (deviation / distance2))*dot;
}

inline float distray3(float *p1, float *p2) {
    if (fabs(p1[2]-p2[2]) > 150.0f) return FLT_MAX;
    float ray[3],dev[3],rayComponent;
    float len = sqrtf(p2[0]*p2[0]+p2[1]*p2[1]+p2[2]*p2[2]);
    ray[0] = p2[0]/len;
    ray[1] = p2[1]/len;
    ray[2] = p2[2]/len;
    rayComponent = ray[0]*p1[0]+ray[1]*p1[1]+ray[2]*p1[2];
    ray[0] *= rayComponent;
    ray[1] *= rayComponent;
    ray[2] *= rayComponent;
    dev[0] = p1[0] - ray[0];
    dev[1] = p1[1] - ray[1];
    dev[2] = p1[2] - ray[2];
    return dev[0]*dev[0]+dev[1]*dev[1]+dev[2]*dev[2];
}

inline float invdistray(float *p3, float refX, float refY, float *iK) {
    float ray[3],dev[3],rayComponent,rayDist2;
    get3DRay(refX,refY,iK,&ray[0],&ray[1],&ray[2]);
    rayComponent = ray[0]*p3[0]+ray[1]*p3[1]+ray[2]*p3[2];
    ray[0] *= rayComponent;
    ray[1] *= rayComponent;
    ray[2] *= rayComponent;
    dev[0] = p3[0] - ray[0];
    dev[1] = p3[1] - ray[1];
    dev[2] = p3[2] - ray[2];
    rayDist2 = dev[0]*dev[0]+dev[1]*dev[1]+dev[2]*dev[2];
    if (rayDist2 > 100.0) rayDist2 = 100.0f;
    return 1-rayDist2*0.01f;
//    if (rayDist2 > 1600.0) rayDist2 = 1600.0f;
//    return 1-rayDist2*0.000625f;
}

inline float invdist2(float *p1, float x, float y) {
    float dist = 0;
    float dX = (p1[0]-x); dist += dX*dX;
    float dY = (p1[1]-y); dist += dY*dY;
    if (dist > 2.0) dist = 2.0f;
    return 1-dist*0.5f;
}

void OMPFunctions::refineDepthMap(Mat &xyzCur, Mat &weightsCur, float *K,  float *T, Mat &xyzRef, Mat &weightsRef, int stride, float depthThreshold, float distanceThreshold) {
    int width  = xyzCur.cols; int height = xyzCur.rows;
    int dstWidth     = xyzRef.cols; int dstHeight    = xyzRef.rows;
    int dstWidthP = dstWidth*stride;
    float *srcPts        = (float*)xyzCur.ptr();
    float *dstPts        = (float*)xyzRef.ptr();
    const int nBlocks = 4;
    float iK[9]; inverse3x3(K,&iK[0]);
    Mat weightImages[nBlocks]; for (int i = 0; i < nBlocks; i++) { weightImages[i] = cv::Mat(dstHeight,dstWidth,CV_32FC1); memset(weightImages[i].ptr(),0,sizeof(float)*dstWidth*dstHeight); }
    Mat fusedDepths[nBlocks];  for (int i = 0; i < nBlocks; i++) { fusedDepths[i]  = cv::Mat(dstHeight,dstWidth,CV_32FC1); memset(fusedDepths[i].ptr(),0,sizeof(float)*dstWidth*dstHeight); }
    // add previous weights into slot 0
    memcpy(weightImages[0].ptr(),weightsRef.ptr(),sizeof(float)*dstWidth*dstHeight);

    float distanceThreshold2 = distanceThreshold*distanceThreshold;
    int blockSize = height/nBlocks;
    int blockID = 0;
#pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        // each block has a separate scratch memories to avoid collisions
        float *weightedDepths   = (float*)fusedDepths[blockID].ptr();
        float *weights    = (float*)weightImages[blockID].ptr();
   //     float *priorWeights    = (float*)weightsCur.ptr();

        // determine start offsets for a image block :
        int offset = blockID*blockSize*width;
        int offsetp = offset*stride;

        for (float yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (float xi = 0; xi < width; xi++,offset++,offsetp+=stride) {
                if (srcPts[offsetp+6] == 0.0f) continue;
                // transform and project point
                float *srcP = &srcPts[offsetp]; float w[2],tP[3],p2[2];
                transformRT3(T,srcP,&tP[0]); w[0] = tP[0]/tP[2]; w[1] = tP[1]/tP[2];
                p2[0] = K[0]*w[0]+K[1]*w[1]+K[2];
                p2[1] = K[3]*w[0]+K[4]*w[1]+K[5];
                // check image bounds
                if (p2[0] >= 0 && p2[0] < (dstWidth-1) && p2[1] >= 0 && p2[1] < (dstHeight-1)) {
                    // also fetch normal for weights
                    float *n = &srcPts[offsetp+3]; float srcN[3];
                    rotate3(T,n,&srcN[0]);
  //                  float priorWeight = priorWeights[offset];
                    float priorWeight = 2.0f;
                    // determine image offset (it also applies to 3d point array )
                    int xi = (int)p2[0]; int yi = (int)p2[1];
                    int off = xi+yi*dstWidth;
                    int offp = off*stride;
                    float newZ = tP[2];
                    // compute bi-linear weights for the surrounding pixels
                    //float fx = p2[0]-xi;      float fy = p2[1]-yi;
                    // skip invalid pixels in the destination map:

                    float *dp0 = &dstPts[offp];
                    float *dp1 = &dstPts[offp+stride];
                    float *dp2 = &dstPts[offp+dstWidthP];
                    float *dp3 = &dstPts[offp+dstWidthP+stride];
                    if (dp0[6] == 0) {
                        get3DPoint(float(xi),float(yi),-tP[2],iK,     &dp0[0], &dp0[1], &dp0[2]); dp0[3] = srcN[0]; dp0[4] = srcN[1]; dp0[5] = srcN[2]; dp0[6] = 1;
                    }
                    if (dp1[6] == 0) {
                        get3DPoint(float(xi+1),float(yi),-tP[2],iK,   &dp1[0], &dp1[1], &dp1[2]); dp1[3] = srcN[0]; dp1[4] = srcN[1]; dp1[5] = srcN[2]; dp1[6] = 1;
                    }
                    if (dp2[6] == 0) {
                        get3DPoint(float(xi),float(yi+1),-tP[2],iK,   &dp2[0], &dp2[1], &dp2[2]); dp2[3] = srcN[0]; dp2[4] = srcN[1]; dp2[5] = srcN[2]; dp2[6] = 1;
                    }
                    if (dp3[6] == 0) {
                        get3DPoint(float(xi+1),float(yi+1),-tP[2],iK, &dp3[0], &dp3[1], &dp3[2]); dp3[3] = srcN[0]; dp3[4] = srcN[1]; dp3[5] = srcN[2]; dp3[6] = 1;
                    }

                    float w = robustweight(&tP[0],&srcN[0],dp0,&dp0[3],depthThreshold,distanceThreshold2);
                    if (w > 0) {
                        w *= priorWeight;
                        float *depth  = &weightedDepths[off];
                        float *weight = &weights[off];
                        // accumulate depth values for pixels involved
                        depth[0]   += w*newZ;
                        weight[0]  += w;
                    }
                    w = robustweight(&tP[0],&srcN[0],dp1,&dp1[3],depthThreshold,distanceThreshold2);
                    if (w > 0) {
                        w *= priorWeight;
                        float *depth  = &weightedDepths[off+1];
                        float *weight = &weights[off+1];
                        // accumulate depth values for pixels involved
                        depth[0]   += w*newZ;
                        weight[0]  += w;
                    }
                    w = robustweight(&tP[0],&srcN[0],dp2,&dp2[3],depthThreshold,distanceThreshold2);
                    if (w > 0) {
                        w *= priorWeight;
                        float *depth = &weightedDepths[off+dstWidth];
                        float *weight = &weights[off+dstWidth];
                        // accumulate depth values for pixels involved
                        depth[0]   += w*newZ;
                        weight[0]  += w;
                    }
                    w = robustweight(&tP[0],&srcN[0],dp3,&dp3[3],depthThreshold,distanceThreshold2);
                    if ( w > 0.0f) {
                        w *= priorWeight;
                        float *depth = &weightedDepths[off+dstWidth+1];
                        float *weight = &weights[off+dstWidth+1];
                        // accumulate depth values for pixels involved
                        depth[0]   += w*newZ;
                        weight[0]  += w;
                    }
                }
            }
        }
    }

    // accumulate weights and weighted depths into slot 0
    float *newDepthSum     = (float*)fusedDepths[0].ptr();
    float *totalWeightSum  = (float*)weightImages[0].ptr();
    for (int i = 1; i < nBlocks; i++) {
        float *weightedDepths   = (float*)fusedDepths[i].ptr();
        float *weights          = (float*)weightImages[i].ptr();
        for (int j = 0; j < dstWidth*dstHeight; j++) {
            if (weights[j] > 0.0f) {
                totalWeightSum[j] += weights[j];
                newDepthSum[j]    += weightedDepths[j];
            }
        }
    }
    normalizePointCloudDepths(dstPts,(float*)weightsRef.ptr(),newDepthSum,totalWeightSum,dstWidth*dstHeight,stride);
    // update reference weights
    weightImages[0].copyTo(weightsRef);
}

double OMPFunctions::residualPhotometric(Mat &xyz,Mat &selection, int nPoints, float *kc, float *K, float *TLR, float *T, cv::Mat &grayRef, float* residual, float *jacobian, float *wjacobian, int layer, float intensityThreshold, int stride) {
    int dstWidth  = grayRef.cols; int dstHeight = grayRef.rows;

    float *srcPts   = (float*)xyz.ptr();    int *selectionIndex = (int*)selection.ptr();
    float *grayData = (float*)grayRef.ptr();

    const int nBlocks = 4;
    int blockSize = nPoints/nBlocks;
    int blockID = 0;
    double errorBlocks[nBlocks];  for (int i = 0; i < nBlocks; i++) errorBlocks[i] = 0;
    int    samplesBlock[nBlocks]; for (int i = 0; i < nBlocks; i++) samplesBlock[i] = 0;

#pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        // determine start offsets for a image block :
        int si = blockID*blockSize;
        double *errors = &errorBlocks[blockID];
        int *samples   = &samplesBlock[blockID];
        for (int xi = 0; xi < blockSize; xi++,si++) {
            int offsetp = selectionIndex[si]*stride;
            // transform and project point
            float *srcP = &srcPts[offsetp]; float sP[3],tP[3],pu[2],p2[2];
            transformRT3(T,srcP,&tP[0]);
            transformRT3(TLR,&tP[0],&sP[0]);
            pu[0] = sP[0]/sP[2]; pu[1] = sP[1]/sP[2];
            distortPointCPU(&pu[0],kc,K,&p2[0]);

            float photoError = 64;
            float weight = 0;
            // check image bounds
            if (p2[0] >= 0 && p2[0] < (dstWidth-1) && p2[1] >= 0 && p2[1] < (dstHeight-1)) {
                // determine image offset (it also applies to 3d point array )
                int xi = (int)p2[0]; int yi = (int)p2[1];
                int off = xi+yi*dstWidth;
                // compute filtered gray value
                float fx = p2[0]-xi; float fy = p2[1]-yi;
                float a = (1-fx)*(1-fy);
                float b = fx*(1-fy);
                float c = (1-fx)*fy;
                float d = fx*fy;

                float v0 = grayData[off];          float v1 = grayData[off+1];
                float v2 = grayData[off+dstWidth]; float v3 = grayData[off+dstWidth+1];
                float grayIntensity = a*v0 + b*v1 + c*v2 + d*v3;
                photoError = srcPts[offsetp+8+layer] - grayIntensity;

                // the weights become quadratic when they are powered to two in final estimation formula (JW'WJ)^TJ^T(W'W)r
                float absError = fabs(photoError);
                // damp large deviations :
                if (absError < intensityThreshold) {
                    weight = 1.0f-absError/intensityThreshold;
                }
                *errors += absError;
                *samples += 1;
            }
            residual[si] = photoError*weight;
            wjacobian[0*nPoints+si] = jacobian[0*nPoints+si]*weight;
            wjacobian[1*nPoints+si] = jacobian[1*nPoints+si]*weight;
            wjacobian[2*nPoints+si] = jacobian[2*nPoints+si]*weight;
            wjacobian[3*nPoints+si] = jacobian[3*nPoints+si]*weight;
            wjacobian[4*nPoints+si] = jacobian[4*nPoints+si]*weight;
            wjacobian[5*nPoints+si] = jacobian[5*nPoints+si]*weight;
        }
    }
    double totalError = 0.0f;
    int totalSamples = 0;
    for (int i = 0; i < nBlocks; i++) { totalError += errorBlocks[i]; totalSamples += samplesBlock[i]; }
//    printf("photoerror: %e %d",totalError,totalSamples);
    return totalError/double(totalSamples+1);
}



double OMPFunctions::residualICP(Mat &xyzRef, Mat &maskRef, float *K, float *T, Mat &xyzCur, float *residual, float *jacobian, float scaleIn, float depthThreshold, int stride) {
    int width     = xyzRef.cols; int height    = xyzRef.rows; int residualSize = width*height;
    int dstWidth  = xyzCur.cols; int dstHeight = xyzCur.rows;
    int dstWidthP = dstWidth*stride;

    // src dimension check
    assert(width == maskRef.cols && height == maskRef.rows);
    float *srcPts        = (float*)xyzRef.ptr();     unsigned char *srcMask = (unsigned char*)maskRef.ptr();
    float *dstPts        = (float*)xyzCur.ptr();

    const int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    double errorPerBlock[nBlocks];          memset(&errorPerBlock[0],0,sizeof(double)*nBlocks);
    unsigned int nSamplesPerBlock[nBlocks]; memset(&nSamplesPerBlock[0],0,sizeof(int)*nBlocks);
    memset(residual,0,sizeof(float)*residualSize);
    memset(jacobian,0,sizeof(float)*residualSize*6);

    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        // determine start offsets for a image block :
        int offset = blockID*blockSize*width;
        int offset3 = offset*3;
        int offsetp = offset*stride;
        for (float yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (float xi = 0; xi < width; xi++,offset++,offset3+=3,offsetp+=stride) {
                // transform and project point
                float *srcP = &srcPts[offsetp]; float w[2],tP[3],dP[3],dB[3],p2[2];
                transformRT3(T,srcP,&tP[0]); w[0] = tP[0]/tP[2]; w[1] = tP[1]/tP[2];
                p2[0] = K[0]*w[0]+K[1]*w[1]+K[2];
                p2[1] = K[3]*w[0]+K[4]*w[1]+K[5];
                // check image bounds
                if (p2[0] >= 0 && p2[0] <= (dstWidth-1) && p2[1] >= 0 && p2[1] <= (dstHeight-1) && srcMask[offset] > 0) {
                    // determine image offset (it also applies to 3d point array )
                    int xi = (int)p2[0]; int yi = (int)p2[1];
                    int off = xi+yi*dstWidth;
                    int offp = off*stride;
                    // skip invalid pixels in the destination map:
                    if (dstPts[offp+6] > 0 && dstPts[offp+stride+6] > 0 && dstPts[offp+dstWidthP+6] > 0 && dstPts[offp+dstWidthP+stride+6] > 0)
                    {
                        // compute bi-linear weights for the surrounding pixels
                        float fx = p2[0]-xi;      float fy = p2[1]-yi;
                        float w0 = (1-fx)*(1-fy); float w1 =  fx*(1-fy);
                        float w2 = (1-fx)*fy;     float w3 = fx*fy;

                        // interpolate matching point
                        float *dstP0 = &dstPts[offp];
                        float *dstP1 = &dstPts[offp+stride];
                        float *dstP2 = &dstPts[offp+dstWidthP];
                        float *dstP3 = &dstPts[offp+dstWidthP+stride];

                        float matchP[3] = {0,0,0};
                        matchP[0] = w0 * dstP0[0] + w1 * dstP1[0] + w2 * dstP2[0] + w3 * dstP3[0];
                        matchP[1] = w0 * dstP0[1] + w1 * dstP1[1] + w2 * dstP2[1] + w3 * dstP3[1];
                        matchP[2] = w0 * dstP0[2] + w1 * dstP1[2] + w2 * dstP2[2] + w3 * dstP3[2];

                        // interpolate matching normal
                        float *dstN0 = &dstPts[offp+3];
                        float *dstN1 = &dstPts[offp+stride+3];
                        float *dstN2 = &dstPts[offp+dstWidthP+3];
                        float *dstN3 = &dstPts[offp+dstWidthP+stride+3];

                        float matchN[3] = {0,0,0};
                        matchN[0] = w0 * dstN0[0] + w1 * dstN1[0] + w2 * dstN2[0] + w3 * dstN3[0];
                        matchN[1] = w0 * dstN0[1] + w1 * dstN1[1] + w2 * dstN2[1] + w3 * dstN3[1];
                        matchN[2] = w0 * dstN0[2] + w1 * dstN1[2] + w2 * dstN2[2] + w3 * dstN3[2];

                        // re-normalize
                        float len = sqrt(matchN[0]*matchN[0]+matchN[1]*matchN[1]+matchN[2]*matchN[2]);
                        matchN[0] /= len; matchN[1] /= len; matchN[2] /= len;

                        float error = (matchP[0] - tP[0])*matchN[0] + (matchP[1]-tP[1])*matchN[1] + (matchP[2]-tP[2])*matchN[2];

                        // the weights become quadratic when they are powered to two in final estimation formula (JW'WJ)^TJ^T(W'W)r
                        float absError = fabs(error);
                        // cumulate statistics:
                        errorPerBlock[blockID] += absError;
                        nSamplesPerBlock[blockID]++;
                        // damp large deviations :
                        if (absError < depthThreshold) {
                            float weight = 1.0f-absError/depthThreshold;
                            // scale units to meters at this stage
                            residual[offset]  = error*weight*scaleIn;
                            dP[0] = 0; dP[1] = -srcP[2]; dP[2] = srcP[1];
                            rotate3(T,&dP[0],&dB[0]);
                            jacobian[offset+residualSize*0]  = weight*(matchN[0]*dB[0]+matchN[1]*dB[1]+matchN[2]*dB[2])*scaleIn;
                            dP[0] = srcP[2]; dP[1] = 0; dP[2] = -srcP[0];
                            rotate3(T,&dP[0],&dB[0]);
                            jacobian[offset+residualSize*1]  = weight*(matchN[0]*dB[0]+matchN[1]*dB[1]+matchN[2]*dB[2])*scaleIn;
                            dP[0] = -srcP[1]; dP[1] = srcP[0]; dP[2] = 0;
                            rotate3(T,&dP[0],&dB[0]);
                            jacobian[offset+residualSize*2]  = weight*(matchN[0]*dB[0]+matchN[1]*dB[1]+matchN[2]*dB[2])*scaleIn;
                            dP[0] = 1; dP[1] = 0; dP[2] = 0;
                            rotate3(T,&dP[0],&dB[0]);
                            jacobian[offset+residualSize*3]  = weight*(matchN[0]*dB[0]+matchN[1]*dB[1]+matchN[2]*dB[2]);
                            dP[0] = 0; dP[1] = 1; dP[2] = 0;
                            rotate3(T,&dP[0],&dB[0]);
                            jacobian[offset+residualSize*4]  = weight*(matchN[0]*dB[0]+matchN[1]*dB[1]+matchN[2]*dB[2]);
                            dP[0] = 0; dP[1] = 0; dP[2] = 1;
                            rotate3(T,&dP[0],&dB[0]);
                            jacobian[offset+residualSize*5]  = weight*(matchN[0]*dB[0]+matchN[1]*dB[1]+matchN[2]*dB[2]);
                        }
                    }
                }
            }
        }
    }
    double   fullError = 0.0f;
    unsigned int nTotalSamples = 0;
    for (int i = 0; i < nBlocks; i++) { fullError += errorPerBlock[i]; nTotalSamples += nSamplesPerBlock[i]; }
    fullError /=  (nTotalSamples+1); // avoid division by zero
    if (nTotalSamples < 1024) return DBL_MAX;
    else return fullError;
}

void OMPFunctions::downSamplePointCloud(cv::Mat &hiresXYZ, cv::Mat &lowresXYZ, int stride) {
    int width     = lowresXYZ.cols;   int height    = lowresXYZ.rows;
    int dstWidth  = hiresXYZ.cols;    //int dstHeight = hiresXYZ.rows;

    float *srcPts          = (float*)lowresXYZ.ptr();
    float *dstPts          = (float*)hiresXYZ.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        // determine start offsets for a image block :
        int offset = blockID*blockSize*width;
        int offset3 = offset*3;
        int offsetp = offset*stride;
        int dwp = dstWidth*stride;
        for (float yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (float xi = 0; xi < width; xi++,offset++,offset3+=3,offsetp+=stride) {
                int hiresOffset = xi*2 + (yi*2)*dstWidth;
                int hiresOffsetP = hiresOffset*stride;

                float point[3] = {0,0,0};
                float normal[3] = {1,0,0};
                float maskValue = 0.0f;
                float grayIntensity = 0.0f;
                float gradMag = 0.0f;

                if ( (dstPts[hiresOffsetP+6] > 0)  && (dstPts[hiresOffsetP+stride+6] > 0) && (dstPts[hiresOffsetP+dwp+6] > 0) && (dstPts[hiresOffsetP+dwp+stride+6] > 0) ) {
                    point[0]   = (dstPts[hiresOffsetP+0] + dstPts[hiresOffsetP+stride+0] + dstPts[hiresOffsetP+0+dwp] + dstPts[hiresOffsetP+stride+0+dwp])/4.0f;
                    point[1]   = (dstPts[hiresOffsetP+1] + dstPts[hiresOffsetP+stride+1] + dstPts[hiresOffsetP+1+dwp] + dstPts[hiresOffsetP+stride+1+dwp])/4.0f;
                    point[2]   = (dstPts[hiresOffsetP+2] + dstPts[hiresOffsetP+stride+2] + dstPts[hiresOffsetP+2+dwp] + dstPts[hiresOffsetP+stride+2+dwp])/4.0f;
                    normal[0]  = (dstPts[hiresOffsetP+3] + dstPts[hiresOffsetP+stride+3] + dstPts[hiresOffsetP+3+dwp] + dstPts[hiresOffsetP+stride+3+dwp])/4.0f;
                    normal[1]  = (dstPts[hiresOffsetP+4] + dstPts[hiresOffsetP+stride+4] + dstPts[hiresOffsetP+4+dwp] + dstPts[hiresOffsetP+stride+4+dwp])/4.0f;
                    normal[2]  = (dstPts[hiresOffsetP+5] + dstPts[hiresOffsetP+stride+5] + dstPts[hiresOffsetP+5+dwp] + dstPts[hiresOffsetP+stride+5+dwp])/4.0f;
                    grayIntensity  = (dstPts[hiresOffsetP+7] + dstPts[hiresOffsetP+stride+7] + dstPts[hiresOffsetP+7+dwp] + dstPts[hiresOffsetP+stride+7+dwp])/4.0f;
                   // gradMag        = (dstPts[hiresOffsetP+8] + dstPts[hiresOffsetP+stride+8] + dstPts[hiresOffsetP+8+dwp] + dstPts[hiresOffsetP+stride+8+dwp])/4.0f;
                    // re-normalize
                    float len = sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
                    normal[0] /= len; normal[1] /= len; normal[2] /= len;
                    maskValue = 255.0f;
                }
                srcPts[offsetp+0]     = point[0];
                srcPts[offsetp+1]     = point[1];
                srcPts[offsetp+2]     = point[2];
                srcPts[offsetp+3]     = normal[0];
                srcPts[offsetp+4]     = normal[1];
                srcPts[offsetp+5]     = normal[2];
                srcPts[offsetp+6]     = maskValue;
                srcPts[offsetp+7]     = gradMag;
                srcPts[offsetp+8]     = grayIntensity;
                srcPts[offsetp+9]     = 0;
                srcPts[offsetp+10]    = 0;

            }
        }
    }
}


void OMPFunctions::downSampleMask(cv::Mat &hiresMask, cv::Mat &lowresMask) {
    int width     = lowresMask.cols;   int height   = lowresMask.rows;
    int dstWidth  = hiresMask.cols;   int dstHeight = hiresMask.rows;

    unsigned char *srcMask = (unsigned char*)lowresMask.ptr();
    unsigned char *dstMask = (unsigned char*)hiresMask.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        // determine start offsets for a image block :
        int offset = blockID*blockSize*width;
        for (float yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (float xi = 0; xi < width; xi++,offset++) {

                int hiresOffset = xi*2 + (yi*2)*dstWidth;
                int   maskValue = 0;
                if ( (dstMask[hiresOffset+0] > 0) && (dstMask[hiresOffset+1] > 0) && (dstMask[hiresOffset+0+dstWidth] > 0) && (dstMask[hiresOffset+1+dstWidth] > 0) ) {
                    maskValue = 255;
                }
                srcMask[offset]       = maskValue;
            }
        }
    }
}

void OMPFunctions::downSampleHdrImage(cv::Mat &hiresImage, cv::Mat &lowresImage) {
    int width     = lowresImage.cols;   int height   = lowresImage.rows;
    int dstWidth  = hiresImage.cols;

    float *src = (float*)lowresImage.ptr();
    float *dst = (float*)hiresImage.ptr();

    int nBlocks = 4;
    int blockSize = height/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        // determine start offsets for a image block :
        int offset = blockID*blockSize*width;
        for (float yi = blockID*blockSize; yi < (blockID+1)*blockSize; yi++) {
            for (float xi = 0; xi < width; xi++,offset++) {
                int hiresOffset = xi*2 + (yi*2)*dstWidth;
                src[offset]       = (dst[hiresOffset+0] + dst[hiresOffset+1] + dst[hiresOffset+0+dstWidth] + dst[hiresOffset+1+dstWidth])/4.0f;
            }
        }
    }
}


void OMPFunctions::generateOrientedPoints(cv::Mat &depthCPU, cv::Mat &xyzImage, float *KL, cv::Mat &normalStatus, float *kc, float *KR, float *TLR, cv::Mat &grayImage, int stride)
{
    int fw = depthCPU.cols;
    int fh = depthCPU.rows;
    assert(fw == xyzImage.cols && fh == xyzImage.rows);

    float *srcData  = (float*)depthCPU.ptr();
    float *pData    = (float*)xyzImage.ptr();
    float *grayData = (float*)grayImage.ptr();
   // unsigned char *mask = (unsigned char*)maskImage.ptr();      memset(mask,0,fw*fh);
    unsigned char *status = (unsigned char*)normalStatus.ptr(); memset(status,0,fw*fh);

    float iKir[9]; inverse3x3(&KL[0],&iKir[0]);

    int nBlocks = 4;
    int blockSize = fh/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        // if this is the last block, do not process the last row :
        int ydec = 0; if (blockID == nBlocks-1) ydec = 1;
        int yinc = 0; if (blockID == 0) yinc = 1;
        for (float yi = blockID*blockSize+yinc; yi < (blockID+1)*blockSize-ydec; yi++) {
            for (float xi = 1; xi < (fw-1); xi++) {
                int offset = xi+yi*fw;
                int offsetp = offset*stride;
                float z    = srcData[offset];
                float zNu1 = srcData[offset+1];
                float zNv1 = srcData[offset+fw];
                float zNu0 = srcData[offset-1];
                float zNv0 = srcData[offset-fw];

                // detect z-dynamics (are we at edge?)
                float minZ  = z; if (zNu1 < minZ) minZ = zNu1;  if (zNu0 < minZ) minZ = zNu0; if (zNv1 < minZ) minZ = zNv1; if (zNv0 < minZ) minZ = zNv0;
                float maxZ  = z; if (zNu1 > maxZ) maxZ = zNu1;  if (zNu0 > maxZ) maxZ = zNu0; if (zNv1 > maxZ) maxZ = zNv1; if (zNv0 > maxZ) maxZ = zNv0;
                float threshold = 250.0f;

                if (fabs(maxZ - minZ) < 100.0f && (minZ > threshold)) {
                    float p[3],u1[3],u0[3],v1[3],v0[3];
                    get3DPoint(float(xi),float(yi),z,iKir, &p[0], &p[1], &p[2]);

                    get3DPoint(float(xi+1),float(yi),zNu1,iKir, &u1[0], &u1[1], &u1[2]);
                    get3DPoint(float(xi-1),float(yi),zNu0,iKir, &u0[0], &u0[1], &u0[2]);
                    get3DPoint(float(xi),float(yi+1),zNv1,iKir, &v1[0], &v1[1], &v1[2]);
                    get3DPoint(float(xi),float(yi-1),zNv0,iKir, &v0[0], &v0[1], &v0[2]);

                    float nu[3],nv[3],n[3];
                    nu[0] = u1[0] - u0[0]; nu[1] = u1[1] - u0[1]; nu[2] = u1[2] - u0[2];
                    nv[0] = v1[0] - v0[0]; nv[1] = v1[1] - v0[1]; nv[2] = v1[2] - v0[2];
                    // compute normal as crossproduct
                    n[0] =  nu[1] * nv[2] - nu[2] * nv[1];
                    n[1] =-(nu[0] * nv[2] - nu[2] * nv[0]);
                    n[2] =  nu[0] * nv[1] - nu[1] * nv[0];
                    // normal to unit length
                    float len = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]+1e-5f);
                    // TODO: use this magnitude (area of square) to prune out invalid normals (mismatch in depth)
                    n[0] /= len; n[1] /= len; n[2] /= len;

                    // compute area code for point selection
                    unsigned char statusIndex = 0; float maxDot = 0; float dot;
                    dot = -1.0*n[0]; if (dot > maxDot) { maxDot = dot; statusIndex = 1; }
                    dot =  1.0*n[0]; if (dot > maxDot) { maxDot = dot; statusIndex = 2; }
                    dot = -1.0*n[1]; if (dot > maxDot) { maxDot = dot; statusIndex = 3; }
                    dot =  1.0*n[1]; if (dot > maxDot) { maxDot = dot; statusIndex = 4; }
                    dot = -1.0*n[2]; if (dot > maxDot) { maxDot = dot; statusIndex = 5; }
                    int areaCodeX = 4*xi/fw;
                    int areaCodeY = 4*yi/fh;
                    int areaStatus = areaCodeX+areaCodeY*4;
                    // store normal direction index for oriented histograms
                    status[offset] = statusIndex+areaStatus*6;

                    // sample gray value and its gradient magnitude:
                    float magGrad=0,grayIntensity=0;
                    float r3[3],p2[3];
                    transformRT3(TLR, &p[0], r3); r3[0] /= r3[2]; r3[1] /= r3[2]; r3[2] = 1.0f;
                    // p2: distorted point
                    distortPointCPU(r3,kc,KR,p2);
                    int xi = (int)p2[0];
                    int yi = (int)p2[1];
                    if (xi > 0 && yi > 0 && xi < (fw-1) && yi < (fh-1)) {
                        int offset2 = xi + yi * fw;
                        // compute gradient magnitude
                        float gx0 =  grayData[offset2-1];
                        float gx1 =  grayData[offset2+1];
                        float gy0 =  grayData[offset2-fw];
                        float gy1 =  grayData[offset2+fw];
                        float dx = gx1-gx0;
                        float dy = gy1-gy0;
                        magGrad  = fabs(dx)+fabs(dy)*256.0; // encode dx and dy into a single float

                        // compute filtered gray value
                        float fx = p2[0]-xi; float fy = p2[1]-yi;
                        float a = (1-fx)*(1-fy);
                        float b = fx*(1-fy);
                        float c = (1-fx)*fy;
                        float d = fx*fy;

                        float v0 = grayData[offset2]; float v1 = gx1;
                        float v2 = gy1;               float v3 = grayData[offset2+fw+1];
                        grayIntensity = a*v0 + b*v1 + c*v2 + d*v3;
                    }
                    pData[offsetp+0]  = p[0];
                    pData[offsetp+1]  = p[1];
                    pData[offsetp+2]  = p[2];
                    pData[offsetp+3]  = -n[0];
                    pData[offsetp+4]  = -n[1];
                    pData[offsetp+5]  = -n[2];
                    pData[offsetp+6]  = 1.0f; // label this point and its neighbors valid
                    pData[offsetp+7]  = magGrad;
                    pData[offsetp+8]  = grayIntensity;
                    pData[offsetp+9]  = 0.0f;
                    pData[offsetp+10] = 0.0f;
                } else {
                    pData[offsetp+0]  = 0.0f;
                    pData[offsetp+1]  = 0.0f;
                    pData[offsetp+2]  = 0.0f;
                    pData[offsetp+3]  = 0.0f;
                    pData[offsetp+4]  = 0.0f;
                    pData[offsetp+5]  = 0.0f;
                    pData[offsetp+6]  = 0.0f; // label this point and its neighbors valid
                    pData[offsetp+7]  = 0.0f;
                    pData[offsetp+8]  = 0.0f;
                    pData[offsetp+9]  = 0.0f;
                    pData[offsetp+10] = 0.0f;
                }
            }
        }
    }
}

// this takes into account bilinear filtering bounds
inline bool inBounds(float *p, int w, int h) {
    if ((p[0] >= 5) && (p[0] < w-5) && (p[1] >= 5) && (p[1] < h-5)) return true;
    return false;
}

// assume K is scaled according to gray image layer
void OMPFunctions::precomputePhotoJacobians(cv::Mat &xyzCur,float *kc, float *K, float *TLR, cv::Mat &gray, int nPoints, cv::Mat &photometricSelection, int stride, cv::Mat &photoJacobian, int layer, float scaleIn)  {
    int dstW = gray.cols;
    int dstH = gray.rows;

    float *ptsCur   = (float*)xyzCur.ptr();
    float *grayData = (float*)gray.ptr();
    int   *select   = (int*)photometricSelection.ptr();
    float *jacobian = (float*)photoJacobian.ptr();

    int nBlocks = 4;
    int blockSize = nPoints/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        for (int si = blockID*blockSize; si < (blockID+1)*blockSize; si++) {
            int j = select[si];
            float *pOriginal = &ptsCur[j*stride+0]; float r[3],pu[3],pd[3];
            transformRT3(TLR,pOriginal,r);
            pu[0] = r[0] / r[2]; pu[1] = r[1] / r[2]; pu[2] = 1.0f;
            distortPointCPU(&pu[0],&kc[0],&K[0],&pd[0]);

            bool pointsOnScreen = inBounds(&pd[0],dstW,dstH);
            if (pointsOnScreen) {
                // change coordinate system scale for better numerical properties in optimization
                float p[3];
                p[0] = pOriginal[0]*scaleIn; p[1] = pOriginal[1]*scaleIn; p[2] = pOriginal[2]*scaleIn;
                r[0] *= scaleIn; r[1] *= scaleIn; r[2] *= scaleIn;

                int xdi,ydi;
                float fx,fy;
                float grayVal,grayValN,grayValE,grayValS,grayValW;
                float v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11;
                float a,b,c,d;
                float gradIx,gradIy;
                float dp[3],dr[3],dn[3],dd[3],dp2[2];

                xdi = (int)pd[0]; fx = pd[0] - xdi;
                ydi = (int)pd[1]; fy = pd[1] - ydi;
                int offset = xdi+ydi*dstW;

                // compute bilinear coeffs
                a = (1-fx)*(1-fy);
                b = fx*(1-fy);
                c = (1-fx)*fy;
                d = fx*fy;
                // fetch surrounding pixels
                // X   v0   v1   X
                // v2  v3   v4  v5
                // v6  v7   v8  v9
                // X   v10  v11  X
                v0  = grayData[offset-dstW];   v1  = grayData[offset-dstW+1];
                v2  = grayData[offset-1];      v3  = grayData[offset];          v4 = grayData[offset+1];      v5 = grayData[offset+2];
                v6  = grayData[offset+dstW-1]; v7  = grayData[offset+dstW];     v8 = grayData[offset+dstW+1]; v9 = grayData[offset+dstW+2];
                v10 = grayData[offset+dstW*2]; v11 = grayData[offset+dstW*2+1];

                grayVal   = a*v3 + b*v4 + c*v7  + d*v8;
                // store reference gray value for residual evaluation
                pOriginal[8+layer] = grayVal;

                // filter intensities
                grayValN  = a*v0 + b*v1 + c*v3  + d*v4;
                grayValW  = a*v2 + b*v3 + c*v6  + d*v7;
                grayValE  = a*v4 + b*v5 + c*v8  + d*v9;
                grayValS  = a*v7 + b*v8 + c*v10 + d*v11;

                gradIx = (grayValE-grayValW)/2.0f;
                gradIy = (grayValS-grayValN)/2.0f;

                //	A[0]  = 0;	 A[1]  = -x[2];  A[2]  = x[1];	A[3]  = x[3];
                //	A[4]  = x[2];A[5]  =     0;	 A[6]  =-x[0];	A[7]  = x[4];
                //	A[8]  =-x[1];A[9]  =  x[0];  A[10] =    0;	A[11] = x[5];
                //	A[12] = 0;	 A[13] =     0;	 A[14] =    0;	A[15] =    0;

                float dN[6];
                dN[0] = 1.0f/r[2]; dN[1] =         0; dN[2] = -r[0]/(r[2]*r[2]);
                dN[3] =         0; dN[4] = 1.0f/r[2]; dN[5] = -r[1]/(r[2]*r[2]);

                float x = pu[0];     float y = pu[1];
                float x2 = x*x;      float y2 = y*y;
                float x4 = x2*x2;    float y4 = y2*y2;
                float r2 = x2+y2;    float r4 = r2*r2;
                float dD[4];
                dD[0] = 1 + kc[0]*(3*x2+y2) + kc[1]*(5*x4+6*x2*y2+y4) + kc[4]*r4*(7*x2+y2);
                dD[1] = kc[0]*2*x*y + kc[1]*4*x*y*r2 + kc[4]*6*x*y*r4;
                dD[2] = kc[0]*2*y*x + kc[1]*4*x*y*r2 + kc[4]*6*x*y*r4;
                dD[3] = 1 + kc[0]*(3*x2+y2) + kc[1]*(5*x4+6*x2*y2+y4) + kc[4]*r4*(7*y2+x2);

                // param1
                dp[0] = 0.0f; dp[1] =-p[2]; dp[2] = p[1];
                rotate3(TLR,dp,dr);
                dn[0] = dN[0]*dr[0] + dN[1]*dr[1] + dN[2]*dr[2];
                dn[1] = dN[3]*dr[0] + dN[4]*dr[1] + dN[5]*dr[2];
                dd[0]  = dD[0]*dn[0]+dD[1]*dn[1];
                dd[1]  = dD[2]*dn[0]+dD[3]*dn[1];
                dp2[0] = K[0]*dd[0]+K[1]*dd[1];
                dp2[1] = K[3]*dd[0]+K[4]*dd[1];
                jacobian[nPoints*0 + si] = dp2[0]*gradIx + dp2[1]*gradIy;

                // param2
                dp[0] = p[2]; dp[1] = 0; dp[2] = -p[0];
                rotate3(TLR,dp,dr);
                dn[0] = dN[0]*dr[0] + dN[1]*dr[1] + dN[2]*dr[2];
                dn[1] = dN[3]*dr[0] + dN[4]*dr[1] + dN[5]*dr[2];
                dd[0]  = dD[0]*dn[0]+dD[1]*dn[1];
                dd[1]  = dD[2]*dn[0]+dD[3]*dn[1];
                dp2[0] = K[0]*dd[0]+K[1]*dd[1];
                dp2[1] = K[3]*dd[0]+K[4]*dd[1];
                jacobian[nPoints*1 + si] = dp2[0]*gradIx + dp2[1]*gradIy;

                // param3
                dp[0] = -p[1]; dp[1] = p[0]; dp[2] = 0;
                rotate3(TLR,dp,dr);
                dn[0] = dN[0]*dr[0] + dN[1]*dr[1] + dN[2]*dr[2];
                dn[1] = dN[3]*dr[0] + dN[4]*dr[1] + dN[5]*dr[2];
                dd[0]  = dD[0]*dn[0]+dD[1]*dn[1];
                dd[1]  = dD[2]*dn[0]+dD[3]*dn[1];
                dp2[0] = K[0]*dd[0]+K[1]*dd[1];
                dp2[1] = K[3]*dd[0]+K[4]*dd[1];
                jacobian[nPoints*2 + si] = dp2[0]*gradIx + dp2[1]*gradIy;

                // param4
                dp[0] = 1; dp[1] = 0; dp[2] = 0;
                rotate3(TLR,dp,dr);
                dn[0] = dN[0]*dr[0] + dN[1]*dr[1] + dN[2]*dr[2];
                dn[1] = dN[3]*dr[0] + dN[4]*dr[1] + dN[5]*dr[2];
                dd[0]  = dD[0]*dn[0]+dD[1]*dn[1];
                dd[1]  = dD[2]*dn[0]+dD[3]*dn[1];
                dp2[0] = K[0]*dd[0]+K[1]*dd[1];
                dp2[1] = K[3]*dd[0]+K[4]*dd[1];
                jacobian[nPoints*3 + si] = dp2[0]*gradIx + dp2[1]*gradIy;

                // param5
                dp[0] = 0; dp[1] = 1; dp[2] = 0;
                rotate3(TLR,dp,dr);
                dn[0] = dN[0]*dr[0] + dN[1]*dr[1] + dN[2]*dr[2];
                dn[1] = dN[3]*dr[0] + dN[4]*dr[1] + dN[5]*dr[2];
                dd[0]  = dD[0]*dn[0]+dD[1]*dn[1];
                dd[1]  = dD[2]*dn[0]+dD[3]*dn[1];
                dp2[0] = K[0]*dd[0]+K[1]*dd[1];
                dp2[1] = K[3]*dd[0]+K[4]*dd[1];
                jacobian[nPoints*4 + si] = dp2[0]*gradIx + dp2[1]*gradIy;

                // param6
                dp[0] = 0; dp[1] = 0; dp[2] = 1;
                rotate3(TLR,dp,dr);
                dn[0] = dN[0]*dr[0] + dN[1]*dr[1] + dN[2]*dr[2];
                dn[1] = dN[3]*dr[0] + dN[4]*dr[1] + dN[5]*dr[2];
                dd[0]  = dD[0]*dn[0]+dD[1]*dn[1];
                dd[1]  = dD[2]*dn[0]+dD[3]*dn[1];
                dp2[0] = K[0]*dd[0]+K[1]*dd[1];
                dp2[1] = K[3]*dd[0]+K[4]*dd[1];
                jacobian[nPoints*5 + si] = dp2[0]*gradIx + dp2[1]*gradIy;
            } else {
                jacobian[nPoints*0 + si] = 0;
                jacobian[nPoints*1 + si] = 0;
                jacobian[nPoints*2 + si] = 0;
                jacobian[nPoints*3 + si] = 0;
                jacobian[nPoints*4 + si] = 0;
                jacobian[nPoints*5 + si] = 0;
                // store reference gray value for residual evaluation
                pOriginal[8+layer] = 0;
            }
        }
    }
}

void OMPFunctions::generateNormals(cv::Mat &xyzImage, int stride)
{
    int fw = xyzImage.cols; int fwp = fw*stride;
    int fh = xyzImage.rows;    
    float *pData   = (float*)xyzImage.ptr();
    int nBlocks = 4;
    int blockSize = fh/nBlocks;
    int blockID = 0;
    #pragma omp parallel for private(blockID)
    for (blockID = 0; blockID < nBlocks; blockID++) {
        // if this is the last block, do not process the last row :
        int ydec = 0; if (blockID == nBlocks-1) ydec = 1;
        int yinc = 0; if (blockID == 0) yinc = 1;
        for (float yi = blockID*blockSize+yinc; yi < (blockID+1)*blockSize-ydec; yi++) {
            for (float xi = 1; xi < (fw-1); xi++) {
                int offset = xi+yi*fw;
                int offsetp = offset*stride;
                float *p = &pData[offsetp];
                float *u1 = &pData[offsetp+stride];
                float *u0 = &pData[offsetp-stride];
                float *v1 = &pData[offsetp+fwp];
                float *v0 = &pData[offsetp-fwp];
                /*float zmin = p[2];
                float zmax = p[2];
                if (u1[2] < zmin) zmin = u1[2];
                if (u1[2] > zmax) zmax = u1[2];
                if (u0[2] < zmin) zmin = u0[2];
                if (u0[2] > zmax) zmax = u0[2];
                if (v1[2] < zmin) zmin = v1[2];
                if (v1[2] > zmax) zmax = v1[2];
                if (v0[2] < zmin) zmin = v0[2];
                if (v0[2] > zmax) zmax = v0[2];
                float zdiff = fabs(zmax-zmin);*/
                // for smooth surface do :
                if (p[6] > 0/* && u1[6] > 0 && u0[6] > 0 && v1[6] > 0 && v0[6] > 0 && zdiff < 50.0f*/)  {
                    float nu[3],nv[3],n[3];
                    nu[0] = u1[0] - u0[0]; nu[1] = u1[1] - u0[1]; nu[2] = u1[2] - u0[2];
                    nv[0] = v1[0] - v0[0]; nv[1] = v1[1] - v0[1]; nv[2] = v1[2] - v0[2];
                    // compute normal as crossproduct
                    n[0] =  nu[1] * nv[2] - nu[2] * nv[1];
                    n[1] =-(nu[0] * nv[2] - nu[2] * nv[0]);
                    n[2] =  nu[0] * nv[1] - nu[1] * nv[0];
                    // normal to unit length
                    float len = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]+1e-5f);
                    // TODO: use this magnitude (area of square) to prune out invalid normals (mismatch in depth)
                    n[0] /= len; n[1] /= len; n[2] /= len;
                    p[3] = -n[0];
                    p[4] = -n[1];
                    p[5] = -n[2];
                } else {
/*                    p[3] = 0;
                    p[4] = 0;
                    p[5] = 1;*/
                }
            }
        }
    }
}
