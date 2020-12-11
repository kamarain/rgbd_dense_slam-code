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
#include <image2/ImagePyramid2.h>
#include "hostUtils.h"
#include <stdio.h>
#include <cwchar>

#include <calib/calib.h>
namespace d2zutils {
	#include "kernelUtils.h"
}
#include <rendering/VertexBuffer2.h>

using namespace d2zutils;

__global__ void d2ZKernel( unsigned short *dPtr, float *zPtr,int width, int height, float *calibDataDev/*, float *vData, unsigned char *imgData*/)
{
	int xi = blockIdx.x*blockDim.x+threadIdx.x;
	int yi = blockIdx.y*blockDim.y+threadIdx.y;
	int dstIdx = xi+yi*width;
	// IR image -> disparity image has constant offset (Konolige's tech guide)
	// http://www.ros.org/wiki/kinect_calibration/technical
	unsigned int sxi = xi - 4;
	unsigned int syi = yi - 3;
	if (sxi < width && syi < height) {
		int srcIdx = sxi + syi * width;
            //    float fx = calibDataDev[KL_OFFSET];
        float c0 = calibDataDev[C0_OFFSET];
        float c1 = calibDataDev[C1_OFFSET];

      //          float b = calibDataDev[b_OFFSET];
//                float B = calibDataDev[B_OFFSET];
		//float minDist = calibDataDev[9*2+5+16+1+2*9+1];
                float maxDist = calibDataDev[MAXD_OFFSET];
		float d = (float)dPtr[srcIdx];
//		float *T = &calibDataDev[9*2+5];
//		float *KR = &calibDataDev[0];
		if (d > 0 && d < 2047) {
            float z = fabs(1.0f/(c0+c1*d));
            //float z = fabs(8.0f*b*fx/(B-d));
			if (z > maxDist) z = 0.0f;
			zPtr[dstIdx] = z/maxDist;		
		/*	float3 p3,r3,p2; 
			p3.x = -(float(xi) - cx) * z / fx;
			p3.y = -(float(yi) - cy) * z / fy;
			p3.z = -z;
			matrixMultVec4(T, p3, r3);
			vData[dstIdx*6+0] = r3.x; 
			vData[dstIdx*6+1] = r3.y;
			vData[dstIdx*6+2] = r3.z;

			matrixMultVec3(KR, r3, p2); p2.x /= p2.z; p2.y /= p2.z;

			unsigned char color = 0;
			bilinearInterpolation(p2, width, height, imgData, color);
			float colorF = float(color)/255.0f;
			vData[dstIdx*6+3] = colorF;
			vData[dstIdx*6+4] = colorF;
			vData[dstIdx*6+5] = colorF;*/
			return;
		} 
	}
	zPtr[dstIdx] = 0.0f;
/*	vData[dstIdx*6+0] = 0;
	vData[dstIdx*6+1] = 0;
	vData[dstIdx*6+2] = 0;
	vData[dstIdx*6+3] = 0.0f;
	vData[dstIdx*6+4] = 0.0f;
	vData[dstIdx*6+5] = 0.0f;*/
}


__global__ void undistortDisparityKernel( unsigned short *dPtr, float *uPtr,int width, int height, float *calibDataDev)
{
    int xi = blockIdx.x*blockDim.x+threadIdx.x;
    int yi = blockIdx.y*blockDim.y+threadIdx.y;
    int idx = xi+yi*width;

    float alpha0 = calibDataDev[ALPHA0_OFFSET];
    float alpha1 = calibDataDev[ALPHA1_OFFSET];
    float *beta  = &calibDataDev[BETA_OFFSET];

    float d = (float)dPtr[idx];
    float ud = 0xffff;
    if (d < 2047) {
        ud = d + beta[idx]*expf(alpha0-alpha1*d);
    }
    uPtr[idx] = ud;
}



__global__ void d2ZLowKernel( unsigned short *dPtr, float *zPtr, int width, int height, float *calibDataDev, float xOff, float yOff)
{
    int xi = blockIdx.x*blockDim.x+threadIdx.x;
    int yi = blockIdx.y*blockDim.y+threadIdx.y;
    int dstIdx = xi+yi*(width/2);

    // IR image -> disparity image has constant offset (Konolige's tech guide)
    // http://www.ros.org/wiki/kinect_calibration/technical
    unsigned int sxi = 2*xi + xOff;
    unsigned int syi = 2*yi + yOff;
    if (sxi < width-1 && syi < height-1) {
        int srcIdx1 = sxi + 0 + (syi + 0) * width;
        int srcIdx2 = sxi + 1 + (syi + 0) * width;
        int srcIdx3 = sxi + 1 + (syi + 1) * width;
        int srcIdx4 = sxi + 0 + (syi + 1) * width;
        // fx and B manually *2 because they RGB_WIDTH_SMALL*2 = DISPARITY_WIDTH (set in the main program according to rgb)
        float c0 = calibDataDev[C0_OFFSET];
        float c1 = calibDataDev[C1_OFFSET];
        //float fx = calibDataDev[KL_OFFSET]*2;
        //float B = calibDataDev[B_OFFSET]*2;
        // the rest values read normally as they are reso invariant
        //float b = calibDataDev[b_OFFSET];
        float minDist = calibDataDev[MIND_OFFSET];
        float maxDist = calibDataDev[MAXD_OFFSET];

        float d1 = (float)dPtr[srcIdx1];
        float d2 = (float)dPtr[srcIdx2];
        float d3 = (float)dPtr[srcIdx3];
        float d4 = (float)dPtr[srcIdx4];
        if ((d1 < 2047) && (d2 < 2047) && (d3 < 2047) && (d4 < 2047)) {
            //			float d = (d1+d2+d3+d4)/4.0f;
            float d = d1;
            if (d2 < d) d = d2;
            if (d3 < d) d = d3;
            if (d4 < d) d = d4;
//            float z = fabs(8.0f*b*fx/(B-d));
            float z = fabs(1.0f/(c0+c1*d));
            if (z > maxDist || z < minDist) z = 0.0f;
            zPtr[dstIdx] = (z-minDist)/(maxDist-minDist);
            return;
        }
    }
    zPtr[dstIdx] = 0.0f;
}


__global__ void d2ZLowHdrKernel( float *dPtr, float *zPtr, int width, int height, float *calibDataDev, float xOff, float yOff)
{
    int xi = blockIdx.x*blockDim.x+threadIdx.x;
    int yi = blockIdx.y*blockDim.y+threadIdx.y;
    int dstIdx = xi+yi*(width/2);

    // IR image -> disparity image has constant offset (Konolige's tech guide)
    // http://www.ros.org/wiki/kinect_calibration/technical
    unsigned int sxi = 2*xi + xOff;
    unsigned int syi = 2*yi + yOff;
    if (sxi < width-1 && syi < height-1) {
        int srcIdx1 = sxi + 0 + (syi + 0) * width;
        int srcIdx2 = sxi + 1 + (syi + 0) * width;
        int srcIdx3 = sxi + 1 + (syi + 1) * width;
        int srcIdx4 = sxi + 0 + (syi + 1) * width;
        // fx and B manually *2 because they RGB_WIDTH_SMALL*2 = DISPARITY_WIDTH (set in the main program according to rgb)
        float c0 = calibDataDev[C0_OFFSET];
        float c1 = calibDataDev[C1_OFFSET];
        //float fx = calibDataDev[KL_OFFSET]*2;
        //float B = calibDataDev[B_OFFSET]*2;
        // the rest values read normally as they are reso invariant
        //float b = calibDataDev[b_OFFSET];
        float minDist = calibDataDev[MIND_OFFSET];
        float maxDist = calibDataDev[MAXD_OFFSET];

        float d1 = dPtr[srcIdx1];
        float d2 = dPtr[srcIdx2];
        float d3 = dPtr[srcIdx3];
        float d4 = dPtr[srcIdx4];
        if ((d1 < 2047) && (d2 < 2047) && (d3 < 2047) && (d4 < 2047)) {
            //			float d = (d1+d2+d3+d4)/4.0f;
            float d = d1;
            if (d2 < d) d = d2;
            if (d3 < d) d = d3;
            if (d4 < d) d = d4;
            float z = fabs(1.0f/(c0+c1*d));
            if (z > maxDist || z < minDist) z = 0.0f;
            zPtr[dstIdx] = (z-minDist)/(maxDist-minDist);
            return;
        }
    }
    zPtr[dstIdx] = 0.0f;
}

__global__ void setMaxZKernel(float *zPtr, float *calibDataDev) {
    int offset = blockIdx.x*blockDim.x+threadIdx.x;
    float maxDist = calibDataDev[MAXD_OFFSET];
    zPtr[offset] = maxDist;
}



__global__ void z2CloudKernel(float *zPtr,int width, int height, float *calibDataDev, float *vData, float *rgbData, float *imgData1, float *zPtrDst, int stride)
{
    int xi = blockIdx.x*blockDim.x+threadIdx.x;
    int yi = blockIdx.y*blockDim.y+threadIdx.y;
    int offset = xi+yi*width;
    // make sure stride has matching number of elements stored here!
    int idxStride = offset*stride;
    float z   = zPtr[offset];
    float maxDist = calibDataDev[MAXD_OFFSET];
    if (z > 0) {
        float fx      = calibDataDev[KL_OFFSET];
        float fy      = calibDataDev[KL_OFFSET+4];
        float cx      = calibDataDev[KL_OFFSET+2];
        float cy      = calibDataDev[KL_OFFSET+5];
        float minDist = calibDataDev[MIND_OFFSET];
        float *T      = &calibDataDev[TLR_OFFSET];
        float *KR     = &calibDataDev[KR_OFFSET];
        float *kc     = &calibDataDev[KcR_OFFSET];

        z   = -(z*(maxDist-minDist) + minDist);

        float3 p3,r3;
        p3.x = (float(xi) - cx) * z / fx;
        p3.y = (float(yi) - cy) * z / fy;
        p3.z = z;
        matrixMultVec4(T, p3, r3);

        float2 pu,p2_1;
        pu.x = r3.x / r3.z;
        pu.y = r3.y / r3.z;

        distortPoint(pu,kc,KR,p2_1);

        bool pointsOnScreen = true;
        if (!inBounds(p2_1,width,height)) pointsOnScreen = false;

        float colorR1 = 0, colorG1 = 0, colorB1 = 0;
        float color1  = 0;
        float gradX1  = 0, gradY1  = 0;
        if (pointsOnScreen) {
            int xdi,ydi;
            float fx,fy;
            xdi = (int)p2_1.x; ydi = (int)p2_1.y; fx = p2_1.x - xdi; fy = p2_1.y - ydi;
            // interpolate rgb color
            bilinearInterpolation(xdi,   ydi,   fx, fy, width, rgbData, colorR1,colorG1,colorB1);
            // faster to compute gray value from RGB than bilinear interpolation:
            color1 = 0.3f*colorR1 + 0.59f*colorG1 + 0.11f*colorB1;

            int zoff = xdi+ydi*width;

            float nZ = (-r3.z-minDist)/(maxDist-minDist);
            zPtrDst[zoff] = nZ;
            zPtrDst[zoff+1] = nZ;
            zPtrDst[zoff+width] = nZ;
            zPtrDst[zoff+width+1] = nZ;

            // interpolate gradient
            float colorN = 0, colorS = 0, colorE = 0, colorW = 0;

            xdi = (int)p2_1.x; ydi = (int)(p2_1.y-1.0f);
            bilinearInterpolation(xdi, ydi, fx, fy, width, imgData1, colorN);

            xdi = (int)(p2_1.x+1.0f); ydi = (int)p2_1.y;
            bilinearInterpolation(xdi, ydi,   fx, fy, width, imgData1, colorE);

            xdi = (int)(p2_1.x-1.0f); ydi = (int)p2_1.y;
            bilinearInterpolation(xdi, ydi,   fx, fy, width, imgData1, colorW);

            xdi = (int)p2_1.x; ydi = (int)(p2_1.y+1.0f);
            bilinearInterpolation(xdi,   ydi, fx, fy, width, imgData1, colorS);
            gradX1 = (colorE-colorW)/2.0f;
            gradY1 = (colorS-colorN)/2.0f;

            vData[idxStride+0]  = r3.x;
            vData[idxStride+1]  = r3.y;
            vData[idxStride+2]  = r3.z;
            vData[idxStride+3]  = 0.0f;//n.x
            vData[idxStride+4]  = 0.0f;//n.y
            vData[idxStride+5]  = 0.0f;//n.z
            vData[idxStride+6]  = p2_1.x;
            vData[idxStride+7]  = p2_1.y;
            vData[idxStride+8]  = colorR1;
            vData[idxStride+9]  = colorG1;
            vData[idxStride+10] = colorB1;
            vData[idxStride+11] = gradX1; // store gradientX for reference image based optimization
            vData[idxStride+12] = gradY1; // store gradientY for reference image based optimization
            vData[idxStride+13] = min(fabs(gradY1)+fabs(gradX1),1.0f); // store gradient magnitude for thresholding, range: [0,1]
            vData[idxStride+14] = color1;
            //            vData[idxStride+15] = gradX2;
            //            vData[idxStride+16] = gradY2;
            //            vData[idxStride+17] = color2;
            //            vData[idxStride+18] = gradX3;
            //            vData[idxStride+19] = gradY3;
            //            vData[idxStride+20] = color3;
            return;
        }
    }

    vData[idxStride+0]  = 0.0f;
    vData[idxStride+1]  = 0.0f;
    vData[idxStride+2]  = -maxDist; // set depth to a large value for ensuring big depth discrepancy for zweighting
    vData[idxStride+3]  = 0.0f;
    vData[idxStride+4]  = 0.0f;
    vData[idxStride+5]  = 0.0f;
    vData[idxStride+6]  = 0.0f;
    vData[idxStride+7]  = 0.0f;
    vData[idxStride+8]  = 0.0f;
    vData[idxStride+9]  = 0.0f;
    vData[idxStride+10] = 0.0f;
    vData[idxStride+11] = 0.0f;
    vData[idxStride+12] = 0.0f;
    vData[idxStride+13] = 0.0f;
    vData[idxStride+14] = 0.0f;
    vData[idxStride+15] = 0.0f;
    vData[idxStride+16] = 0.0f;
    vData[idxStride+17] = 0.0f;
    vData[idxStride+18] = 0.0f;
    vData[idxStride+19] = 0.0f;
    vData[idxStride+20] = 0.0f;

}

__global__ void z2CloudKernelFast(float *zPtr,int width, int height, float *calibDataDev, float *vData, float *rgbData, float *imgData1, float *zPtrDst, int stride)
{
    int xi = blockIdx.x*blockDim.x+threadIdx.x;
    int yi = blockIdx.y*blockDim.y+threadIdx.y;
    int offset = xi+yi*width;
    // make sure stride has matching number of elements stored here!
    int idxStride = offset*stride;
    float z   = zPtr[offset];
    float maxDist = calibDataDev[MAXD_OFFSET];
    if (z > 0) {
            float fx      = calibDataDev[KL_OFFSET];
            float fy      = calibDataDev[KL_OFFSET+4];
            float cx      = calibDataDev[KL_OFFSET+2];
            float cy      = calibDataDev[KL_OFFSET+5];
            float minDist = calibDataDev[MIND_OFFSET];
            float *T      = &calibDataDev[TLR_OFFSET];
            float *KR     = &calibDataDev[KR_OFFSET];
            float *kc     = &calibDataDev[KcR_OFFSET];

            z   = -(z*(maxDist-minDist) + minDist);

            float3 p3,r3;
            p3.x = (float(xi) - cx) * z / fx;
            p3.y = (float(yi) - cy) * z / fy;
            p3.z = z;
            matrixMultVec4(T, p3, r3);

            float2 pu,p2_1;
            pu.x = r3.x / r3.z;
            pu.y = r3.y / r3.z;
            distortPoint(pu,kc,KR,p2_1);

            float colorR1 = 0, colorG1 = 0, colorB1 = 0;
            float color1  = 0;

            int xdi = (int)p2_1.x;
            int ydi = (int)p2_1.y;

            if (xdi >= 0 && ydi >= 0 && xdi <= width-2 && ydi <= height-2)
            {
                // xdi in [2,width - 4]
                // ydi in [2,height- 4]
                // -> (p2.x,p2.y) maps into valid lowres domain too + bilinear interpolation
                float fx = p2_1.x - xdi;
                float fy = p2_1.y - ydi;
                // interpolate rgb color
                bilinearInterpolation(xdi,   ydi,   fx, fy, width, rgbData, colorR1,colorG1,colorB1);
                // faster to compute gray value from RGB than bilinear interpolation:
                color1 = 0.3f*colorR1 + 0.59f*colorG1 + 0.11f*colorB1;
                int zoff = xdi+ydi*width;

                float nZ = (-r3.z-minDist)/(maxDist-minDist);
                zPtrDst[zoff] = nZ;
                zPtrDst[zoff+1] = nZ;
                zPtrDst[zoff+width] = nZ;
                zPtrDst[zoff+width+1] = nZ;
                vData[idxStride+0]  = r3.x;
                vData[idxStride+1]  = r3.y;
                vData[idxStride+2]  = r3.z;
                vData[idxStride+3]  = 0.0f; //n.x
                vData[idxStride+4]  = 0.0f; //n.y
                vData[idxStride+5]  = 0.0f; //n.z
                vData[idxStride+6]  = p2_1.x;
                vData[idxStride+7]  = p2_1.y;
                vData[idxStride+8]  = colorR1;//n.x;
                vData[idxStride+9]  = colorG1;//n.y;
                vData[idxStride+10] = colorB1;//n.z;
                vData[idxStride+11] = 0;
                vData[idxStride+12] = 0;
                vData[idxStride+13] = 0.0f; // store zero gradient magnitude (no pixel selection support in this method)
                vData[idxStride+14] = color1;
    //            vData[idxStride+15] = gradX2;
    //            vData[idxStride+16] = gradY2;
    //            vData[idxStride+17] = color2;
    //            vData[idxStride+18] = gradX3;
    //            vData[idxStride+19] = gradY3;
    //            vData[idxStride+20] = color3;
                return;
            }
    }

    vData[idxStride+0]  = 0.0f;
    vData[idxStride+1]  = 0.0f;
    vData[idxStride+2]  = -maxDist; // set depth to a large value for ensuring big depth discrepancy for zweighting
    vData[idxStride+3]  = 0.0f;
    vData[idxStride+4]  = 0.0f;
    vData[idxStride+5]  = 0.0f;
    vData[idxStride+6]  = 0.0f;
    vData[idxStride+7]  = 0.0f;
    vData[idxStride+8]  = 0.0f;
    vData[idxStride+9]  = 0.0f;
    vData[idxStride+10] = 0.0f;
    vData[idxStride+11] = 0.0f;
    vData[idxStride+12] = 0.0f;
    vData[idxStride+13] = 0.0f;
    vData[idxStride+14] = 0.0f;
    vData[idxStride+15] = 0.0f;
    vData[idxStride+16] = 0.0f;
    vData[idxStride+17] = 0.0f;
    vData[idxStride+18] = 0.0f;
    vData[idxStride+19] = 0.0f;
    vData[idxStride+20] = 0.0f;

}

__global__ void setNormalsCudaKernel(float *vertexData,float *normalData, float scale, int stride) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idxStride = idx*stride;
    float x = vertexData[idxStride+0];
    float y = vertexData[idxStride+1];
    float z = vertexData[idxStride+2];
    vertexData[idxStride+3] = x+normalData[idx*3+0]*scale;
    vertexData[idxStride+4] = y+normalData[idx*3+1]*scale;
    vertexData[idxStride+5] = z+normalData[idx*3+2]*scale;
}


__global__ void extractGradientKernel(float *vertexData, int stride, int slot, float *gradientScratchDev) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int idxStride = idx*stride;
    float gradMag = vertexData[idxStride+slot];
    gradientScratchDev[idx] = gradMag;
}

__global__ void addVertexAttributesKernel(int *indexPointer, float *vData, float *zPtr, int width, int height, float *calibDataDev, float *imgData1, float *imgData2, float *imgData3, int stride)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int offset = indexPointer[idx];
    // make sure stride has matching number of elements stored here!
    int idxStride = offset*stride;
    // zvalues for computing point normal

    float *kc     = &calibDataDev[KcR_OFFSET];
    float *KR     = &calibDataDev[KR_OFFSET];

    float3 r3;
    r3.x = vData[idxStride+0];
    r3.y = vData[idxStride+1];
    r3.z = vData[idxStride+2];

    /*
    float zNu = zPtr[offset+1];
    float zNv = zPtr[offset+width];
    zNu   = -(zNu*(maxDist-minDist) + minDist);
    zNv   = -(zNv*(maxDist-minDist) + minDist);


    float3 p3;
    p3.x = (float(xi+1) - cx) * zNu / fx;
    p3.y = (float(yi+0) - cy) * zNu / fy;
    p3.z = zNu;
    matrixMultVec4(T, p3, r3u);

    p3.x = (float(xi+0) - cx) * zNv / fx;
    p3.y = (float(yi+1) - cy) * zNv / fy;
    p3.z = zNv;
    matrixMultVec4(T, p3, r3v);

    float3 nu,nv,n;
    nu.x = r3u.x - r3.x; nu.y = r3u.y - r3.y; nu.z = r3u.z - r3.z;
    nv.x = r3v.x - r3.x; nv.y = r3v.y - r3.y; nv.z = r3v.z - r3.z;
    // compute normal as crossproduct
    n.x =  nu.y * nv.z - nu.z * nv.y;
    n.y =-(nu.x * nv.z - nu.z * nv.x);
    n.z =  nu.x * nv.y - nu.y * nv.x;
    // normal to unit length
    float len = sqrt(n.x*n.x + n.y*n.y + n.z*n.z+1e-5f);
    // TODO: use this magnitude (area of square) to prune out invalid normals (mismatch in depth)
    n.x /= len; n.y /= len; n.z /= len;
    */
    float color2  = 0, color3  = 0;
    float colorN,colorW,colorE,colorS;
    float gradX2  = 0, gradY2  = 0;
    float gradX3  = 0, gradY3  = 0;

    float2 p_1,p_2,p_3;

    float2 pu;
    pu.x = r3.x / r3.z;
    pu.y = r3.y / r3.z;
    distortPoint(pu,kc,KR,p_1);

    bool pointsOnScreen = true;
    if (!inBounds(p_1,width,height)) pointsOnScreen = false;

    if (pointsOnScreen) {
       // compute low-resolution coordinates
        float a = 0.5f; float b = -0.25f;
        p_2.x  = a*p_1.x  + b; p_2.y = a*p_1.y + b;

        p_3.x  = a*p_2.x  + b; p_3.y  = a*p_2.y + b;

        int xdi,ydi;
        float fracX,fracY;

        xdi = (int)p_2.x; fracX = p_2.x - xdi;
        ydi = (int)p_2.y; fracY = p_2.y - ydi;
        bilinearInterpolation(xdi,   ydi,   fracX, fracY, width/2, imgData2, color2);

        xdi = (int)p_2.x; 
        ydi = (int)p_2.y-1; 
        bilinearInterpolation(xdi, ydi, fracX, fracY, width/2, imgData2, colorN);

        xdi = (int)p_2.x-1; 
        ydi = (int)p_2.y; 
        bilinearInterpolation(xdi, ydi, fracX, fracY, width/2, imgData2, colorW);

        xdi = (int)p_2.x+1; 
        ydi = (int)p_2.y; 
        bilinearInterpolation(xdi, ydi, fracX, fracY, width/2, imgData2, colorE);

        xdi = (int)p_2.x; 
        ydi = (int)p_2.y+1; 
        bilinearInterpolation(xdi, ydi, fracX, fracY, width/2, imgData2, colorS);

        gradX2 = (colorE-colorW)/2.0f;
        gradY2 = (colorS-colorN)/2.0f;

        xdi = (int)p_3.x; fracX = p_3.x - xdi;
        ydi = (int)p_3.y; fracY = p_3.y - ydi;
        bilinearInterpolation(xdi,   ydi,   fracX, fracY, width/4, imgData3, color3);

        xdi = (int)p_3.x; 
        ydi = (int)p_3.y-1; 
        bilinearInterpolation(xdi,   ydi, fracX, fracY, width/4, imgData3, colorN);

        xdi = (int)p_3.x-1; 
        ydi = (int)p_3.y; 
        bilinearInterpolation(xdi, ydi,   fracX, fracY, width/4, imgData3, colorW);

        xdi = (int)p_3.x+1; 
        ydi = (int)p_3.y; 
        bilinearInterpolation(xdi, ydi,   fracX, fracY, width/4, imgData3, colorE);

        xdi = (int)p_3.x; 
        ydi = (int)p_3.y+1; 
        bilinearInterpolation(xdi,   ydi, fracX, fracY, width/4, imgData3, colorS);

       gradX3 = (colorE-colorW)/2.0f;
       gradY3 = (colorS-colorN)/2.0f;
    }
    // normal points are currently computed on CPU only for keyframes
    //vData[idxStride+3]  = r3.x - n.x*100.0f;
    //vData[idxStride+4]  = r3.y - n.y*100.0f;
    //vData[idxStride+5]  = r3.z - n.z*100.0f;
//  vData[idxStride+11] = gradX1; // store gradientX for reference image based optimization
//  vData[idxStride+12] = gradY1; // store gradientY for reference image based optimization
//  vData[idxStride+13] = 127.9f*(fabs(gradY1)+fabs(gradX1)); // store gradient magnitude for thresholding, int range: [0,255]
//  vData[idxStride+14] = color1;

    vData[idxStride+15] = gradX2;
    vData[idxStride+16] = gradY2;
    vData[idxStride+17] = color2;
    vData[idxStride+18] = gradX3;
    vData[idxStride+19] = gradY3;
    vData[idxStride+20] = color3;
}

extern "C" void d2ZCuda(unsigned short *disparity16U, Image2 *zImage, float *calibDataDev, float xOff, float yOff)
{
    if (disparity16U == 0 || zImage == 0 || zImage->devPtr == NULL || calibDataDev == NULL) return;
	float *zPtr= (float*)zImage->devPtr;
	dim3 cudaBlockSize(32,30,1);
    dim3 cudaGridSize(zImage->width/cudaBlockSize.x,zImage->height/cudaBlockSize.y,1);
    d2ZLowKernel<<<cudaGridSize,cudaBlockSize,0,zImage->cudaStream>>>(disparity16U,zPtr,zImage->width*2,zImage->height*2,calibDataDev, xOff, yOff);
}

extern "C" void d2ZCudaHdr(float *disparityHdr, Image2 *zImage, float *calibDataDev, float xOff, float yOff) {
    if (disparityHdr == 0 || zImage == 0 || zImage->devPtr == NULL || calibDataDev == NULL) return;
    float *zPtr= (float*)zImage->devPtr;
    dim3 cudaBlockSize(32,30,1);
    dim3 cudaGridSize(zImage->width/cudaBlockSize.x,zImage->height/cudaBlockSize.y,1);
    d2ZLowHdrKernel<<<cudaGridSize,cudaBlockSize,0,zImage->cudaStream>>>(disparityHdr,zPtr,zImage->width*2,zImage->height*2,calibDataDev, xOff,yOff);
}

extern "C" void undistortDisparityCuda(unsigned short *disparity16U, float *uPtr, float *calibDataDev, int width, int height, cudaStream_t stream = 0)
{
    if (disparity16U == 0 || uPtr == NULL || calibDataDev == NULL) return;
    dim3 cudaBlockSize(32,30,1);
    dim3 cudaGridSize(width/cudaBlockSize.x,height/cudaBlockSize.y,1);
    undistortDisparityKernel<<<cudaGridSize,cudaBlockSize,0,stream>>>(disparity16U,uPtr,width,height,calibDataDev);
}


extern "C" void z2CloudCuda(Image2 *zImageIR, float *calibDataDev, VertexBuffer2 *vbuffer, Image2 *rgbImage, ImagePyramid2 *grayPyramid, Image2 *zImage, bool computeGradients)
{
    if (zImageIR == 0 || zImageIR->devPtr == NULL || zImage == 0 || zImage->devPtr == NULL || calibDataDev == NULL || vbuffer == NULL || vbuffer->devPtr == NULL || rgbImage == NULL || rgbImage->devPtr == NULL || grayPyramid == NULL) {
        printf("null given to z2CloudCuda!\n");
        if (zImage->devPtr == NULL) printf("zImage not locked!\n");
        fflush(stdin);
        fflush(stdout);
        return;
    }
    float *imgData = (float*)grayPyramid->getImageRef(0).devPtr;
    if (imgData == NULL) {
        return;
    }
    float *zPtr= (float*)zImageIR->devPtr;
    float *zPtrDst= (float*)zImage->devPtr;
    float *vData = (float*)vbuffer->devPtr;
    float *rgbData = (float*)rgbImage->devPtr;
    dim3 cudaBlockSize(32,15,1);
    dim3 cudaGridSize(zImage->width/cudaBlockSize.x,zImage->height/cudaBlockSize.y,1);
    vbuffer->setVertexAmount(zImage->width * zImage->height);
    if (computeGradients) {
        z2CloudKernel<<<cudaGridSize,cudaBlockSize,0,vbuffer->cudaStream>>>(zPtr,zImageIR->width,zImageIR->height,calibDataDev,vData,rgbData,imgData,zPtrDst,vbuffer->getStride());
    } else {
        z2CloudKernelFast<<<cudaGridSize,cudaBlockSize,0,vbuffer->cudaStream>>>(zPtr,zImageIR->width,zImageIR->height,calibDataDev,vData,rgbData,imgData,zPtrDst,vbuffer->getStride());
    }
    checkCudaError("z2CloudCuda error");
}

extern "C" void addVertexAttributesCuda(Image2 *zImage, float *calibDataDev, VertexBuffer2 *vbuffer, ImagePyramid2 *grayPyramid)
{
    if (vbuffer == NULL || vbuffer->devPtr == NULL || vbuffer->indexDevPtr == NULL || grayPyramid == NULL || calibDataDev == NULL || zImage == NULL || zImage->devPtr == NULL) {
        printf("addVertexAttributesCuda: null pointer given!\n"); return;
    }

    float *imgData[3];
    assert(grayPyramid->nLayers == 3);
    for (int i = 0; i < 3; i++) {
        imgData[i] = (float*)grayPyramid->getImageRef(i).devPtr;
        if (imgData[i] == NULL) {
            printf("addVertexAttributesCuda error: grayPyramid layer %d not locked! panik exit \n",i);
            return;
        }
    }

     // enforce multiple of 1024 for element count -> max performance
     if (vbuffer->getElementsCount()%512 != 0) {
          printf("addVertexAttributesCuda: vbuffer has wrong number of selected pixels! (%d)\n",vbuffer->getElementsCount());
          return;
    }

    float *zPtr= (float*)zImage->devPtr;
    int *indexPointer = (int*)vbuffer->indexDevPtr;
    float *vertexData = (float*)vbuffer->devPtr;
    int nElements = vbuffer->getElementsCount();
    dim3 cudaBlockSize(512,1,1);
    dim3 cudaGridSize(nElements/cudaBlockSize.x,1,1);
    addVertexAttributesKernel<<<cudaGridSize,cudaBlockSize,0,vbuffer->cudaStream>>>(indexPointer,vertexData,zPtr,zImage->width,zImage->height,calibDataDev,imgData[0],imgData[1],imgData[2],vbuffer->getStride());
    checkCudaError("addVertexAttributesCuda error");
}

extern "C" void setNormalsCuda(VertexBuffer2 *vbuffer, float *normalData, float scale) {
    if (vbuffer == NULL || vbuffer->devPtr == NULL || normalData == NULL) {
        printf("setNormalsCuda: null pointer given!\n"); return;
    }

     // enforce multiple of 1024 for element count -> max performance
     if (vbuffer->getVertexCount()%1024 != 0) {
          printf("setNormalsCuda: vbuffer has wrong number of vertices! (%d)\n",vbuffer->getVertexCount());
          return;
    }

    float *vertexData = (float*)vbuffer->devPtr;
    int nElements = vbuffer->getVertexCount();
    dim3 cudaBlockSize(1024,1,1);
    dim3 cudaGridSize(nElements/cudaBlockSize.x,1,1);
    setNormalsCudaKernel<<<cudaGridSize,cudaBlockSize,0,vbuffer->cudaStream>>>(vertexData,normalData,scale,vbuffer->getStride());
    checkCudaError("setNormalsCuda error");
}

extern "C" void extractGradientMagnitudes(VertexBuffer2 *vbuffer, float *gradientScratchDev)
{
    if (vbuffer == NULL || vbuffer->devPtr == NULL || gradientScratchDev == NULL) {
        printf("extractGradientMagnitudes: null pointer given!\n"); return;
    }

     // enforce multiple of 1024 for element count -> max performance
     if (vbuffer->getVertexCount()%1024 != 0) {
          printf("extractGradientMagnitudes: vbuffer has wrong number of vertices! (%d)\n",vbuffer->getVertexCount());
          return;
    }

     if (vbuffer->getStride() != VERTEXBUFFER_STRIDE) {
         printf("extractGradientMagnitudes: vertexbuffer has illegal stride (%d), must be %d!\n",vbuffer->getStride(),VERTEXBUFFER_STRIDE);
         fflush(stdin); fflush(stdout);
         return;
     }

    float *vertexData = (float*)vbuffer->devPtr;
    int nVertices = vbuffer->getVertexCount();
    dim3 cudaBlockSize(1024,1,1);
    dim3 cudaGridSize(nVertices/cudaBlockSize.x,1,1);

    extractGradientKernel<<<cudaGridSize,cudaBlockSize,0,vbuffer->cudaStream>>>(vertexData,vbuffer->getStride(),13,gradientScratchDev);
    checkCudaError("extractGradientMagnitudes error");
}
