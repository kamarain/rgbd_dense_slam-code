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
#include <string.h>
#include "ICP.h"
#include <multicore.h>
#include <basic_math.h>
//#include <image2/Image2.h>
#include <expm/expmHost.h>

static int frame = 0;

ICP::ICP() {
    identity3x3(&KL[0]);
    identity3x3(&origKL[0]);
    identity3x3(&KR[0]);
    identity3x3(&origKR[0]);
    identity4x4(&TLR[0]);
    identity4x4(&T[0]);
    identity4x4(&Tinc[0]);
    for (int ri = 0; ri < MAX_PHOTOMETRIC_REFERENCES; ri++) {
        identity4x4(&TphotoBase[ri][0]);
    }
    nFramesReceived = 0;
    nFramesFused = 0;
    for (int i = 0; i < 5; i++) kc[i] = 0;
    // number of photometric points in use
    this->nPhotometric = 0;
    this->nPhotometricReferences = 0;
    scaleIn = 1e-3f;
    scaleOut = 1e+3f;
    usePhotometric = true;
    useICP = true;
    referenceResolutionX = 320;
    referenceResolutionY = 240;
}


void ICP::release() {
    for (int i = 0; i < 3; i++) {
        xyzRef[i].release();
        xyzCur[i].release();
        for (int ri = 0; ri < nPhotometricReferences; ri++) grayCurPrev[ri][i].release();
        grayCur[i].release();
        icpSelectionMask[i].release();
        photoJacobian[i].release();
    }
    photometricSelection.release();    
    xyzHiresRef.release();
    weightsRef.release();
    weightsCur.release();
    depthMapCur.release();
    residualImage.release();
    jacobianImage.release();
    normalStatus.release();
}

ICP::~ICP() {

}
/*
Image2 &ICP::getDepthRefTex(int layer) {
    //return normalMap3C[layer];
    return depthCur1C[layer];
}
*/
/*void absImage(cv::Mat &image) {
    int sz = image.cols*image.rows*image.channels();
    float *ptr = (float*)image.ptr();
    for (int i = 0; i < sz; i++) {
        ptr[i] = fabs(ptr[i]);
    }
}*/

void ICP::generatePyramids(cv::Mat &depthMap, cv::Mat &grayImage,int nPhotometricPoints, int nPhotometricReferences) {
    OMPFunctions *multicore = getMultiCoreDevice();
    xyzStride = 11;
    this->nPhotometric           = nPhotometricPoints;
    this->nPhotometricReferences = (nPhotometricReferences < MAX_PHOTOMETRIC_REFERENCES) ? nPhotometricReferences : MAX_PHOTOMETRIC_REFERENCES;
    xyzRef[0]       = cv::Mat(depthMap.rows,depthMap.cols,CV_32FC(xyzStride));
    xyzHiresRef     = cv::Mat(referenceResolutionY,referenceResolutionX,CV_32FC(xyzStride));
    weightsRef      = cv::Mat(referenceResolutionY,referenceResolutionX,CV_32FC1);     memset(weightsRef.ptr(),0,sizeof(float)*referenceResolutionY*referenceResolutionX);
    weightsCur       = cv::Mat(weightsRef.rows,weightsRef.cols,CV_32FC1); memset(weightsCur.ptr(),0,sizeof(float)*weightsRef.rows*weightsRef.cols);
    photoJacobian[0] = cv::Mat(6,nPhotometric,CV_32FC1); memset(photoJacobian[0].ptr(),0,sizeof(float)*nPhotometric*6);
    photoJacobian[1] = cv::Mat(6,nPhotometric,CV_32FC1); memset(photoJacobian[0].ptr(),0,sizeof(float)*nPhotometric*6);
    photoJacobian[2] = cv::Mat(6,nPhotometric,CV_32FC1); memset(photoJacobian[0].ptr(),0,sizeof(float)*nPhotometric*6);
    depthMapCur          = depthMap.clone();
    xyzCur[0]            = cv::Mat(depthMap.rows,depthMap.cols,CV_32FC(xyzStride));
    icpSelectionMask[0]  = cv::Mat(depthMap.rows,depthMap.cols,CV_8UC1);
    photometricSelection = cv::Mat(depthMap.rows,depthMap.cols,CV_32SC1);

    // also allocate temporary "images" for residual and weights:
    residualImage  = cv::Mat(depthMapCur.rows,  depthMapCur.cols,CV_32FC1);
    jacobianImage  = cv::Mat(6,depthMapCur.rows*depthMapCur.cols,CV_32FC1);
    normalStatus   = cv::Mat(depthMapCur.rows,  depthMapCur.cols,CV_8UC1);
    grayCur[0]     = cv::Mat(depthMapCur.rows,  depthMapCur.cols,CV_32FC1);    grayImage.copyTo(grayCur[0]);
    for (int i = 0; i < nPhotometricReferences; i++) {
        grayCurPrev[i][0] = cv::Mat(depthMapCur.rows,  depthMapCur.cols,CV_32FC1); grayImage.copyTo(grayCurPrev[i][0]);
    }

    // initialize both reference&currrent point cloud based on the first measurement
    cv::bilateralFilter(depthMap, depthMapCur, 5, 100.0f, 5.0f);
    scaleIntrinsic(0);
    multicore->generateOrientedPoints(depthMapCur,xyzCur[0],&KL[0],normalStatus,&kc[0],&KR[0],&TLR[0],grayCur[0],xyzStride);
    // initialize incoming frames with with weight 1.0
    initializeWeights(xyzCur[0],weightsCur,1.0f,xyzStride);
    // filter out points out which do not contain new directions/area
    selectPointsICP(xyzCur[0],xyzCur[0].rows*xyzCur[0].cols+1,normalStatus,icpSelectionMask[0],xyzStride);
    selectPointsPhotometric(xyzCur[0],nPhotometric,photometricSelection,xyzStride);
    // compute low-res intensities into xyzCur also (extra attributes only at the highest resolution)
    // multicore->addExtraAttributes(xyzCur[0],nPhotometric,photometricSelection,xyzStride);

    for (int i = 1; i < 3; i++) {
        // instead of allocation, downsample these:
        xyzRef[i]           = cv::Mat(xyzRef[i-1].rows/2,xyzRef[i-1].cols/2,CV_32FC(xyzStride));
        xyzCur[i]           = cv::Mat(xyzCur[i-1].rows/2,xyzCur[i-1].cols/2,CV_32FC(xyzStride));
        icpSelectionMask[i] = cv::Mat(icpSelectionMask[i-1].rows/2,icpSelectionMask[i-1].cols/2,CV_8UC1);
        grayCur[i]          = cv::Mat(xyzCur[i-1].rows/2,  xyzCur[i-1].cols/2,CV_32FC1);
        for (int ri = 0; ri < nPhotometricReferences; ri++) {
            grayCurPrev[ri][i]      = cv::Mat(xyzCur[i-1].rows/2,  xyzCur[i-1].cols/2,CV_32FC1);
        }
        // downsample data to lowres layers
        multicore->downSamplePointCloud(xyzCur[i-1],xyzCur[i],xyzStride);
        multicore->downSampleMask(icpSelectionMask[i-1],icpSelectionMask[i]);
        multicore->downSampleHdrImage(grayCur[i-1],grayCur[i]);
        for (int ri = 0; ri < nPhotometricReferences; ri++) {
            grayCur[i].copyTo(grayCurPrev[ri][i]);
        }
    }
}

void ICP::setMode(bool usePhotometric, bool useICP) {
    if (!usePhotometric && !useICP) return;
    else {
        this->usePhotometric = usePhotometric;
        this->useICP = useICP;
    }
}

void ICP::initializeWeights(cv::Mat &xyzImage, cv::Mat &weights, int stride, float w) {
    // initialize weights
    int npts = xyzImage.rows*xyzImage.cols;
    float *wptr = (float*)weights.ptr();
    float *data = (float*)xyzImage.ptr();
    int offp = 0;
    for (int i = 0; i < npts; i++,offp+=stride) {
        if (data[offp+6]>0) wptr[i] = w;
        else wptr[i] = 0.0f;
    }
}

void normalizeWeights(cv::Mat &weights, float scale) {
    int sz = weights.rows*weights.cols;
    float *w = (float*)weights.ptr();
    float maxW  = 0.0f;
    for (int i = 0; i < sz; i++) if (w[i]>maxW) maxW = w[i];
    for (int i = 0; i < sz; i++) w[i] = scale*w[i]/maxW;
}


void ICP::setData(float *dstData, float *srcData)
{
    for (int i = 0; i < xyzStride; i++) dstData[i] = srcData[i];
}
/*
void ICP::upsampleCloud(Mat &src, Mat &dst) {
    int bw = dst.cols/src.cols;
    int bh = dst.rows/src.rows;
    float *srcPtr = (float*)src.ptr();
    float *dstPtr = (float*)dst.ptr();
    int srcOff = 0;
    for (int j = 0; j < src.rows; j++) {
        for (int i = 0; i < src.cols; i++,srcOff+=xyzStride) {
            int dstOff = (i*bw+j*bh*dst.cols)*xyzStride;
            float *srcData = &srcPtr[srcOff];
            for (int r = 0; r < bh; r++) {
                for (int c = 0; c < bw; c++) {
                    setData(&dstPtr[dstOff+c*xyzStride],srcData);
                }
                dstOff += dst.cols*xyzStride;
            }
        }
    }
}*/

void ICP::upsampleCloud(Mat &src, Mat &dst) {
    int bw = dst.cols/src.cols;
    int bh = dst.rows/src.rows;
    float *srcPtr = (float*)src.ptr();
    float *dstPtr = (float*)dst.ptr();
    int dstOff = 0;
    for (int j = 0; j < dst.rows; j++) {
        for (int i = 0; i < dst.cols; i++,dstOff+=xyzStride) {
            int srcOff = ((i/bw)+(j/bh)*src.cols)*xyzStride;
            float *srcData = &srcPtr[srcOff];
             for (int di = 0; di < xyzStride; di++) dstPtr[dstOff+di] = srcData[di];
 //           setData(&dstPtr[dstOff],srcData);
        }
    }
}

void ICP::updateReference() {  
    if (xyzCur[0].cols == 0) return;

    // copy data from current to reference:
    for (int i = 0; i < 3; i++) {
        xyzCur[i].copyTo(xyzRef[i]);
    }

    if (xyzCur[0].cols == xyzHiresRef.cols) xyzCur[0].copyTo(xyzHiresRef);
    else upsampleCloud(xyzCur[0],xyzHiresRef);
     initializeWeights(xyzRef[0],weightsRef,xyzStride,1.0f);
    //initializeWeights(maskRef[0],weightsRef);
    identity4x4(&T[0]);    // start from identity transform
    nFramesFused = 0;
}

int ICP::getNumberOfPhotometricReferences() {
    int nPotentialReferences = nFramesReceived-1;
    if (nPotentialReferences < nPhotometricReferences) return nPotentialReferences;
    else return nPhotometricReferences;
}

void ICP::updatePhotoReference() {
    // recycle previous gray images and poses
    for (int ri = nPhotometricReferences-1; ri >= 1; ri--) {
        memcpy(&TphotoBase[ri][0],&TphotoBase[ri-1][0],sizeof(float)*16);
        for (int layer = 0; layer < 3; layer++) {
            grayCurPrev[ri-1][layer].copyTo(grayCurPrev[ri][layer]);
        }
    }
    // update freed slot with new reference image
    identity4x4(&TphotoBase[0][0]);
    for (int layer = 0; layer < 3; layer++) grayCur[layer].copyTo(grayCurPrev[0][layer]);
}

unsigned int convertMagnitude(unsigned int magval, bool xDir) {
    unsigned int magval2 = 0;
    if (xDir) magval2 = magval&0xff;
    else magval2 = (magval>>8)&0xff;
    return magval2;
}

int selectPoints(int desiredMass, float *pts, int cnt, int stride, unsigned char *mask, bool xDir, int *select)
{
    int offp = 0;
    unsigned int histogram[256]; memset(&histogram[0],0,256*sizeof(int));
    int totalMass = 0;
    // compute histograms in x or y-direction
    for (int i = 0; i < cnt; i++,offp+=stride) {
        if (pts[offp+6] == 0.0f || mask[i]) continue;
        unsigned int magval2 = convertMagnitude((unsigned int)pts[offp+7],xDir);
        histogram[magval2]++;
        totalMass++;
    }

    // select all points?
    if (desiredMass >= totalMass) {
        printf("totalMass available: %d, desiredAmount: %d!\n",totalMass,desiredMass);
        fflush(stdin); fflush(stdout);
    }

    // compute mass threshold for % of all pixel values
    int currentMass = 0;
    float breakThreshold   = 0;
    for (int i = 255; i >= 1; i--) {
        currentMass += histogram[i];
        if (currentMass > desiredMass) { breakThreshold = float(i); break;}
    }
    // mark % best pixels into mask
    offp = 0;
    int selected = 0;
    for (int i  = 0; i < cnt; i++, offp += stride) {
        if (pts[offp+6] == 0.0f || mask[i]) continue;
            unsigned int magval2 = convertMagnitude((unsigned int)pts[offp+7],xDir);
            if (magval2 > breakThreshold) {
                select[selected] = i; mask[i]=255; selected++;
            }
    }
    // if bin amounts do not match with selection, loop through semi-ok points to get selectNPoints
    if (selected < desiredMass) {
        offp = 0;
        for (int i  = 0; i < cnt; i++, offp += stride) {
            if (pts[offp+6] == 0.0f || mask[i]) continue;
            unsigned int magval2 = convertMagnitude((unsigned int)pts[offp+7],xDir);
            if (magval2 == breakThreshold) { select[selected] = i; mask[i]=255; selected++;  if (selected == desiredMass) return selected; }
        }
    }
    return selected;
    //printf("problemas!\n"); fflush(stdin); fflush(stdout);
}


void ICP::selectPointsPhotometric(cv::Mat &xyzImage, int selectNPoints, cv::Mat &selectionImage, int stride) {
    int cnt = xyzImage.cols*xyzImage.rows;
    int   *select = (int*)selectionImage.ptr();
    float *pts =  (float*)xyzImage.ptr();

    //for (int i = 0; i < cnt; i++) select[i]=i;

    cv::Mat maskImage(xyzImage.rows,xyzImage.cols,CV_8UC1); unsigned char *mask = maskImage.ptr(); memset(mask,0,cnt);
    int selected = 0;
    selected = selectPoints(selectNPoints/2, pts, cnt, stride,mask,true,&select[0]); //printf("selected x : %d\n",selected);
    selected = selectPoints(selectNPoints/2, pts, cnt, stride,mask,false,&select[selectNPoints/2]); //printf("selected y : %d\n",selected);

//    selectPoints(selectNPoints/2, pts, cnt, stride,mask.ptr(),true,&select[0]);
    //selectPoints(selectNPoints/2, pts, cnt, stride,mask.ptr(),false,&select[selectNPoints/2]);
}

void ICP::selectPointsICP(cv::Mat &xyzImage, int selectNPoints, cv::Mat &normalStatus, cv::Mat &selectionImage, int stride) {
    int cnt = normalStatus.cols*normalStatus.rows;
    unsigned char *select = (unsigned char*)selectionImage.ptr();
    float *mask = (float*)xyzImage.ptr();

    int offp = 0;
    for (int i = 0; i < cnt; i++,offp += stride) if (mask[offp+6]>0) select[i] = 255; else select[i] = 0;

    // select all points?
    if (selectNPoints >= cnt) return;


    unsigned char *status  = (unsigned char*)normalStatus.ptr();
    unsigned int histogram[256]; memset(&histogram[0],0,256*sizeof(int));
    for (int i = 0; i < cnt; i++) {
        histogram[status[i]]++;
    }

/*    for (int i = 0; i < 6; i++) {
        printf("normalCnt[%d] : %d\n",i,histogram[i]);
    }
*/
    // mark zero # to invalid bin
    for (int i = 0; i < 256; i+=6) histogram[i] = 0;

    unsigned int validPoints = 0; for (int i = 0; i < 256; i++) validPoints += histogram[i];
    // select all points?
    if (selectNPoints >= validPoints) return;

    // select points cyclically ensuring maximally uniform normals :
    int selectionCounts[256]; memset(&selectionCounts[0],0,256*sizeof(int));
    int currentBin = 1;
    while (selectNPoints >= 0) {
        if (histogram[currentBin]>0) {
            histogram[currentBin]--;
            selectionCounts[currentBin]++;
            selectNPoints--;
        }
        currentBin = (currentBin+1)%256;
    }
    for (int i = 0; i < cnt; i++) {
        int dir = status[i];
        if (selectionCounts[dir] > 0)  {
            selectionCounts[dir]--;
        } else {
            select[i] = 0;
        }
    }
}

void ICP::updatePyramids(cv::Mat &depthMap, cv::Mat &grayImage) {
    OMPFunctions *multicore = getMultiCoreDevice();
    // construct point cloud
    cv::bilateralFilter(depthMap, depthMapCur, 5, 100.0f, 5.0f);
//    depthMap.copyTo(depthMapCur);

    // update current gray image
    grayImage.copyTo(grayCur[0]);
    for (int i = 1; i < 3; i++) multicore->downSampleHdrImage(grayCur[i-1],grayCur[i]);

    scaleIntrinsic(0);
    multicore->generateOrientedPoints(depthMapCur,xyzCur[0],&KL[0],normalStatus,&kc[0],&KR[0],&TLR[0],grayCur[0],xyzStride);
    // initialize incoming frames with with weight 1.0
    initializeWeights(xyzCur[0],weightsCur,xyzStride,1.0f);
    // filter out points out which do not contain new directions/area    
    selectPointsICP(xyzCur[0],xyzCur[0].rows*xyzCur[0].cols+1,normalStatus,icpSelectionMask[0],xyzStride);
    selectPointsPhotometric(xyzCur[0],nPhotometric,photometricSelection,xyzStride);
    for (int i = 0; i < 3; i++) {
        scaleIntrinsic(i);
        multicore->precomputePhotoJacobians(xyzCur[0],&kc[0],&KR[0],&TLR[0],grayCur[i],nPhotometric,photometricSelection,xyzStride, photoJacobian[i],i,scaleIn);
    }
    scaleIntrinsic(0);
    // compute low-res intensities into xyzCur also (extra attributes only at the highest resolution)


    // downsample data to lowres layers
    for (int i = 1; i < 3; i++) {
        multicore->downSamplePointCloud(xyzCur[i-1],xyzCur[i],xyzStride);
        multicore->downSampleMask(icpSelectionMask[i-1],icpSelectionMask[i]);
    }
}

void ICP::setDepthMap(cv::Mat &depthMap, cv::Mat &grayImage, int nPhotometricPoints, int nPhotometricReferences) {
    if (depthMapCur.rows == 0 || depthMapCur.cols == 0)  {
        generatePyramids(depthMap, grayImage,nPhotometricPoints,nPhotometricReferences);
    } else {
        updatePyramids(depthMap, grayImage);
    }
    if (nFramesReceived == 0) {        
        updatePhotoReference();
        updateReference();
    }
    nFramesReceived++;
}

void ICP::setCalib(float *extKL, float *extKR, float *extTLR, float *extKc) {
    memcpy(&KL[0],extKL,sizeof(float)*9);
    memcpy(&origKL[0],extKL,sizeof(float)*9);
    memcpy(&KR[0],extKR,sizeof(float)*9);
    memcpy(&origKR[0],extKR,sizeof(float)*9);
    memcpy(&TLR[0],extTLR,sizeof(float)*16);
    memcpy(&kc[0],extKc,sizeof(float)*5);
}
/*
void ICP::generateWeights(float *residual, float *absResidual, int cnt, float *weights) {
    float medianError = quickMedian(absResidual,cnt);

    OMPFunctions *multicore = getMultiCoreDevice();
    multicore->tukeyWeights()

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


    float s = 1.4826f * float(4*median64[0]+3)/255.0f;
    float b = 4.6851f;

    float u = fabs(r)/s;
    float w = 0.0f;
    if (u <= b) {
        float ub = u / b; float ub2 = ub*ub;
        w = ( 1.0f - ub2 );
    }
    weightedResidual[idx] = r*w;

}
*/

void ICP::cgm(double *A, double *b, double *x)  {
    double r[6];
    double dir[6];
    // add tikhonov regularizer
    for (int i = 0; i < 6; i++) A[i+i*6] += (double)1e-6;
    for (int i = 0; i < 6; i++) { x[i] = 0.0; r[i] = b[i]; dir[i] = b[i]; }
    int nSteps = 0;
    int maxSteps = 12;
    float tol=1e-8;
    double rr = 0;
    while (nSteps < maxSteps) {
        double Adir[6];
        matrixMultVec6CPU(A,dir,Adir);
        //step length
        rr = dotProduct6CPU(r,r);
        if (rr < tol) {
            return;
        }
        double Adirdir = dotProduct6CPU(Adir,dir);
        double stepLength = rr/Adirdir;
        // update x
        for (int i = 0; i < 6; i++) { x[i] += stepLength*dir[i]; r[i] -= stepLength*Adir[i]; }
        double rr2 = dotProduct6CPU(r,r);
        double beta = rr2/rr;
        for (int i = 0; i < 6; i++) { dir[i] = r[i] + beta*dir[i]; }
        nSteps++;
    }
//    printf("residual: %e, step: %d\n",rr,nSteps);
}

void ICP::generateMatrix(double *x, double *Tx) {
    double alpha = x[0]; double cosAlpha = cos(alpha); double sinAlpha = sin(alpha);
    double beta  = x[1]; double cosBeta  = cos(beta);  double sinBeta  = sin(beta);
    double gamma = x[2]; double cosGamma = cos(gamma); double sinGamma = sin(gamma);

    Tx[0] = cosGamma*cosBeta; Tx[1] =-(sinGamma*cosAlpha)+(cosGamma*sinBeta*sinAlpha); Tx[2]  = (sinGamma*sinAlpha) + (cosGamma*sinBeta*cosAlpha); Tx[3]  = x[3]*1.0e3;
    Tx[4] = sinGamma*cosBeta; Tx[5] = (cosGamma*cosAlpha)+(sinGamma*sinBeta*sinAlpha); Tx[6]  =-(cosGamma*sinAlpha) + (sinGamma*sinBeta*cosAlpha); Tx[7]  = x[4]*1.0e3;
    Tx[8] =-sinBeta;          Tx[9] = cosBeta*sinAlpha;                                Tx[10] = cosBeta*cosAlpha;                                  Tx[11] = x[5]*1.0e3;
    Tx[12] = 0.0f;            Tx[13] = 0.0f;                                           Tx[14] = 0.0f;                                              Tx[15] = 1.0f;
}


void ICP::generateMatrixLie(double *x, double *Tx, double scaleOut) {
    double A[16];
    A[0]  = 0;	 A[1]  = -x[2];  A[2]  = x[1];	A[3]  = x[3];
    A[4]  = x[2];A[5]  =     0;	 A[6]  =-x[0];	A[7]  = x[4];
    A[8]  =-x[1];A[9]  =  x[0];  A[10] =    0;	A[11] = x[5];
    A[12] = 0;	 A[13] =     0;	 A[14] =    0;	A[15] =    0;
    expmHost(&A[0],&Tx[0],scaleOut);
}


void ICP::reset() {
    nFramesReceived = 0;
    frame = 0;
    identity4x4(&T[0]);    // start from identity transform
    identity4x4(&Tinc[0]);    // start from identity transform
}

void sumMatrix(double *A, double *B, int cnt, double *C) {
    for (int i = 0; i < cnt; i++) C[i] = A[i]+B[i];
}


void weightedSumMatrix(double *A, double wa, double *B, double wb, int cnt, double *C) {
    for (int i = 0; i < cnt; i++) C[i] = wa*A[i]+wb*B[i];
}

bool ICP::photosGotBetter(double *photoError, double tolerance, double *prevPhotoError,double minimumDecrease) {
    int nPhotoRefs = getNumberOfPhotometricReferences();
    int goodCount = 0;
    int badCount = 0;
    for (int ri = 0; ri < nPhotoRefs; ri++) {
        if (photoError[ri]*tolerance <= prevPhotoError[ri]*minimumDecrease) goodCount++;
        else badCount++;
    }
    return goodCount >= badCount;
}

void ICP::setReferenceResolution(int icpReferenceResoX,int icpReferenceResoY) {
    referenceResolutionX = icpReferenceResoX;
    referenceResolutionY = icpReferenceResoY;
    printf("icp reso set to: %d x %d\n",referenceResolutionX,referenceResolutionY);

    xyzHiresRef = cv::Mat(referenceResolutionY,referenceResolutionX,CV_32FC(xyzStride));
    if (weightsRef.cols > 0) weightsRef.release();
    weightsRef = cv::Mat(referenceResolutionY,referenceResolutionX,CV_32FC1);
    updateReference();
}

void ICP::optimize(int *nIterations, bool verbose) {
    if (nFramesReceived < 2) return;

    // store current T to determine increment from this reference
    float  prevT[16];      invertRT4(&T[0],&prevT[0]);

   // verbose=;

    if (usePhotometric && useICP) optimizeDICP(nIterations,verbose);
    else if (usePhotometric) optimizePhotometric(nIterations,verbose);
    else optimizeGeometric(nIterations,verbose);
    //optimizePhotometric(nIterations,verbose);

    OMPFunctions *multicore = getMultiCoreDevice();
    scaleIntrinsic(-log2(xyzHiresRef.cols/xyzCur[0].cols));
    multicore->refineDepthMap(xyzCur[0], weightsCur, &KL[0], &T[0], xyzHiresRef, weightsRef,xyzStride, 50.0f, 15.0f);
    multicore->generateNormals(xyzHiresRef, xyzStride);
    nFramesFused++;

    //selectXYZRange(xyzRef[0],500,4000,depthCur1C[0]);
    //normalMap3C[0].updateTexture(normalMapRef[0].ptr());

    // downsample data to lowres layers
    for (int i = 0; i < 3; i++) {
        if ( i == 0) {
            // is hires structure actually of same resolution than highest xyzRef?
            if (xyzHiresRef.cols == xyzRef[0].cols) xyzHiresRef.copyTo(xyzRef[0]);
            else {
                //otherwise downsample from hires
                multicore->downSamplePointCloud(xyzHiresRef,xyzRef[0],xyzStride);
            }
        } else {
            multicore->downSamplePointCloud(xyzRef[i-1],xyzRef[i],xyzStride);
        }
        //selectXYZRange(xyzRef[i],500,4000,depthCur1C[i]);
      //  normalMap3C[i].updateTexture(normalMapRef[i].ptr());
    }
    matrixMult4x4(&prevT[0],&T[0],&Tinc[0]);
    frame++;

}

void ICP::optimizePhotometric(int *nIterations, bool verbose) {
    OMPFunctions *multicore = getMultiCoreDevice();

    // NOTE: residualImage and jacobian,are merely large enough scratch memories
    float *residual  = (float*)residualImage.ptr();
    float *jacobian  = (float*)jacobianImage.ptr();
    float *jacobian0 = NULL;

    float  TphotoBaseNew[MAX_PHOTOMETRIC_REFERENCES][16],Tnew[16];
    double A[6*6],Aphoto[6*6];
    double B[6],Bphoto[6];
    double x[6];
    double Tx[16];
    int nPhotoRefs = getNumberOfPhotometricReferences();

    double prevPhotoError[MAX_PHOTOMETRIC_REFERENCES];
    double photoError[MAX_PHOTOMETRIC_REFERENCES];
    double minimumDecrease  = 0.995f;
    float  photometricThreshold = 32.0f;
    int photoCnt = nPhotometric;
    // initialize test matrices using the current transform
    memcpy(&Tnew[0],&T[0],sizeof(float)*16);
    for (int ri = 0; ri < nPhotoRefs; ri++) {
        memcpy(&TphotoBaseNew[ri][0],&TphotoBase[ri][0],sizeof(float)*16);
    }
    for (int layer = 2; layer >= 0; layer--) {
        jacobian0 = (float*)photoJacobian[layer].ptr();
        int iterationCount = nIterations[layer];
        for (int ri = 0; ri < nPhotoRefs; ri++) prevPhotoError[ri] = DBL_MAX;
        if (verbose) printf("LAYER %d\n",layer);
        while (iterationCount >= 0) {
            //// if (iterationCount == 0) dicpMode = true;
            scaleIntrinsic(layer);
            // reset linear system
            memset(&A[0],0,sizeof(double)*36);
            memset(&B[0],0,sizeof(double)*6);
            // collect photometric linear system
            for (int ri = 0; ri < nPhotoRefs; ri++) {
                photoError[ri] = multicore->residualPhotometric(xyzCur[0],photometricSelection,photoCnt,&kc[0],&KR[0],&TLR[0],&TphotoBaseNew[ri][0], grayCurPrev[ri][layer], residual, jacobian0,jacobian,layer,photometricThreshold,xyzStride);
                multicore->Jtresidual(jacobian,residual,photoCnt, 6, &Bphoto[0]);
                multicore->AtA6(jacobian,photoCnt,&Aphoto[0]);
                sumMatrix(&A[0],&Aphoto[0],36,&A[0]);
                sumMatrix(&B[0],&Bphoto[0],6,&B[0]);
            }
            // observe photometrical error before accepting the current increment candidate
            if (photosGotBetter(&photoError[0],1.0f,&prevPhotoError[0],minimumDecrease)) {
                // good increment, store error values as reference and continue working
                for (int ri = 0; ri < nPhotoRefs; ri++)  prevPhotoError[ri] = photoError[ri];
            }  else {
                break;
            }
            // approve this increment
            memcpy(&T[0],&Tnew[0],sizeof(float)*16);
            for (int ri = 0; ri < nPhotoRefs; ri++) {
                memcpy(&TphotoBase[ri][0],&TphotoBaseNew[ri][0],sizeof(float)*16);
            }
            iterationCount--;

            float avgPhotoError = 0.0f;     for (int ri = 0; ri < nPhotoRefs; ri++) avgPhotoError     += photoError[ri];     avgPhotoError     /= float(nPhotoRefs);
            float prevAvgPhotoError = 0.0f; for (int ri = 0; ri < nPhotoRefs; ri++) prevAvgPhotoError += prevPhotoError[ri]; prevAvgPhotoError /= float(nPhotoRefs);

            if (verbose) {
                printf("iteration: %02d, photoerror : %e\n", nIterations[layer]-1-iterationCount, avgPhotoError);
            }
            // estimate next increment
            cgm(&A[0],&B[0],&x[0]);
            generateMatrixLie(&x[0],&Tx[0],scaleOut);
            matrixMult4x4(&T[0],&Tx[0],&Tnew[0]);
            for (int ri = 0; ri < nPhotoRefs; ri++) {
                matrixMult4x4(&TphotoBase[ri][0],&Tx[0],&TphotoBaseNew[ri][0]);
            }
        }
    }
    // test last increment
    // collect photometric linear system
    for (int ri = 0; ri < nPhotoRefs; ri++) {
        int layer = 0;
        photoError[ri] = multicore->residualPhotometric(xyzCur[0],photometricSelection,photoCnt,&kc[0],&KR[0],&TLR[0],&TphotoBaseNew[ri][0], grayCurPrev[ri][layer], residual, jacobian0,jacobian,layer,photometricThreshold,xyzStride);
    }
    if (photosGotBetter(&photoError[0],1.0f,&prevPhotoError[0],minimumDecrease)) {
        // approve the last increment
        memcpy(&T[0],&Tnew[0],sizeof(float)*16);
        for (int ri = 0; ri < nPhotoRefs; ri++) {
            memcpy(&TphotoBase[ri][0],&TphotoBaseNew[ri][0],sizeof(float)*16);
        }
        printf("iteration: %02d, photoerror : %e\n", nIterations[0], float(photoError[0]));
    }
}

void ICP::optimizeGeometric(int *nIterations, bool verbose) {
    OMPFunctions *multicore = getMultiCoreDevice();

    float  Tnew[16];
    double A[6*6];
    double B[6];
    double x[6];
    double Tx[16];

    // NOTE: residualImage and jacobian,are merely large enough scratch memories
    float *residual  = (float*)residualImage.ptr();
    float *jacobian  = (float*)jacobianImage.ptr();
//  JtJ =  [Jz^T lambda*Ji^T][Jz]    = [Jz^TJz + lambda^2*Ji^T*Ji]
//                    [lambda*Ji]
//  b   = [Jz^T lambda*Ji^T][ez; lambda*ei] =  Jz^Tez + lambda*lambda*Ji^T*ei
    double lambda2 = 1e-9f;
    double prevErrorICP = DBL_MAX;
    float  depthThreshold = 300.0f;
    // initialize test matrices using the current transform
    memcpy(&Tnew[0],&T[0],sizeof(float)*16);
    double icpError = DBL_MAX;
    for (int layer = 2; layer >= 0; layer--) {
        icpError = DBL_MAX;
        int icpCnt = xyzRef[layer].cols*xyzRef[layer].rows;
        int iterationCount = nIterations[layer];
        prevErrorICP   = DBL_MAX;
        if (verbose) printf("LAYER %d\n",layer);
        while (iterationCount >= 0) {
            scaleIntrinsic(0);
            // reset linear system
            memset(&A[0],0,sizeof(double)*36);
            memset(&B[0],0,sizeof(double)*6);
            // collect photometric linear system
            scaleIntrinsic(0);
            icpError = multicore->residualICP(xyzCur[layer], icpSelectionMask[layer], &KL[0], &Tnew[0], xyzRef[0], residual, jacobian, scaleIn, depthThreshold,xyzStride);
            // observe geometric error before accepting the current increment candidate
            if (icpError < prevErrorICP) {
                // good increment, store error values as reference and continue working
                prevErrorICP = icpError;
            }  else {
                break;
            }
            multicore->Jtresidual(jacobian,residual,icpCnt, 6, &B[0]);
            multicore->AtA6(jacobian,icpCnt,&A[0]);
            // approve this increment
            memcpy(&T[0],&Tnew[0],sizeof(float)*16);

            iterationCount--;

            if (verbose) {
                 printf("iteration: %02d, icperror : %e\n", nIterations[layer]-1-iterationCount, icpError);
            }
            // estimate next increment
            cgm(&A[0],&B[0],&x[0]);
            generateMatrixLie(&x[0],&Tx[0],scaleOut);
            matrixMult4x4(&T[0],&Tx[0],&Tnew[0]);
        }
    }

    int layer = 0;
    scaleIntrinsic(0);
    icpError = multicore->residualICP(xyzCur[layer], icpSelectionMask[layer], &KL[0], &Tnew[0], xyzRef[0], residual, jacobian, scaleIn, depthThreshold,xyzStride);

    if (icpError < prevErrorICP) {
        // approve the last increment
        memcpy(&T[0],&Tnew[0],sizeof(float)*16);
        printf("iteration: %02d, icperror : %e\n", nIterations[0], float(icpError));
    }
}

void ICP::optimizeDICP(int *nIterations, bool verbose) {
    OMPFunctions *multicore = getMultiCoreDevice();

    float  TphotoBaseNew[MAX_PHOTOMETRIC_REFERENCES][16],Tnew[16];
    double A[6*6],Aphoto[6*6],Aicp[6*6];
    double B[6],Bphoto[6],Bicp[6];
    double x[6];
    double Tx[16];
    int nPhotoRefs = getNumberOfPhotometricReferences();

    // NOTE: residualImage and jacobian,are merely large enough scratch memories
    float *residual  = (float*)residualImage.ptr();
    float *jacobian  = (float*)jacobianImage.ptr();
    float *jacobian0 = NULL;
//  JtJ =  [Jz^T lambda*Ji^T][Jz]    = [Jz^TJz + lambda^2*Ji^T*Ji]
//                    [lambda*Ji]
//  b   = [Jz^T lambda*Ji^T][ez; lambda*ei] =  Jz^Tez + lambda*lambda*Ji^T*ei

    double lambda2 = 1e-9f;
    double prevPhotoError[MAX_PHOTOMETRIC_REFERENCES];
    double photoError[MAX_PHOTOMETRIC_REFERENCES];
    double prevErrorICP = DBL_MAX;
    double photometricToleranceWithICP = 0.995f;//1.0f;//1.0f;//0.9f;
    double minimumDecrease  = 0.995f;
    float  photometricThreshold = 32.0f;
    float  depthThreshold = 300.0f;
    int photoCnt = nPhotometric;
    // initialize test matrices using the current transform
    memcpy(&Tnew[0],&T[0],sizeof(float)*16);
    for (int ri = 0; ri < nPhotoRefs; ri++) {
        memcpy(&TphotoBaseNew[ri][0],&TphotoBase[ri][0],sizeof(float)*16);
    }
    bool dicpMode = false;
    for (int layer = 2; layer >= 0; layer--) {
        double icpError = DBL_MAX;
        jacobian0 = (float*)photoJacobian[layer].ptr();
        int icpCnt = xyzRef[layer].cols*xyzRef[layer].rows;
        dicpMode = false;
        int iterationCount = nIterations[layer];
        for (int ri = 0; ri < nPhotoRefs; ri++) prevPhotoError[ri] = DBL_MAX;
        prevErrorICP   = DBL_MAX;
        if (verbose) printf("LAYER %d\n",layer);
        while (iterationCount >= 0) {
            if (iterationCount == 0) dicpMode = true;
            scaleIntrinsic(layer);
            // reset linear system
            memset(&A[0],0,sizeof(double)*36);
            memset(&B[0],0,sizeof(double)*6);
            // collect photometric linear system
            for (int ri = 0; ri < nPhotoRefs; ri++) {
                photoError[ri] = multicore->residualPhotometric(xyzCur[0],photometricSelection,photoCnt,&kc[0],&KR[0],&TLR[0],&TphotoBaseNew[ri][0], grayCurPrev[ri][layer], residual, jacobian0,jacobian,layer,photometricThreshold,xyzStride);
                multicore->Jtresidual(jacobian,residual,photoCnt, 6, &Bphoto[0]);
                multicore->AtA6(jacobian,photoCnt,&Aphoto[0]);
                sumMatrix(&A[0],&Aphoto[0],36,&A[0]);
                sumMatrix(&B[0],&Bphoto[0],6,&B[0]);
            }
            // observe photometrical error before accepting the current increment candidate
            bool iterationOk = false;
            if (!dicpMode) {
                if (photosGotBetter(&photoError[0],1.0f,&prevPhotoError[0],minimumDecrease)) {
                    // good increment, store error values as reference and continue working
                    for (int ri = 0; ri < nPhotoRefs; ri++)  prevPhotoError[ri] = photoError[ri];
                }  else {
                    // bad increment, switch to dicp mode, use previous transformations and return to previous iteration
                    dicpMode = true;
                    // return back one step and continue using dicp
                    memcpy(&Tnew[0],&T[0],sizeof(float)*16);
                    for (int ri = 0; ri < nPhotoRefs; ri++) {
                        memcpy(&TphotoBaseNew[ri][0],&TphotoBase[ri][0],sizeof(float)*16);
                    }
                    // re-iterate this:
                    iterationCount++;
                    continue;
                }
            } else {
                scaleIntrinsic(0);
                icpError = multicore->residualICP(xyzCur[layer], icpSelectionMask[layer], &KL[0], &Tnew[0], xyzRef[0], residual, jacobian, scaleIn, depthThreshold,xyzStride);
                // after stepping back, first icp evaluation always succeeds:
                iterationOk = photosGotBetter(&photoError[0],photometricToleranceWithICP,&prevPhotoError[0],1.0f) && (icpError < prevErrorICP*minimumDecrease);
                //printf("testing icp error %02d, photoerror : %e, photoanchor: %e, icperror: %e, dicpMode: %d, icpapproved: %d, photoapproved:%d, totalapproved: %d\n", nIterations[layer]-1-iterationCount, float(photoError[0]),float(prevPhotoError[0]),icpError,int(dicpMode),(icpError < prevErrorICP*minimumDecrease),photosGotBetter(&photoError[0],photometricToleranceWithICP,&prevPhotoError[0],1.0f),int(iterationOk));
                if (!iterationOk) {
                    // we have tried everything but doesnt work, lets quit this layer:
                    break;
                }
                multicore->Jtresidual(jacobian,residual,icpCnt, 6, &Bicp[0]);
                multicore->AtA6(jacobian,icpCnt,&Aicp[0]);
                weightedSumMatrix(&Aicp[0],1.0f,&A[0], lambda2/nPhotoRefs, 36,&A[0]);
                weightedSumMatrix(&Bicp[0],1.0f,&B[0], lambda2/nPhotoRefs, 6, &B[0]);
                prevErrorICP   = icpError;
            }
            // approve this increment
            memcpy(&T[0],&Tnew[0],sizeof(float)*16);
            for (int ri = 0; ri < nPhotoRefs; ri++) {
                memcpy(&TphotoBase[ri][0],&TphotoBaseNew[ri][0],sizeof(float)*16);
            }
            iterationCount--;

            float avgPhotoError = 0.0f;     for (int ri = 0; ri < nPhotoRefs; ri++) avgPhotoError     += photoError[ri];     avgPhotoError     /= float(nPhotoRefs);
            float prevAvgPhotoError = 0.0f; for (int ri = 0; ri < nPhotoRefs; ri++) prevAvgPhotoError += prevPhotoError[ri]; prevAvgPhotoError /= float(nPhotoRefs);

            if (verbose) {
                if (!dicpMode) {
                    printf("iteration: %02d, photoerror : %e, dicpMode: %d\n", nIterations[layer]-1-iterationCount, avgPhotoError,int(dicpMode));
                } else {
                    printf("iteration: %02d, photoerror : %e, photoanchor: %e, icperror: %e, dicpMode: %d\n", nIterations[layer]-1-iterationCount, avgPhotoError,prevAvgPhotoError,icpError,int(dicpMode));
                }
            }
            // estimate next increment
            cgm(&A[0],&B[0],&x[0]);
            generateMatrixLie(&x[0],&Tx[0],scaleOut);
            matrixMult4x4(&T[0],&Tx[0],&Tnew[0]);
            for (int ri = 0; ri < nPhotoRefs; ri++) {
                matrixMult4x4(&TphotoBase[ri][0],&Tx[0],&TphotoBaseNew[ri][0]);
            }
        }
    }
    // test last increment
    // collect photometric linear system
    scaleIntrinsic(0);
    for (int ri = 0; ri < nPhotoRefs; ri++) {
        int layer = 0;
        photoError[ri] = multicore->residualPhotometric(xyzCur[0],photometricSelection,photoCnt,&kc[0],&KR[0],&TLR[0],&TphotoBaseNew[ri][0], grayCurPrev[ri][layer], residual, jacobian0,jacobian,layer,photometricThreshold,xyzStride);
    }
    if (photosGotBetter(&photoError[0],1.0f,&prevPhotoError[0],minimumDecrease)) {
        // approve the last increment
        memcpy(&T[0],&Tnew[0],sizeof(float)*16);
        for (int ri = 0; ri < nPhotoRefs; ri++) {
            memcpy(&TphotoBase[ri][0],&TphotoBaseNew[ri][0],sizeof(float)*16);
        }
        printf("iteration: %02d, photoerror : %e, dicpMode: %d\n", nIterations[0], float(photoError[0]),int(dicpMode));
    }
}

void ICP::getReferenceCloud(cv::Mat **xyzImage, int *stride) {
    *xyzImage = &xyzHiresRef;
    *stride = xyzStride;
}

float *ICP::getBaseTransform() {
    return &T[0];
}


float *ICP::getIncrement() {
    return &Tinc[0];
}

void ICP::scaleIntrinsic(int layer) {    
    float a = 1.0f/pow(2.0,layer);
    float b = 0.5f*(a-1.0f);

    KL[0] = origKL[0]*a; KL[1] = origKL[1]*a; KL[2] = origKL[2]*a+b;
    KL[3] = origKL[3]*a; KL[4] = origKL[4]*a; KL[5] = origKL[5]*a+b;

    KR[0] = origKR[0]*a; KR[1] = origKR[1]*a; KR[2] = origKR[2]*a+b;
    KR[3] = origKR[3]*a; KR[4] = origKR[4]*a; KR[5] = origKR[5]*a+b;

    //[a 0 b][k11 k12 k13] [x]
    //[0 a b][k21 k22 k23] [y]
    //[0 0 1][0    0    1] [1]
}

void ICP::markPhotometricSelection(cv::Mat &rgbImage, int colorR, int colorG, int colorB) {
    // initialize image with current gray values
    unsigned char *rgb = rgbImage.ptr();
    float *gray = (float*)grayCur[0].ptr();
    int sz = grayCur[0].cols*grayCur[0].rows;
    for (int i = 0; i < sz; i++) { rgb[i*3+0] = (unsigned char)gray[i]; rgb[i*3+1] = (unsigned char)gray[i]; rgb[i*3+2] = (unsigned char)gray[i]; }

    // mark selected pixels as overlay graphics
    int *select = (int*)photometricSelection.ptr();
    float *ptsCur = (float*)xyzCur[0].ptr();
    scaleIntrinsic(0);
    for (int i = 0; i < nPhotometric; i++)  {
        int j = select[i];
        float *p = &ptsCur[j*xyzStride+0]; float r[3],pd[3];
        transformRT3(TLR,p,r); r[0] /= r[2]; r[1] /= r[2]; r[2] = 1.0f;
        distortPointCPU(&r[0],&kc[0],&KR[0],&pd[0]);
        int xi = pd[0];
        int yi = pd[1];
        if (xi >= 0 && xi < grayCur[0].cols && yi >= 0 && yi < grayCur[0].rows) {
            int offset = xi+yi*grayCur[0].cols;
            rgb[offset*3+0] = colorR; rgb[offset*3+1] = colorG; rgb[offset*3+2] = colorB;
        }
    }
}

/*
cv::Mat Ain(6,6,CV_64FC1); double *src = (double*)Ain.ptr();
for (int i = 0; i < 36; i++) src[i] = A[i];
cv::Mat E, V;
cv::eigen(Ain,E,V);
double minEigen = DBL_MAX;
for (int i = 0; i < 6; i++)
    if (fabs(E.at<double>(i)) < minEigen) minEigen = fabs(E.at<double>(i));
printf("eigenvalue: %e, frame: %d, layer: %d, iteration: %d\n",minEigen,frame,layer,n);
*/


/*            cv::Mat Aout(6,6,CV_64FC1); double *dst = (double*)Aout.ptr();
cv::Mat Bout(6,1,CV_64FC1); double *bout = (double*)Bout.ptr();
cv::Mat Xout(6,1,CV_64FC1); double *xout = (double*)Xout.ptr();


for (int i = 0; i < 6; i++) bout[i] = b[i];
for (int i = 0; i < 6; i++) xout[i] = x[i];

double stability = cv::invert(Ain,Aout,DECOMP_SVD);
if (stability < stabilityValue) {
    stabilityValue = stability;
    printf("frame: %d, stability : %e, iteration: %d, layer: %d\n",frame,stabilityValue,n,layer); fflush(stdin); fflush(stdout);
}

Xout = Aout*Bout;
for (int i = 0; i < 6; i++) x[i] = xout[i];*/
//  printf("layer: %d, iteration: %d\n",layer,n);
//generateMatrix(&x[0],&Tx[0]);
