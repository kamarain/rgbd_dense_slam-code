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

#include "configuration.h"
#include <opencv2/opencv.hpp>
#include <image2/ImagePyramid2.h>
#include <reconstruct/basic_math.h>
#include <multicore/multicore.h>

Configuration::Configuration() {
	referenceFrame = 0;
        nBins = 0;
        zAccumulator = NULL;
        counterImage = NULL;
        rgbReference = NULL;
        rgbMask      = NULL;
        stdevImage   = NULL;
        rgbGradient  = NULL;
        identity4x4(&refT[0]);
        identity3x3(&refK[0]);
}

Configuration::~Configuration() {
        if (zAccumulator != NULL) delete[] zAccumulator;
        if (counterImage != NULL) delete[] counterImage;
        if (rgbReference != NULL) delete[] rgbReference;
        if (rgbGradient != NULL) delete[] rgbGradient;
        if (rgbMask != NULL) delete[] rgbMask;
        if (stdevImage != NULL) delete[] stdevImage;
        clearNeighbors();
}


/*
void improveStructureMedian(int nViews, cv::Mat &rangeImage) {
        KeyFrame *refKey = testConfiguration.getKeyFrame(0);
        if (refKey == NULL || nViews >= testConfiguration.getKeyFrameCount()) return;
        Image *depthMap = refKey->depthMap;
        float *refData = (float*)depthMap->data;
        int fw = depthMap->width;
        int fh = depthMap->height;
        Image *refTexture0 = refKey->texture->getImagePtr(0);
        float *rangeData = (float*)rangeImage.ptr();
        timer.StartCounter();

        // allocate enough scratch space for storing all depths of back-warped 3d points
        int maxHits = 16*(1+nViews);
        float *zAccumulator = new float[maxHits*fw*fh];
        unsigned char *counterImage = new unsigned char[fw*fh];
        memset(zAccumulator,0,sizeof(float)*fw*fh*maxHits);
        unsigned char *refMask = refKey->mask->data;
        int offset = 0;
        for (int j = 0; j < fh; j++) {
                for (int i = 0; i < fw; i++,offset++) {
                        if (refMask[offset] == 0) continue;
                        zAccumulator[offset*maxHits] = refData[offset];
                        counterImage[offset] = 1;
                }
        }

        for (int k = 1; k < nViews; k++) {
                KeyFrame *curKey = testConfiguration.getKeyFrame(k);
                unsigned char *mask = curKey->mask->data;
                Image *depthMap = curKey->depthMap;
                float *data = (float*)depthMap->data;
                float iK[9];
                inverse3x3(curKey->K,iK);
                float P[16],Tz[4];
                float pc[3],xc[2],v[4],z;
                testConfiguration.projectInitZ(k, 0, P, Tz);
                int offset = 0;
                for (int j = 0; j < fh; j++) {
                        for (int i = 0; i < fw; i++,offset++) {
                                if (mask[offset]==0) continue;
                                float zf = data[offset];
                                if (zf == 0.0f) continue;
                                get3DPoint(float(i),float(j), zf, iK, &v[0], &v[1], &v[2]);
                                testConfiguration.projectFastZ(v,pc,&z,P,Tz);
                                unsigned int x = unsigned int(pc[0]);
                                unsigned int y = unsigned int(pc[1]);
                                if (x > fw-2) continue;
                                if (y > fh-2) continue;
                                int offset = x+y*fw;
                                unsigned char &i0 = counterImage[offset];
                                unsigned char &i1 = counterImage[offset+1];
                                unsigned char &i2 = counterImage[offset+1+fw];
                                unsigned char &i3 = counterImage[offset+fw];
                                if (i0 < maxHits) { zAccumulator[(offset+0)*maxHits+i0]    = -z; i0++; }
                                if (i1 < maxHits) { zAccumulator[(offset+1)*maxHits+i1]    = -z; i1++; }
                                if (i2 < maxHits) { zAccumulator[(offset+1+fw)*maxHits+i2] = -z; i2++; }
                                if (i3 < maxHits) { zAccumulator[(offset+fw)*maxHits+i3]   = -z; i3++; }
                        }
                }
        }

        offset=0;
        for (int j = 0; j < fh; j++) {
                for (int i = 0; i < fw; i++,offset++) {
                        if (refMask[offset]==0) continue;
                        refData[offset] = quickMedian(&zAccumulator[offset*maxHits],counterImage[offset]);
                }
        }

        // free scratch space
        delete[] zAccumulator;
        delete[] counterImage;

        timer.StopCounter();
        double elapsedTime = timer.GetElapsedTime()*1000;
        printf("elapsedTime: %f\n",elapsedTime);
        setupDepthRefinementRange2(depthMap, &depthGradientImage, pixelSelectionMask,rangeImage);
        uploadImage(depthMap);
}
*/
/*
inline void get3DPoint(float x, float y, float z, float *iK, float *xc, float *yc, float *zc) {
        float pd[4],cd[4];
        pd[0] = x; pd[1] = y; pd[2] = 1;
        matrixMultVec3(iK,pd,cd);
        //float t = z/cd[2];
        *xc = -cd[0]*z;
        *yc = -cd[1]*z;
        *zc = -z;
}*/


float Configuration::average(float *arr, int n) {
    if (n == 0) return 0;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += arr[i];
    return sum / n;
}

float Configuration::robustAverage(float *arr, int n, float medianZ, float robustDistance2, float *depthStdev) {
    float weightSum = 0.0f;
    float robustZ = 0.0f;
    for (int i = 0; i < n; i++) {
        float z = arr[i];
        float zerr = z-medianZ;
        if (zerr*zerr < robustDistance2) {
            robustZ += z;
            weightSum += 1.0f;
        }
    }

    robustZ /= weightSum;

    float sumDeviation = 0.0f;
    for (int i = 0; i < n; i++) {
        float z = arr[i];
        float zerr = z-medianZ;
        if (zerr*zerr < robustDistance2) {
            float dev = robustZ - z;
            sumDeviation += dev*dev;
        }
    }
    *depthStdev = sqrt(sumDeviation/weightSum+1e-7f);
    return robustZ;
}

float Configuration::robustInvAverage(float *arr, int n, float medianZ, float robustDistance2, float *depthStdev) {
    float weightSum = 0.0f;
    float invRobustZSum = 0.0f;
    for (int i = 0; i < n; i++) {
        float z = arr[i];
        float zerr = z-medianZ;
        if (zerr*zerr < robustDistance2) {
            invRobustZSum += (1.0f/z);
            weightSum += 1.0f;
        }
    }

    float invAvg = invRobustZSum / weightSum;
    float robustZ = 1.0f/invAvg;
    float sumDeviation = 0.0f;
    for (int i = 0; i < n; i++) {
        float z = arr[i];
        float zerr = z-medianZ;
        if (zerr*zerr < robustDistance2) {
            float dev = robustZ - z;
            sumDeviation += dev*dev;
        }
    }
    *depthStdev = sqrt(sumDeviation/weightSum+1e-7f);
    return robustZ;
}

void Configuration::clearNeighbors() {
    std::vector<cv::Mat *>::iterator ni;
    for (ni = neighborImages.begin(); ni != neighborImages.end(); ni++) {
        cv::Mat *rgbImage = *ni;
        //sprintf(buf,"scratch/neighborImage%04d.ppm",i);
        //imwrite(buf,*rgbImage);
        rgbImage->release();
    }
    neighborImages.clear();
    neighborPoses.clear();
}

void Configuration::generatePixelSelectionMask(unsigned char *rgbGradientData, float *refData, int width, int height, int pixelSelectionThreshold, unsigned char *mask) {

    memset(mask,0,width*height);

    int pitch = width*3;
    int border = 3;
    for (int y = border; y < (height-border); y++) {
        for (int x = border; x < (width-border); x++) {
            int grayoffset = x + y*width;
            if (fabs(refData[grayoffset]) <= 500.0f) continue;
            if (fabs(refData[grayoffset-1]) <= 500.0f) continue;
            if (fabs(refData[grayoffset+1]) <= 500.0f) continue;
            if (fabs(refData[grayoffset-width]) <= 500.0f) continue;
            if (fabs(refData[grayoffset+width]) <= 500.0f) continue;

            int offset = x*3 + y*pitch;
            int grad = (rgbGradientData[offset+0]+rgbGradientData[offset+1]+rgbGradientData[offset+2])/3;
            if (grad > pixelSelectionThreshold) {
                mask[grayoffset] = 255;
            }
        }
    }
}


void colorSelection(unsigned char *rgbReference, unsigned char *rgbMask, cv::Mat &result, int r, int g, int b) {
    int fw = result.cols; int fh = result.rows;
    unsigned char *output = result.ptr();
    int size = fh*fw;
    for (int i = 0; i < size; i++) {
        if (rgbMask[i]>0) {
            output[i*3+0] = r;
            output[i*3+1] = g;
            output[i*3+2] = b;
        } else {
            output[i*3+0] = rgbReference[i*3+0];
            output[i*3+1] = rgbReference[i*3+1];
            output[i*3+2] = rgbReference[i*3+2];
        }
    }
}

void hdr2Gray(float *stdev, float minRange, float maxRange, cv::Mat &result)  {
    int fw = result.cols; int fh = result.rows;
    unsigned char *output = result.ptr();
    int size = fh*fw;
    for (int i = 0; i < size; i++) {
        float val = stdev[i];
        if (val < minRange) val = minRange;
        if (val > maxRange) val = maxRange;
        val /= (maxRange-minRange);
        val *= 255;
        output[i] = (unsigned char )val;
    }
}

void Configuration::init(float *refT, cv::Mat *refMap, float *TLR, float *refKir, float *refKrgb, float *kcRGB, int nBins, cv::Mat *rgbMap) {
    int fw = refMap->cols;
    int fh = refMap->rows;
    float *refData = (float*)refMap->ptr();
    unsigned char *rgbData = rgbMap->ptr();
    this->nBins = nBins;
    memcpy(&this->refT[0],refT,sizeof(float)*16);
    memcpy(&this->refK[0],refKir,sizeof(float)*9);
    float iKir[9]; inverse3x3(refKir,iKir);

    if (zAccumulator != NULL) delete[] zAccumulator; zAccumulator = new float[nBins*fw*fh];
    if (counterImage != NULL) delete[] counterImage; counterImage = new int[fw*fh];
    if (rgbReference != NULL) delete[] rgbReference; rgbReference = new unsigned char[fw*fh*3];
    if (rgbGradient != NULL) delete[] rgbGradient;   rgbGradient = new unsigned char[fw*fh*3];
    if (rgbMask != NULL)      delete[] rgbMask;      rgbMask      = new unsigned char[fw*fh];
    if (stdevImage != NULL)   delete[] stdevImage;   stdevImage   = new float[fw*fh];

    clearNeighbors();

    int offset = 0;
    int offset3 = 0;
    for (int j = 0; j < fh; j++) {
        for (int i = 0; i < fw; i++,offset++,offset3+=3) {
            if (refData[offset] > 0) {
                float zf = refData[offset];
                zAccumulator[offset*nBins] = zf;
                float v[3],w[3],p2[3];
                get3DPoint(float(i),float(j),zf, iKir, &v[0], &v[1], &v[2]);
                transformRT3(TLR,v,w); w[0] = w[0]/w[2]; w[1] = w[1]/w[2]; w[2] = 1.0f;
                distortPointCPU(w,kcRGB,refKrgb,p2);
                unsigned char r,g,b;
                interpolateRGBPixel(rgbData,fw,fh,p2[0],p2[1],&r,&g,&b);
                rgbReference[offset3+0] = r;
                rgbReference[offset3+1] = g;
                rgbReference[offset3+2] = b;
                unsigned char rh1,gh1,bh1;
                interpolateRGBPixel(rgbData,fw,fh,p2[0]+1.0f,p2[1],&rh1,&gh1,&bh1);
                unsigned char rh2,gh2,bh2;
                interpolateRGBPixel(rgbData,fw,fh,p2[0]-1.0f,p2[1],&rh2,&gh2,&bh2);
                unsigned char rv1,gv1,bv1;
                interpolateRGBPixel(rgbData,fw,fh,p2[0],p2[1]+1.0f,&rv1,&gv1,&bv1);
                unsigned char rv2,gv2,bv2;
                interpolateRGBPixel(rgbData,fw,fh,p2[0],p2[1]-1.0f,&rv2,&gv2,&bv2);
                rgbGradient[offset3+0] = (abs(rh1-rh2)+abs(rv1-rv2))/2;
                rgbGradient[offset3+1] = (abs(gh1-gh2)+abs(gv1-gv2))/2;
                rgbGradient[offset3+2] = (abs(bh1-bh2)+abs(bv1-bv2))/2;
                counterImage[offset] = 1;
            } else {
                zAccumulator[offset*nBins] = 0;
                counterImage[offset] = 0;
                rgbReference[offset3+0] = 0;
                rgbReference[offset3+1] = 0;
                rgbReference[offset3+2] = 0;
                rgbGradient[offset3+0] = 0;
                rgbGradient[offset3+1] = 0;
                rgbGradient[offset3+2] = 0;
            }
            stdevImage[offset] = 0.0f;
        }
    }
    generatePixelSelectionMask(&rgbGradient[0],&refData[0],fw,fh,10,rgbMask);
/*
    static int joo = 0; joo++;
    cv::Mat neighborImage(fh,fw,CV_8UC3);//,rgbMask);
    colorSelection(rgbReference,rgbMask,neighborImage,0,255,0);
    char buf[512];
    sprintf(buf,"scratch/pixelselect%d.png",joo);
    imwrite(buf,neighborImage);
    */
}

void Configuration::storeNeighbor(float *curT, float *TLR, cv::Mat *rgbMap) {

    unsigned char *rgbData = rgbMap->ptr();
    float m16[16];

    // from ref IR -> cur IR                    // cur IR -> cur RGB
    relativeTransform(refT, curT, &m16[0]);  matrixMult4x4(TLR,&m16[0],&m16[0]);
    // insert pose matrix into neighbor pose array
    vector<float> poseMat(&m16[0], &m16[16] ); neighborPoses.push_back(poseMat);
    // enumerate all neighbor images for image based optimization
    cv::Mat *neighborImage = new cv::Mat(rgbMap->rows,rgbMap->cols,CV_8UC3);
    memcpy(neighborImage->ptr(),rgbData,rgbMap->rows*rgbMap->cols*3);
    neighborImages.push_back(neighborImage);
}

bool Configuration::sanityCheckPoint(int refX, int refY, int width, unsigned char curR, unsigned char curG,unsigned char curB, float *iKir, float *p3, int colorThreshold, float rayThreshold2, int &dstOffset)
{
    bool colorBit = true;

    // check rgb value against reference
    dstOffset = refX+refY*width;
    int refR = rgbReference[dstOffset*3+0];
    int refG = rgbReference[dstOffset*3+1];
    int refB = rgbReference[dstOffset*3+2];
    if (abs(refR-curR)+abs(refG-curG)+abs(refB-curB) > colorThreshold) colorBit = false;

    // check distance to ray though pixel center
    float ray[3],dev[3],rayComponent,rayDist2;
    get3DRay(refX,refY,iKir,&ray[0],&ray[1],&ray[2]);
    rayComponent = ray[0]*p3[0]+ray[1]*p3[1]+ray[2]*p3[2];
    ray[0] *= rayComponent;
    ray[1] *= rayComponent;
    ray[2] *= rayComponent;
    dev[0] = p3[0] - ray[0];
    dev[1] = p3[1] - ray[1];
    dev[2] = p3[2] - ray[2];
    rayDist2 = dev[0]*dev[0]+dev[1]*dev[1]+dev[2]*dev[2];
    if (rayDist2 > rayThreshold2) colorBit = false;

    return colorBit;
}

void Configuration::warpToReference(float *curT, cv::Mat *curMap, float *TLR, float *Kir, float *Krgb, float *kc, cv::Mat *rgbMap, int colorThreshold, float rayThreshold) {
    if (curT == NULL || curMap == NULL || Kir == NULL || Krgb == NULL || kc == NULL) { printf("configuration: no keyframes found!\n"); return;}
    if (zAccumulator == NULL || counterImage == NULL) return;
    int fw = curMap->cols;
    int fh = curMap->rows;
    float *data = (float*)curMap->ptr();
    unsigned char *rgbData = rgbMap->ptr();
    float iKir[9]; inverse3x3(Kir,iKir);
    float iKrgb[9];inverse3x3(Krgb,iKrgb);
    float P[16],Tx[4],Ty[4],Tz[4];
    float pc[3],v[4],w[4],p3[3],p2[3];
    float rayThreshold2 = rayThreshold*rayThreshold;
    // STORE NEIGHBOR IMAGES + POSES INTO INTERNAL ARRAY
    storeNeighbor(curT,TLR, rgbMap);

    projectInitXYZ(curT, refK, refT, P, Tx,Ty,Tz);

    int offset = 0, offset3 = 0;
    for (int j = 0; j < fh; j++) {
        for (int i = 0; i < fw; i++,offset++,offset3+=3) {
            float zf = data[offset];
            if (zf <= 0.0f) continue;
            get3DPoint(float(i),float(j),zf, iKir, &v[0], &v[1], &v[2]);
            transformRT3(TLR,v,w); w[0] = w[0]/w[2]; w[1] = w[1]/w[2]; w[2] = 1.0f;
            distortPointCPU(w,kc,Krgb,p2);

            unsigned char r,g,b;
            interpolateRGBPixel(rgbData,fw,fh,p2[0],p2[1],&r,&g,&b);

            // project point in current IR view into reference IR view
            projectFastXYZ(v,pc,&p3[0],P,Tx,Ty,Tz);

            int x = (int)(pc[0]); if (x < 0 || x > fw-2) continue;
            int y = (int)(pc[1]); if (y < 0 || y > fh-2) continue;

            bool colorBit[4];
            int dstOffset0,dstOffset1,dstOffset2,dstOffset3;
            colorBit[0] = sanityCheckPoint(x+0,y+0, fw, r,g,b, iKir, p3, colorThreshold, rayThreshold2, dstOffset0);
            colorBit[1] = sanityCheckPoint(x+1,y+0, fw, r,g,b, iKir, p3, colorThreshold, rayThreshold2, dstOffset1);
            colorBit[2] = sanityCheckPoint(x+1,y+1, fw, r,g,b, iKir, p3, colorThreshold, rayThreshold2, dstOffset2);
            colorBit[3] = sanityCheckPoint(x+0,y+1, fw, r,g,b, iKir, p3, colorThreshold, rayThreshold2, dstOffset3);

            int i0 = counterImage[dstOffset0];
            int i1 = counterImage[dstOffset1];
            int i2 = counterImage[dstOffset2];
            int i3 = counterImage[dstOffset3];
            if (i0 < (nBins-1) && colorBit[0]) { zAccumulator[dstOffset0*nBins+i0] = -p3[2]; counterImage[dstOffset0] = i0+1;  }
            if (i1 < (nBins-1) && colorBit[1]) { zAccumulator[dstOffset1*nBins+i1] = -p3[2]; counterImage[dstOffset1] = i1+1;  }
            if (i2 < (nBins-1) && colorBit[2]) { zAccumulator[dstOffset2*nBins+i2] = -p3[2]; counterImage[dstOffset2] = i2+1;  }
            if (i3 < (nBins-1) && colorBit[3]) { zAccumulator[dstOffset3*nBins+i3] = -p3[2]; counterImage[dstOffset3] = i3+1;  }
        }
    }

}

// Note: depth values are positive in accumulator/maps!
void Configuration::filterDepth(cv::Mat *resultMap, float *Kir, float *Krgb, float *kcRGB, float robustDistance, int nMinimumSamples) {
    if (zAccumulator == NULL || counterImage == NULL) return;
    int fw = resultMap->cols;
    int fh = resultMap->rows;
    float *refData = (float*)resultMap->ptr();
    cv::Mat photoMask(fh,fw,CV_8UC1);
    unsigned char *mask = photoMask.ptr();

    memset(refData,0,sizeof(float)*fw*fh);
    memcpy(mask,rgbMask,sizeof(char)*fw*fh);

    float robustDistance2 = robustDistance*robustDistance;
    float minStdev = FLT_MAX;
    float maxStdev = 0.0f;
    int offset=0;
    for (int j = 0; j < fh; j++) {
        for (int i = 0; i < fw; i++,offset++) {
            stdevImage[offset] = 0.0f;
            int cnt = counterImage[offset];
            if (cnt <= nMinimumSamples) {
                // statistically inconsistent points are neglected
                mask[offset] = 0;
                continue;
            }
            float *zarr = &zAccumulator[offset*nBins];
            //float pivotZ = quickMedian(zarr,cnt);
            float pivotZ = zarr[0]; // take first sample as pivot
            float depthStdev = 0.0f;
            float z = robustAverage(zarr,cnt,pivotZ,robustDistance2,&depthStdev);
            refData[offset] = z;
            stdevImage[offset] = min(max(depthStdev*2,30.0f),robustDistance); // 3cm minimum deviation, capped to robustDistance
            if (stdevImage[offset] > maxStdev) maxStdev = stdevImage[offset];
            if (stdevImage[offset] < minStdev) minStdev = stdevImage[offset];
        }
    }

    printf("min stdev: %f, max stdev: %f\n",minStdev,maxStdev);

/*
    static int joo = 0; joo++;
    cv::Mat stdevMat(fh,fw,CV_8UC1);
    hdr2Gray(stdevImage, 0, robustDistance*2, stdevMat);
 //   colorSelection(rgbReference,rgbMask,neighborImage,0,255,0);
    char buf[512];
    sprintf(buf,"scratch/stdevImage%d.png",joo);
    imwrite(buf,stdevMat);*/

    // optimize gradient edges photometrically
    int NSAMPLES = 64;
    OMPFunctions *multicore = getMultiCoreDevice();
    multicore->optimizePhotometrically(&refData[0],mask,&rgbReference[0],fw,fh,stdevImage,NSAMPLES,Kir,Krgb,kcRGB,neighborPoses,neighborImages);
}
