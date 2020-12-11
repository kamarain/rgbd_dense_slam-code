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

#include "sparseBundleConfiguration.h"
#include <reconstruct/basic_math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

namespace localFuncs2 {
char *skipLine(char *ptr)
{
    int i = 0;
    while (ptr[i] != '\n') {i++;}
    while (ptr[i] == '\n' || ptr[i] == '\r') {i++;}
    return &ptr[i];
}

char *findWhiteSpace(char *ptr)
{
    int i = 0;
    while (ptr[i] != ' ') i++;
    return &ptr[i];
}

unsigned int getRowCount(char *ptr,int numChars)
{
    int nRows = 0;
    for (int i = 0; i < numChars; i++)
        if (ptr[i] == '\n') nRows++;
    return nRows;
}
}

using namespace localFuncs2;

int SparseBundleConfiguration::getCameraCount() {
    return ncams;
}

void SparseBundleConfiguration::getProjection(int camID, int selectedID, float *px, float *py) {
    *px = points2d[camID*maxpoints*2+selectedID*2+0];
    *py = points2d[camID*maxpoints*2+selectedID*2+1];
}

bool SparseBundleConfiguration::isProjection(int camID, int selectedID) {
    return stored2d[camID*maxpoints+selectedID]>0;

}


int SparseBundleConfiguration::deletePoint(int pointID) {
    for (int i = 0; i < ncams; i++) {
        bool removedFlag = stored2d[i*maxpoints+pointID]>0;
        for (int j = pointID; j < maxpoints-1; j++) {
            points2d[i*maxpoints*2+j*2+0] = points2d[i*maxpoints*2+(j+1)*2+0];
            points2d[i*maxpoints*2+j*2+1] = points2d[i*maxpoints*2+(j+1)*2+1];
            stored2d[i*maxpoints+j] = stored2d[i*maxpoints+j+1];
        }
        if (removedFlag && npoints2d[i]>0) npoints2d[i]--;
        stored2d[i*maxpoints+maxpoints-1] = 0;
    }
    for (int j = pointID; j < maxpoints-1; j++) {
        points3dLoad[j*3+0] = points3dLoad[(j+1)*3+0];
        points3dLoad[j*3+1] = points3dLoad[(j+1)*3+1];
        points3dLoad[j*3+2] = points3dLoad[(j+1)*3+2];

        points3dSave[j*3+0] = points3dSave[(j+1)*3+0];
        points3dSave[j*3+1] = points3dSave[(j+1)*3+1];
        points3dSave[j*3+2] = points3dSave[(j+1)*3+2];
    }
    if (npoints > 0) npoints--;
}

SparseBundleConfiguration::SparseBundleConfiguration(int max2DVar) {
    cameraMatrices = NULL;
    cameraParamsBA = NULL;
    points3dLoad = NULL;
    points3dSave = NULL;
    points2d = NULL;
    npoints2d = NULL;
    stored2d  = NULL;
    ncams = 0;
    maxpoints = 0;
    npoints = 0;
    max2DVariance = max2DVar;
}

void SparseBundleConfiguration::release()
{
    if (cameraMatrices != NULL) delete[] cameraMatrices; cameraMatrices = NULL;
    if (cameraParamsBA != NULL) delete[] cameraParamsBA; cameraParamsBA = NULL;
    if (points3dLoad != NULL)       delete[] points3dLoad; points3dLoad = NULL;
    if (points3dSave != NULL)       delete[] points3dSave; points3dSave = NULL;
    if (points2d != NULL)       delete[] points2d; points2d = NULL;
    if (npoints2d != NULL)      delete[] npoints2d; npoints2d = NULL;
    if (stored2d != NULL)       delete[] stored2d; stored2d = NULL;

}
SparseBundleConfiguration::~SparseBundleConfiguration() {
    release();
}

void SparseBundleConfiguration::loadPoints(const char *ptsfn) {
    FILE *f = fopen(ptsfn,"rb");
    if (f == NULL) { printf("%s not found!\n",ptsfn); return; }
    fseek(f,0,SEEK_END); int fileSize = ftell(f); fseek(f,0,SEEK_SET);
    char *buf = new char[fileSize];
    int ret = fread(buf,1,fileSize,f);
    fclose(f);

    int numRows = getRowCount(buf,fileSize);
//    printf("row count: %d\n",numRows);

    char *ptr = buf;
    for (int i = 0; i < numRows; i++) {
        float x = atof(ptr);
        ptr = findWhiteSpace(ptr)+1;
        float y = atof(ptr);
        ptr = findWhiteSpace(ptr)+1;
        float z = atof(ptr);
        ptr = findWhiteSpace(ptr)+1;
        int pointID = addPoint(x,y,z);
        int nviews = atoi(ptr);
//        printf("p3d: %f %f %f %d\n",x,y,z,nviews);
        for (int j = 0; j < nviews; j++) {
            ptr = findWhiteSpace(ptr)+1;
            int viewIndex = atoi(ptr);
            ptr = findWhiteSpace(ptr)+1;
            float px = atof(ptr);
            ptr = findWhiteSpace(ptr)+1;
            float py = atof(ptr);
  //          printf("%d %f %f\n",viewIndex,px,py);
            setProjection(viewIndex, pointID, px,py);
        }
        ptr = skipLine(ptr);
    }

    delete[] buf;
}

void SparseBundleConfiguration::setCalib(float *calib3x3) {
    memcpy(&K[0],calib3x3,sizeof(float)*9);
}

float *SparseBundleConfiguration::getCamera(int camID) {
    transpose4x4(&cameraMatrices[camID*16],&outMatrix[0]);
    return &outMatrix[0];
}

void SparseBundleConfiguration::getFov(float *fovAngleX, float *fovAngleY) {
    // note: principal point is assumed to be in the middle of the screen!
    *fovAngleX = 180.0f*2.0f*atan(1.0f/fabs(K[0]/160.0f))/3.141592653f;
    *fovAngleY = 180.0f*2.0f*atan(1.0f/fabs(K[4]/120.0f))/3.141592653f;
}

void SparseBundleConfiguration::init(int ncams, int maxpoints,const char *camfn, const char *ptsfn, const char *calibfn) {
    release();
    this->ncams     = ncams;
    this->maxpoints = maxpoints;
    this->npoints   = 0;
    cameraMatrices  = new float[16*ncams];
    cameraParamsBA  = new double[7*ncams];
    points3dLoad    = new float[3*maxpoints];
    points3dSave    = new double[3*maxpoints];
    points2d        = new float[2*maxpoints*ncams];
    npoints2d       = new int[ncams];
    stored2d        = new int[maxpoints*ncams];
    for (int i = 0; i < ncams; i++) {
        npoints2d[i] = 0;
        identity4x4(&cameraMatrices[i*16]);
        for (int j = 0; j < 7; j++) cameraParamsBA[i*7+j] = 0;
        for (int j = 0; j < maxpoints; j++) {
            points2d[i*maxpoints*2+2*j+0] = 0;
            points2d[i*maxpoints*2+2*j+1] = 0;
            stored2d[i*maxpoints+j]     = 0;
        }

    }
    for (int i = 0; i < maxpoints; i++) {
        points3dLoad[i*3 + 0] = 0;
        points3dLoad[i*3 + 1] = 0;
        points3dLoad[i*3 + 2] = 0;
        points3dSave[i*3 + 0] = 0;
        points3dSave[i*3 + 1] = 0;
        points3dSave[i*3 + 2] = 0;
    }
    strcpy(sbaCamsFilename,camfn); sprintf(sbaCamsFilenameArg,"%s.arg",sbaCamsFilename);
    strcpy(sbaPointsFilename,ptsfn); sprintf(sbaPointsFilenameArg,"%s.arg",sbaPointsFilename);
    strcpy(sbaCalibFilename,calibfn); sprintf(sbaCalibFilenameArg,"%s.arg",sbaCalibFilename);
    printf("sba config: %s %s %s\n",sbaCamsFilename,sbaPointsFilename,sbaCalibFilename);

    if (ptsfn != NULL) {
        loadPoints(ptsfn);
    }
}

void SparseBundleConfiguration::setCamera(int camID, float *m4x4) {
    transpose4x4(m4x4,&cameraMatrices[camID*16]);
/*
    float *m = &cameraMatrices[camID*16];
    float mleft[16];
    memcpy(&mleft[0],m,sizeof(float)*16); mleft[0] = -mleft[0]; mleft[4] = -mleft[4]; mleft[8] = -mleft[8];

    float qf[4];
    rot2Quaternion(mleft,4,&qf[0]);

    double *q = &cameraParamsBA[camID*7];
    for (int j = 0; j < 4; j++) q[j] = (float)qf[j];
    q[4] = (double)m[3];
    q[5] = (double)m[7];
    q[6] = (double)m[11];*/
}

int SparseBundleConfiguration::addCameraPoint(int camID, float x, float y, float z) {
    if (npoints >= maxpoints) return -1;
    float *camRef = &cameraMatrices[0];
    float *camCur = &cameraMatrices[camID*16];
    float invRef[16]; invertRT4(camRef,&invRef[0]);
    float T[16]; matrixMult4x4(&invRef[0],camCur,&T[0]);
    float p[4],r[4];

    dumpMatrix("ref",camRef,4,4);
    dumpMatrix("cur",camCur,4,4);


    p[0] = x; p[1] = y; p[2] = z; p[3] = 1;
    transformRT3(T,p,r);

    return addPoint(r[0],r[1],r[2]);
}

int SparseBundleConfiguration::addPoint(float x, float y, float z) {
    if (npoints >= maxpoints) return -1;

    points3dLoad[npoints*3+0] = x;
    points3dLoad[npoints*3+1] = y;
    points3dLoad[npoints*3+2] = z;

    points3dSave[npoints*3+0] = x;
    points3dSave[npoints*3+1] = y;
    points3dSave[npoints*3+2] = z;

    npoints++;
    return npoints-1;
}

int SparseBundleConfiguration::getProjectionCount(int camID) {
    return npoints2d[camID];
}


void SparseBundleConfiguration::getProjections(int camID, float **pts2d, int **storedFlags, int *count) {
    *pts2d  = &points2d[camID*maxpoints*2];
    *storedFlags = &stored2d[camID*maxpoints];
    *count = getProjectionCount(camID);
}

void SparseBundleConfiguration::getPoints(float **pts3d, int *count) {
    *pts3d  = &points3dLoad[0];
    *count = npoints;
}

void SparseBundleConfiguration::getPointsSave(double **pts3d, int *count) {
    *pts3d  = &points3dSave[0];
    *count = npoints;
}

void SparseBundleConfiguration::getPoint3D(int index, float *v, bool loadPoint) {
    if (index >= npoints) return;
    if (loadPoint) {
        v[0] = points3dLoad[index*3+0];
        v[1] = points3dLoad[index*3+1];
        v[2] = points3dLoad[index*3+2];
        return;
    } else {
        v[0] = (float)points3dSave[index*3+0];
        v[1] = (float)points3dSave[index*3+1];
        v[2] = (float)points3dSave[index*3+2];
        return;
    }
}

void SparseBundleConfiguration::setPoint3D(int index, float *v) {
    if (index >= npoints) return;
    points3dLoad[index*3+0] = v[0];
    points3dLoad[index*3+1] = v[1];
    points3dLoad[index*3+2] = v[2];
    points3dSave[index*3+0] = v[0];
    points3dSave[index*3+1] = v[1];
    points3dSave[index*3+2] = v[2];
}


int SparseBundleConfiguration::getPointCount() { return npoints; }

int SparseBundleConfiguration::setProjection(int camID, int pointID, float px, float py)
{
    if (camID < 0 || camID >= ncams) return 0;
    if (pointID < 0 || pointID >= npoints) return 0;

    int pi = npoints2d[camID];
    if (pi >= maxpoints) return 0;

    points2d[camID*maxpoints*2+2*pointID+0] = px;
    points2d[camID*maxpoints*2+2*pointID+1] = py;
    if (stored2d[camID*maxpoints+pointID] == false) npoints2d[camID]++;
    stored2d[camID*maxpoints+pointID]     = 1;

    return 1;
}

void SparseBundleConfiguration::savePoints(const char *filename, bool saveCovariance) {
    FILE *f = fopen(filename,"wb");
    for (int i = 0; i < npoints; i++) {
        int nprojections = 0;
        for (int j = 0; j < ncams; j++) if (stored2d[j*maxpoints+i]) nprojections++;
        // do not save bundle adjusted 3d points to disk (points3dSave could be used for this)
        if (nprojections > 0) {
            fprintf(f,"%f %f %f %d ",points3dLoad[i*3+0],points3dLoad[i*3+1],points3dLoad[i*3+2],nprojections);
            for (int j = 0; j < ncams; j++) {
                if (stored2d[j*maxpoints+i]) {
                    if (!saveCovariance) {
                        fprintf(f,"%d %f %f ",j,points2d[j*maxpoints*2+2*i+0],points2d[j*maxpoints*2+2*i+1]);
                    } else {
                        //File format is X_{0}...X_{pnp-1}  nframes  frame0 x0 y0 [covx0^2 covx0y0 covx0y0 covy0^2] frame1 x1 y1 [covx1^2 covx1y1 covx1y1 covy1^2] ...
                        float varDelta = max2DVariance / float(ncams-1);
                        float variance = 1+float(j)*varDelta; float zeroVal = 0.0f;
                        fprintf(f,"%d %f %f %f %f %f %f ",j,points2d[j*maxpoints*2+2*i+0],points2d[j*maxpoints*2+2*i+1],variance,zeroVal,zeroVal,variance);
                    }
                }
            }
            fprintf(f,"\n");
        }
    }
    fclose(f);
}

/* bundle example (BaseGeometry.cpp), flip 3x3 rotation z-axis and translation z
void CameraInfo::Reflect()
{
    m_R[2] = -m_R[2];
    m_R[5] = -m_R[5];
    m_R[6] = -m_R[6];
    m_R[7] = -m_R[7];

    m_t[2] = -m_t[2];

    Finalize();
}
*/

// convert rotation matrices to correspond left handed system! tommi tykkala 8.9.2012 (check also intrisic matrix and point params!)
void reflectCamera(float *m, float *mR) {
    // reflect 4x4 camera matrix respect to z-axis (bundler example!)
    memcpy(mR,m,sizeof(float)*16);
    mR[2] = -mR[2];
    mR[6] = -mR[6];
    mR[8] = -mR[8];
    mR[9] = -mR[9];
    mR[11] = -mR[11];
}

void SparseBundleConfiguration::updateCameras() {
    for (int i = 0; i < ncams; i++) {
        float q[4],t[3];
        float *m = &cameraMatrices[i*16];
        double md[16];
        quaternion2Rot(&cameraParamsBA[i*7],md);
        md[3]  = cameraParamsBA[i*7+4];
        md[7]  = cameraParamsBA[i*7+5];
        md[11] = cameraParamsBA[i*7+6];

        float tmp[16],tmpR[16];
        for (int j = 0; j < 16; j++) tmp[j] = (float)md[j];
        invertRT4(tmp,tmpR);
        reflectCamera(tmpR,m);
//        m[0] = -m[0]; m[4] = -m[4]; m[8] = -m[8];
    }
    canonizeTrajectory();
}

void SparseBundleConfiguration::saveCameras(const char *filename) {
    FILE *f = fopen(filename,"wb");
    for (int i = 0; i < ncams; i++) {
        float q[4],t[3];
        float tmpR[16];
        memcpy(tmpR,&cameraMatrices[i*16],sizeof(float)*16);
        reflectCamera(tmpR,tmpR);
        float tmp[16];
        invertRT4(tmpR,tmp);
        rot2Quaternion(tmp,4,&q[0]); t[0] = tmp[3]; t[1] = tmp[7]; t[2] = tmp[11];
        fprintf(f,"%f %f %f %f %f %f %f\n",q[0],q[1],q[2],q[3],t[0],t[1],t[2]);
    }
    fclose(f);
}

double *SparseBundleConfiguration::getCameraParams() {
    return &cameraParamsBA[0];
}

char *SparseBundleConfiguration::getCamsFileName() {
    return &sbaCamsFilename[0];
}

char *SparseBundleConfiguration::getCalibFileName() {
    return &sbaCalibFilename[0];
}

char *SparseBundleConfiguration::getPointsFileName() {
    return &sbaPointsFilename[0];
}

char *SparseBundleConfiguration::getCamsFileNameArg() {
    return &sbaCamsFilenameArg[0];
}

char *SparseBundleConfiguration::getCalibFileNameArg() {
    return &sbaCalibFilenameArg[0];
}

char *SparseBundleConfiguration::getPointsFileNameArg() {
    return &sbaPointsFilenameArg[0];
}

void SparseBundleConfiguration::saveCalib(const char *filename) {
    FILE *f = fopen(filename,"wb");
    fprintf(f,"%f %f %f \n",K[0],K[1],K[2]);
    fprintf(f,"%f %f %f \n",K[3],K[4],K[5]);
    fprintf(f,"%f %f %f \n",K[6],K[7],K[8]);
    fclose(f);
}

int SparseBundleConfiguration::save() {
    printf("saving bundle configuration!\n");
    savePoints(sbaPointsFilename);
    saveCameras(sbaCamsFilename);
    saveCalib(sbaCalibFilename);
}

int SparseBundleConfiguration::saveArguments() {
    printf("saving bundle arguments!\n");
    savePoints(sbaPointsFilenameArg,true);
    saveCameras(sbaCamsFilenameArg);
    saveCalib(sbaCalibFilenameArg);
}


void SparseBundleConfiguration::canonizeTrajectory() {
    float mi[16];
    invertRT4(&cameraMatrices[0],mi);
    for (int i = 0; i < ncams; i++) {
        matrixMult4x4(mi,&cameraMatrices[i*16],&cameraMatrices[i*16]);
    }
}

void SparseBundleConfiguration::smooth() {
    printf("smoothing!\n");
    fflush(stdout);
}
