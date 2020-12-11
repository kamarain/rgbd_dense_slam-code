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

class SparseBundleConfiguration {
public:
    SparseBundleConfiguration(int max2DVariance = 1);
	~SparseBundleConfiguration();
    void init(int ncams, int maxpoints, const char *camfn, const char *ptsfn, const char *calibfn);
    void setCamera(int camID, float *m4x4);
    void setCalib(float *calib3x3);
    // points are automatically switched into reference view 0 before further processing
    int addPoint(float x, float y, float z);
    int setProjection(int camID, int pointID, float px, float py);
    int getProjectionCount(int camID);
    void getProjections(int camID, float **pts2d, int **stored, int *count);
    void getProjection(int camID, int selectedID, float *px, float *py);
    bool isProjection(int camID, int selectedID);
    int  save();
    int  saveArguments();
    void release();
    int  deletePoint(int pointID);
    int  getCameraCount();
    void getPoints(float **pts3d, int *count);
    void getPointsSave(double **pts3d, int *count);
    void getPoint3D(int index, float *v, bool loadPoint=true);
    void setPoint3D(int index, float *v);
    int  addCameraPoint(int camID, float x, float y, float z);
    char *getCamsFileName();
    char *getCalibFileName();
    char *getPointsFileName();
    char *getCamsFileNameArg();
    char *getCalibFileNameArg();
    char *getPointsFileNameArg();
    double *getCameraParams();
    void updateCameras();
    float *getCamera(int i);
//    void setCamera(int camID, float *m16);
    void getFov(float *fovX, float *fovY);
    void smooth();
    int getPointCount();
private:
    void savePoints(const char *filename, bool saveCovariance = false);
    void saveCameras(const char *filename);
    void saveCalib(const char *filename);
    void canonizeTrajectory();
    float K[9];
    float max2DVariance;
    float  *cameraMatrices;
    double *cameraParamsBA;
    float *points3dLoad;
    double *points3dSave;
    float *points2d; int *npoints2d;
    int   *stored2d;
    int ncams, npoints, maxpoints;
    char sbaCamsFilename[512];
    char sbaPointsFilename[512];
    char sbaCalibFilename[512];
    // same configuration is stored as "arguments" for sba execution:
    char sbaCamsFilenameArg[512];
    char sbaPointsFilenameArg[512];
    char sbaCalibFilenameArg[512];
    float outMatrix[16];
    void loadPoints(const char *ptsfn);
};
