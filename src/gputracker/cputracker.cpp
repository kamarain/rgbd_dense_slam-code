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
#include <GL/glew.h>
#include "cputracker.h"
#include <opencv2/opencv.hpp>
#include <calib/calib.h>
#include <capture/VideoPreProcessorCPU.h>
#include <timer/performanceCounter.h>
#include <calib/ResultTxt.h>
#include <reconstruct/basic_math.h>
#include <imagematch/ICP.h>
#include <tinyxml.h>

using namespace cv;

static Calibration calibKinect;
static VideoPreProcessorCPU *videoPreprocessor = NULL;
static VertexBuffer2 *vbuffer = NULL;
static float *imDepthDev = NULL;
static int pointSelectionAmount = 8192;//320*240; must be divisible by 1024 for matching 1024 threads (max count)
static int keyFramePointSelectionAmount = 7*1024;//320*240; must be divisible by 1024 for matching 1024 threads (max count)
static bool firstRefFrameUpdated = false;
static bool estimationOn = true;
static int photometricUpdateFreq = 1;
static int nPhotometricReferences = 3;
static int icpUpdateFreq = 1;
static float icpPoseMatrix[16];
static bool cameraTrackingEnabled = true;
static float planeMean[3] = {0,0,0};
static float planeNormal[3] = {0,0,1};
static int currentFrame = 0;
static int slamDistTol = 200;
static int slamAngleTol = 20;
static int slamMaxKeys = 50;
static int nIterations[3] = {10,3,3};
static ICP icpMatcher;
cv::Mat icpSelectionImage(240,320,CV_8UC3);
bool verboseOpt = true;//true;//false;

void cputracker::set_estimation(bool mode) {
    estimationOn = mode;
}

void cputracker::set_selected_points(int nPoints) {
//    if (trackingMode != INCREMENTAL) return;
    pointSelectionAmount = nPoints;
    if (videoPreprocessor != NULL ) {
        videoPreprocessor->setPixelSelectionAmount(pointSelectionAmount);
    }
}

void cputracker::set_camera_tracking(bool flag) {
    if (videoPreprocessor == NULL) return;
    bool isPlaying = !videoPreprocessor->isPaused();
    if (isPlaying != flag) videoPreprocessor->pause();
    cameraTrackingEnabled = flag;
}

int cputracker::get_selected_points() {
    return pointSelectionAmount;
}

void cputracker::get_pose(float *poseMatrixDst) {
    // convert from IR coordinate system to RGB camera coordinate system
    float *TRL = &calibKinect.getCalibData()[TRL_OFFSET];
    float *TLR = &calibKinect.getCalibData()[TLR_OFFSET];
    float T[16];
    // inv(TLR*inv(icpPoseMat)) = icpPoseMat * TRL
    matrixMult4x4(&icpPoseMatrix[0],TRL,&T[0]);
    matrixMult4x4(TLR,&T[0],&T[0]);
    transpose4x4(&T[0],&poseMatrixDst[0]);
}

unsigned char *cputracker::get_rgb_ptr() {
    return videoPreprocessor->getRGBImage().ptr();

}

float *cputracker::get_depth_ptr() {
    return (float*)videoPreprocessor->depthMapR.ptr();

}

void cputracker::fill_rgbtex(unsigned int texID) {
    glEnable(GL_TEXTURE_2D);
    //glBindTexture(GL_TEXTURE_2D, texID);
    //glTexSubImage2D(GL_TEXTURE_2D,0,0,0,320,240,GL_RGB,GL_FLOAT,NULL);
    glBindTexture(GL_TEXTURE,0);
    //videoPreprocessor->getRGBImage().ptr());
}

void cputracker::fill_depthtex(unsigned int depthID) {
/*    glBindTexture(GL_TEXTURE_2D, depthID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, depth1C.pbo);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 320,240, 0, GL_LUMINANCE, GL_FLOAT, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);
    glBindTexture(GL_TEXTURE_2D,0);
    */
}



bool cputracker::playing() {
    return cameraTrackingEnabled;
}

void parseIntegers(const char *str, int nInt, int *data)
{
    int stringIndex = 0;
    int arrayIndex = 0;

    int nRest = nInt;
    while (nRest>0) {
        int val = atoi(&str[stringIndex]);
        data[arrayIndex] = val;
        if (nRest>1)
            while(str[stringIndex] != ' ') { stringIndex++;}
        stringIndex++;
        arrayIndex++;
        nRest--;
    }
}

int load_parameters(const char *xmlConfigFile) {
    TiXmlDocument *doc = new TiXmlDocument;
    bool loadOkay = doc->LoadFile(xmlConfigFile,TIXML_DEFAULT_ENCODING);

    if (!loadOkay) {
        printf("problem loading %s!\n", xmlConfigFile);
        fflush(stdin); fflush(stdout);
        return 0;
    }

    TiXmlHandle hDoc(doc);
    TiXmlElement* pElem;
    TiXmlHandle hRoot(0);
    int val[1024];

    pElem = hDoc.FirstChildElement().Element();
    if(!pElem) { printf("no xml root found!\n"); return 0; }
    hRoot = TiXmlHandle(pElem);

    printf("configuration is : \n");

    pElem=hRoot.FirstChild("nKeyPoints1024").Element();
    if (pElem!=0){
        parseIntegers(pElem->GetText(),1,&val[0]);
        printf("    nKeyPoints  : %d\n",val[0]*1024);
        keyFramePointSelectionAmount = val[0]*1024;
    } else { printf("nKeyPoints1024 not found!\n"); return 0; }

    pElem=hRoot.FirstChild("incrementalParams").Element();
    if (pElem!=0){
        parseIntegers(pElem->GetText(),3,&val[0]);
        printf("    updateFreqPhotometric  : %d, updateFreqICP : %d, nPoints: %d\n",val[0],val[1],val[2]*1024);
        photometricUpdateFreq = val[0];
        icpUpdateFreq = val[1];
        pointSelectionAmount = val[2]*1024;
    } else { printf("incrementalParams not found!\n"); return 0; }


    pElem=hRoot.FirstChild("slamParams").Element();
    if (pElem!=0){
        parseIntegers(pElem->GetText(),3,&val[0]);
        printf("    slamParams  : %d %d %d %d\n",val[0],val[1],val[2],keyFramePointSelectionAmount);
        slamMaxKeys  = val[0];
        slamDistTol  = val[1];
        slamAngleTol = val[2];

    } else { printf("slamParams not found!\n"); return 0; }

    pElem=hRoot.FirstChild("optParams").Element();
    if (pElem!=0){
        parseIntegers(pElem->GetText(),3,&val[0]);
        printf("    optParams   : %d %d %d\n",val[0],val[1],val[2]);
        for (int i = 0; i < 3; i++) nIterations[i] = val[i];
    } else { printf("optParams not found!\n"); return 0; }

    delete doc;
}

int cputracker::initialize(const char *xmlConfigFile) {
    printf("cputracker init based on %s!\n",xmlConfigFile);
    load_parameters(xmlConfigFile);
}

void cputracker::reset() {
    currentFrame = 0;
    firstRefFrameUpdated = false;
    icpMatcher.reset();
    identity4x4(&icpPoseMatrix[0]);
}

void cputracker::get_first_plane(float *mean, float *normal) {
    memcpy(mean,&planeMean[0],sizeof(float)*3);
    memcpy(normal,&planeNormal[0],sizeof(float)*3);
}

void cputracker::set_calibration(const char *xmlCalibFileName) {
    calibKinect.init(xmlCalibFileName,false);
    calibKinect.setupCalibDataBuffer(320,240);
    icpMatcher.setCalib(&calibKinect.getCalibData()[KL_OFFSET],&calibKinect.getCalibData()[KR_OFFSET],&calibKinect.getCalibData()[TLR_OFFSET],&calibKinect.getCalibData()[KcR_OFFSET]);
}

void cputracker::set_source(VideoSource *stream) {
    if (videoPreprocessor == NULL) {
        videoPreprocessor = new VideoPreProcessorCPU(stream,3,&calibKinect);
    }
    videoPreprocessor->setVideoSource(stream);
    reset();
}

float cputracker::get_fov_x() {
    return calibKinect.getFovX();
}

float cputracker::get_fov_y() {
    return calibKinect.getFovY();
}

int cputracker::get_update_freq() {
    return photometricUpdateFreq;
}

void cputracker::release() {
    if (videoPreprocessor != NULL) {
        videoPreprocessor->release(); delete videoPreprocessor; videoPreprocessor = NULL;
        icpSelectionImage.release();
        icpMatcher.release();
        fflush(stdin);
        fflush(stdout);

    }
}


int cputracker::track_frame() {
    if (videoPreprocessor == NULL) return 0;

    int ret = videoPreprocessor->preprocess();
    videoPreprocessor->getPlane(&planeMean[0],&planeNormal[0]);
    // also update depth map on CPU:
    icpMatcher.setDepthMap(videoPreprocessor->getDepthImageL(),videoPreprocessor->getGrayImage(),pointSelectionAmount, nPhotometricReferences);

    if (!ret && videoPreprocessor->isPlaying()) {
        reset();
        return 0;
    }

    videoPreprocessor->getPlane(&planeMean[0],&planeNormal[0]);

    if (estimationOn) {
        bool updatePhotometricRef = !firstRefFrameUpdated || !videoPreprocessor->isPlaying();
        bool updateICPRef = updatePhotometricRef;
        if (currentFrame % photometricUpdateFreq == 0) updatePhotometricRef =  true;
        if (currentFrame % icpUpdateFreq == 0) updateICPRef =  true;

        if (videoPreprocessor->isPlaying() && firstRefFrameUpdated) {
            // also align depth maps on CPU:
            icpMatcher.optimize(&nIterations[0],verboseOpt);
            matrixMult4x4(&icpPoseMatrix[0],icpMatcher.getIncrement(),&icpPoseMatrix[0]);
            currentFrame++;
        }
        icpMatcher.updatePhotoReference();
        if (updateICPRef) {
            // also update reference depth map on CPU:
            icpMatcher.updateReference();
            firstRefFrameUpdated = true;
        }
    }
    return 1;
}

