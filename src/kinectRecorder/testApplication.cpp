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

#include <GL/glew.h>
#include <cudakernels/cuda_funcs.h>
#include <string>
#include <SDL/SDL.h>
#include <opencv2/opencv.hpp>
#include <calib/calib.h>
//#include <types.h>
#include <reconstruct/basic_math.h>
#include <image2/Image2.h>
#include <image2/ImagePyramid2.h>
#include <tinyxml.h>
#include <assert.h>
#include <camera/OrbitingCamera.h>
//#include <rendering/GridBuffer.h>
#include <rendering/LineBuffer.h>
#include <rendering/BaseBuffer2.h>
#include <rendering/teapot.h>
#include <timer/performanceCounter.h>
#include <rendering/TrueTypeText.h>
#include <capture/kinect.h>
#include <capture/fileSource.h>
#include <rendering/GLWindow.h>
#include "testApplication.h"
#include <calib/ResultTxt.h>
#include <calib/GroundTruth.h>
#include "gputracker.h"

using namespace cv;
//PandaFramework framework;

extern float arPos[3];
extern float arRot[3];
extern float teapotScale;
float currentPose[16];
float currentPoseICP[16];
bool cameraTrackingOn = true;
bool shiftPressed = false;

//#define PERFORMANCE_TEST

static TrueTypeText font;
static float frameMillis = 0;
static float kinectExposure = 1.0f;
static int pixelSelectionAmount = 0;//320*240; must be divisible by 1024 for matching 1024 threads (max count)
static int trackingMode = INCREMENTAL;
static int refUpdateFreq = 5;
static int depthLayer = 0;

bool calibModified = false;
OrbitingCamera camera(-2500);
OrbitingCamera camera2(0);
bool toggleKinect = false;
bool overlayCross = false;
bool toggleAR = false;
bool toggleAR2 = false;
int maxResidualSize = 0;

VideoSource *videoStream = NULL;
VideoSource *videoStreamLib[10] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
VideoSource *kinect = NULL;

// configuration
const int nMultiResolutionLayers = 3;
int multiResolutionLayer = 0;

LineBuffer *groundTruthTrajectory = NULL;
float *cameraTrajectoryRef = NULL;
LineBuffer *kinfuTrajectory = NULL;
float *cameraTrajectoryKinfu = NULL;
LineBuffer *icpTrajectory = NULL;

unsigned char subIndex[6] = {0,1,2,3,4,5};

static int videoStreamIndex = 0;
char videoStreamBasePath[] =  "sequences/kinectScratch";
char keyframeBasePath[] =  "../sequences/kinectSequences";
char keyFrameSet[] = "keyframes.studio1";

char videoStreamPath[512];
static int keyFrameIndex = 0;
int hudTextCounter = 0;
char hudText1[512];
char hudText2[512];

Image2 reddot;

Mat flippedImage;
ResultTxt resultTxt;

void setHudText(const char *text1, const char *text2) {
	hudTextCounter = 30;
	strcpy(hudText1,text1);
	strcpy(hudText2,text2);
}

void updateReferenceTrajectory(const char *videoStreamPath) {
    if (groundTruthTrajectory != NULL) delete groundTruthTrajectory;
    if (kinfuTrajectory != NULL) delete kinfuTrajectory;

    icpTrajectory->reset();

    if (videoStreamPath == NULL) {
        groundTruthTrajectory = NULL;
        kinfuTrajectory = NULL;
    } else {
        groundTruthTrajectory = new LineBuffer(5000*2);
        int tmpRows;
        char buf[512];
        sprintf(buf,"%s/cameraMatrixEst-ref.txt",videoStreamPath);
        if (cameraTrajectoryRef != NULL) delete[] cameraTrajectoryRef;
        cameraTrajectoryRef  = loadCameraMatrices(buf,&tmpRows,0);
        if (cameraTrajectoryRef == NULL) printf("reference trajectory not found!\n");
        else {
            canonizeTrajectory(cameraTrajectoryRef,tmpRows);
            for (int i = 0; i < tmpRows-1; i++) {
                float x1 = cameraTrajectoryRef[(i)*16+3];
                float y1 = cameraTrajectoryRef[(i)*16+7];
                float z1 = cameraTrajectoryRef[(i)*16+11];
                float x2 = cameraTrajectoryRef[(i+1)*16+3];
                float y2 = cameraTrajectoryRef[(i+1)*16+7];
                float z2 = cameraTrajectoryRef[(i+1)*16+11];
                groundTruthTrajectory->addLine(x1,y1,z1,x2,y2,z2,255,255,255);

            }
            groundTruthTrajectory->upload();
        }

        kinfuTrajectory = new LineBuffer(5000*2);
        sprintf(buf,"%s/cameraMatrixEst-kinfu.txt",videoStreamPath);
//        sprintf(buf,"/home/tommi/Downloads/www.tml.tkk.fi/~hthartik/raide_12m.txt");
//        sprintf(buf,"/home/tommi/Downloads/www.tml.tkk.fi/~hthartik/tmp/freiburg640_3m.txt");
  //      sprintf(buf,"/home/tommi/Downloads/www.tml.tkk.fi/~hthartik/tmp/freiburg1_desk_8m.txt");

        if (cameraTrajectoryKinfu != NULL) delete[] cameraTrajectoryKinfu;
        int tmpRows2;
        cameraTrajectoryKinfu  = loadCameraMatrices(buf,&tmpRows2,0);
        if (cameraTrajectoryKinfu == NULL) printf("kinfu trajectory not found!\n");
        else {
            canonizeTrajectory(cameraTrajectoryKinfu,tmpRows2);
            float convMatrix[16]; identity4x4(&convMatrix[0]); convMatrix[5] = -1;  convMatrix[10] = -1;
            // convert matrix from kinfu
            for (int i = 0; i < tmpRows2; i++) {
                matrixMult4x4(convMatrix,&cameraTrajectoryKinfu[i*16],&cameraTrajectoryKinfu[i*16]);
                cameraTrajectoryKinfu[i*16+3] *= 1000.0f; cameraTrajectoryKinfu[i*16+7] *= 1000.0f; cameraTrajectoryKinfu[i*16+11] *= 1000.0f;
            }
            for (int i = 0; i < tmpRows2-1; i++) {
                float x1 = cameraTrajectoryKinfu[(i)*16+3];
                float y1 = cameraTrajectoryKinfu[(i)*16+7];
                float z1 = cameraTrajectoryKinfu[(i)*16+11];
                float x2 = cameraTrajectoryKinfu[(i+1)*16+3];
                float y2 = cameraTrajectoryKinfu[(i+1)*16+7];
                float z2 = cameraTrajectoryKinfu[(i+1)*16+11];
                kinfuTrajectory->addLine(x1,y1,z1,x2,y2,z2,255,0,255);
            }
            printf("uploading kinfu!\n");
            kinfuTrajectory->upload();
        }
    }
}

void resetDataSet() {
    camera.reset();
    if (videoStream != NULL && videoStream != kinect) videoStream->reset();
    gputracker::reset();
    icpTrajectory->reset();
    identity4x4(&currentPose[0]);
    identity4x4(&currentPoseICP[0]);
}

void TestApplication::setupVideoStream(int index) {
	char buf1[512],buf2[512]; 
	videoStreamIndex = index;
	sprintf(videoStreamPath,"%s/%d",videoStreamBasePath,index);

    sprintf(buf1,"%s/cameraMatrixEst.txt",videoStreamPath);
    resultTxt.save();
    resultTxt.init(buf1,true);

    sprintf(buf1,"%s/calib/calib.yml",videoStreamPath);
    gputracker::set_calibration(buf1);

    printf("setup video stream!\n");
	if (!toggleKinect) { 
        if (kinect != NULL) kinect->stop();
		videoStream = videoStreamLib[index]; 
		videoStream->reset(); 
		sprintf(buf1,"files");
        updateReferenceTrajectory(videoStreamPath);
	} else { 
        if (kinect != NULL) kinect->start();
		videoStream = kinect; 
		sprintf(buf1,"kinect");
        updateReferenceTrajectory(NULL);
	}
    //if (videoPreprocessor != NULL) videoPreprocessor->setVideoSource(videoStream);
    gputracker::set_source(videoStream);
	sprintf(buf2,"slot %d",index);
    setHudText(buf1,buf2);
    resetDataSet();
}

void drawTexture(Image2 &t, float x0, float y0, float size, float z = -1.0f) {
	if (t.data == NULL) return;
	t.bind();
	glBegin(GL_QUADS);		                // begin drawing a cube
	glNormal3f( 0.0f, 0.0f, 1.0f);                              // front face points out of the screen on z.
	glTexCoord2f(0.0f, 0.0f); glVertex3f(x0,y0, z);	// Bottom Left Of The Texture and Quad
	glTexCoord2f(1.0f, 0.0f); glVertex3f(x0+size, y0, z);	// Bottom Right Of The Texture and Quad
	glTexCoord2f(1.0f, 1.0f); glVertex3f(x0+size,  y0+size, z);	// Top Right Of The Texture and Quad
	glTexCoord2f(0.0f, 1.0f); glVertex3f(x0,y0+size, z);	// Top Left Of The Texture and Quad
	glEnd();                                    // done with the polygon.
}

void draw2dPoints(float *points2d, int pixelSelectionAmount, float r, float g, float b) {
    glColor4f(r,g,b,1);
    glBegin(GL_POINTS);
    int offset = 0;
    for (int i = 0; i < pixelSelectionAmount; i++) {
        glVertex3f(points2d[i*2+0],points2d[i*2+1],-0.5f);
    }
    glEnd();
    glPopAttrib();
}


void drawHUD() {
   if (hudTextCounter <= 0) return;
   glColor4f(1.0f,0.0f,0.0f,1.0f); 
   printTTF(font,0.7f,0.9f,-0.1f,0.004f,hudText1);
   printTTF(font,0.7f,0.8f,-0.1f,0.004f,hudText2);
   hudTextCounter--;
}

void renderTrajectory(LineBuffer *trajectory) {
    if (trajectory == NULL) { return;}
    glPushMatrix();
    glLineWidth(2.0f);
    trajectory->render();
    glPopMatrix();
}


void drawCross(float size=1.0f) {
    glDisable(GL_TEXTURE_2D);
//    glEnable(GL_COLOR_MATERIAL);
    float x0 = 0.0f;
    float y0 = 0.0f;
    float hs = size/2.0f;
    glLineWidth(5.0f);
    glColor4f(0,0,0,1);
    glBegin(GL_LINES);
    glVertex3f(x0-hs, y0,    -0.5f);	// Bottom Left Of The Texture and Quad
    glVertex3f(x0+hs, y0,    -0.5f);	// Bottom Right Of The Texture and Quad
    glVertex3f(x0,    y0-hs, -0.5f);	// Top Right Of The Texture and Quad
    glVertex3f(x0,    y0+hs, -0.5f);	// Top Left Of The Texture and Quad
    glEnd();
    glEnable(GL_TEXTURE_2D);
}

void drawBase(float *m, float xR, float xG, float xB, float yR, float yG, float yB, float zR, float zG, float zB, float len=350.0f) {

    //dumpMatrix("icp pose",m,4,4);
    glBegin(GL_LINES);
    glColor3f(xR,xG,xB);
    glVertex3f(m[12],m[13],m[14]); glVertex3f(m[12]+len*m[0],m[13]+len*m[1],m[14]+len*m[2]);
    glColor3f(yR,yG,yB);
    glVertex3f(m[12],m[13],m[14]); glVertex3f(m[12]+len*m[4],m[13]+len*m[5],m[14]+len*m[6]);
    glColor3f(zR,zG,zB);
    glVertex3f(m[12],m[13],m[14]); glVertex3f(m[12]+len*m[8],m[13]+len*m[9],m[14]+len*m[10]);
    glEnd();
}


void arTeapot() {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    float *proj = camera2.getProjectionMatrix();
    glLoadMatrixf(proj);

 //   gluPerspective(62.58660f,(GLfloat)320.0f/(GLfloat)240.0f,0.1f,100000.0f);
    //glLoadMatrixf(camera.getProjectionMatrix());
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glPushMatrix();

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE); glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);
    gputracker::render_rgbd();
    glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);

    float imtx[16];
    float mtx[16];
    transpose4x4(&currentPose[0],&mtx[0]);
    invertRT4(&mtx[0],&imtx[0]);
    transpose4x4(&imtx[0],&mtx[0]);
    glMultMatrixf(&mtx[0]);
    glTranslatef(arPos[0],arPos[1],arPos[2]);
    glRotatef(arRot[0],1.0,0,0);//
    glRotatef(arRot[1],0.0,1,0);//
    glRotatef(arRot[2],0.0,0,1);//
    drawTeapot(teapotScale);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void drawGrid(float *gridObject, int dimX, int dimY, int nLayers, Calibration &calib, float *relT, float *projectedGridObject) {

    float *K = &calib.getCalibData()[KR_OFFSET];

    glPushAttrib(GL_LIST_BIT | GL_CURRENT_BIT | GL_ENABLE_BIT | GL_TRANSFORM_BIT | GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_2D);
    glPointSize(3.0f);

    int gridSize = dimX*dimY*nLayers;
    for ( int i = 0; i < gridSize; i++) {
        float p[3],pr[3],r[3];
        p[0] = gridObject[i*3+0];
        p[1] = gridObject[i*3+1];
        p[2] = gridObject[i*3+2];
        transformRT3(relT, p, pr);
        matrixMultVec3(K,pr,r); r[0] /= r[2]; r[1] /= r[2]; r[0] = -1.0f+2*(r[0]/320.0f); r[1] = 1.0-2*(r[1]/240.0f);
        projectedGridObject[i*2+0] = r[0];
        projectedGridObject[i*2+1] = r[1];
    }
    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    glColor4f(0,0,0.0,1);
    glBegin(GL_QUADS);
    int offset = 0;
    for (int k = 0; k < nLayers; k++) {
        for (int j = 0; j < dimY-1; j++) {
            for (int i = 0; i < dimX-1; i++,offset++) {
                glVertex3f(projectedGridObject[(i+j*dimX+k*dimX*dimY)*2+0],projectedGridObject[(i+j*dimX+k*dimX*dimY)*2+1],-0.5f);
                glVertex3f(projectedGridObject[((i+1)+j*dimX+k*dimX*dimY)*2+0],projectedGridObject[((i+1)+j*dimX+k*dimX*dimY)*2+1],-0.5f);
                glVertex3f(projectedGridObject[((i+1)+(j+1)*dimX+k*dimX*dimY)*2+0],projectedGridObject[((i+1)+(j+1)*dimX+k*dimX*dimY)*2+1],-0.5f);
                glVertex3f(projectedGridObject[(i+(j+1)*dimX+k*dimX*dimY)*2+0],projectedGridObject[(i+(j+1)*dimX+k*dimX*dimY)*2+1],-0.5f);
            }
        }
    }
    glEnd();
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    glPopAttrib();
}

void renderFunc1() {
    glColor4f(1.0f,1.0f,1.0f,1.0f);

    static float zeta = -1.0f;

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDepthFunc(GL_LEQUAL);
    glDepthRange(0.0f, 100000.0f);
    gputracker::render_rgb_tex(zeta,false);


    char buf[512];
    if (videoStream->isRecording()) {
        glPushMatrix();
        glColor4f(1.0f,1.0f,1.0f,1.0f);//opacity);
        drawTexture(reddot,0.4,-1.1,0.5f,-0.6f);
        glPopMatrix();

        sprintf(buf,"tape remaining %3.1fs",videoStream->getSecondsRemaining());
        glColor4f(1,0,0,1);
        printTTF(font,-0.9f,0.8f,-0.1f,0.004f,buf);
    }
    glColor4f(1.0f,1.0f,1.0f,1.0f);
    if (videoStream != kinect)
        //sprintf(buf,"rgb image %04d, delay: %3.1fms",videoStream->getFrame(),frameMillis);
        sprintf(buf,"rgb image %04d",gputracker::get_frame_index());
    else
        sprintf(buf,"rgb image, delay: %3.1fms",frameMillis);

    printTTF(font,-0.9f,0.9f,-0.1f,0.004f,buf);
    if (overlayCross) {
        drawCross();
    }
    drawHUD();


    if (toggleAR) {
        arTeapot();
    }
}

void renderFunc2() {
    glPushAttrib(GL_CURRENT_BIT);
    glLineWidth(2.0f);
    glDisable(GL_TEXTURE_2D);
    glBegin(GL_LINES);
    glColor3f(1,0,0);
    glVertex3f(0,0,0); glVertex3f(1000,0,0);
    glColor3f(0,1,0);
    glVertex3f(0,0,0); glVertex3f(0,1000,0);
    glColor3f(0,0,1);
    glVertex3f(0,0,0); glVertex3f(0,0,1000);
    glEnd();
    glColor3f(1,0,0);
    glPointSize(2.0f);

  /*  float mean[3],normal[3];
    gputracker::get_first_plane(&mean[0],&normal[0]);
    glLineWidth(2.0f);
    glPushMatrix();
    glBegin(GL_LINES);
    glVertex3f(mean[0],mean[1],mean[2]);
    glVertex3f(mean[0]+normal[0]*1000.0f,mean[1]+normal[1]*1000.0f,mean[2]+normal[2]*1000.0f);
    glEnd();
    glPopMatrix();
*/

    gputracker::render_vertices(true);

    /////vbuffer->renderAll();
/*
    if (toggleAR2) {
       glPushMatrix();
       glTranslatef(arPos[0],arPos[1],arPos[2]);
       glRotatef(arRot[0],1.0,0,0);//arRot[1],arRot[2]);
       glRotatef(arRot[1],0,1,0);//arRot[1],arRot[2]);
       glRotatef(arRot[2],0,0,1);//arRot[1],arRot[2]);
       drawTeapot(teapotScale);
       glPopMatrix();
    }*/
    char buf[512];
    sprintf(buf,"current 3d points");
    printTTF(font,-0.7f,0.75f,-2.0f,0.004f,buf);

    glEnable(GL_TEXTURE_2D);
    glPopAttrib();
}


void renderFunc3() {
    glPushAttrib(GL_CURRENT_BIT);
    glLineWidth(2.0f);
    glDisable(GL_TEXTURE_2D);

    if (trackingMode == KEYFRAME || trackingMode == HYBRID) {
        gputracker::render_keyframes();
        gputracker::render_keys();
    }

    gputracker::render_base();

    drawBase(&currentPoseICP[0],0,1,0,0,1,0,0,1,0);

//    gputracker::render_vertices(true);

    //keyFrameRing.renderBase();
    if (cameraTrajectoryRef != NULL) {
        float T[16];
        transpose4x4(&cameraTrajectoryRef[videoStream->getFrame()*16],&T[0]);
        drawBase(&T[0],1,1,1,1,1,1,1,1,1);
    }
    renderTrajectory(groundTruthTrajectory);
    renderTrajectory(icpTrajectory);


    /*if (cameraTrajectoryKinfu != NULL) {
        float T[16];
        int lastKinfuFrame  = kinfuTrajectory->getMaxPointCount()/2-5;
        int cFrame      = videoStream->getFrame();
        if (cFrame >= lastKinfuFrame) cFrame = lastKinfuFrame;
        transpose4x4(&cameraTrajectoryKinfu[cFrame*16],&T[0]);
        drawBase(&T[0],1,0,1,1,0,1,1,0,1);
        kinfuTrajectory->nPoints = cFrame*2;
        renderTrajectory(kinfuTrajectory);
    }*/


//    kinfuTrajectory->nPoints = videoStream->getFrame()*2;
//    renderTrajectory(kinfuTrajectory);

    /*
    if (trackingMode == INCREMENTAL) {
        drawBase(keyFrameRing.getCurrentPose(),1,0,0,0,1,0,0,0,1);
    }*/


    /*
    if (trackingMode == KEYFRAME && similarKeyFrame != NULL) {
    ////    drawBase(&similarKeyFrame->T[0],1,0,0,0,1,0,0,0,1);
        for (int i = 0; i < poseCount; i++) {
            float T[16];
            transpose4x4(&cpuPose[i*16],&T[0]);
            drawBase(T,0,cpuWeights[i],0,0,cpuWeights[i],0,0,cpuWeights[i],0);

        }
    }*/


    char buf[512];
    sprintf(buf,"photometric");
    glColor3f(1,0,0);
    printTTF(font,-1.0f,-0.55f,-2.0f,0.004f,buf);
    sprintf(buf,"photometric+icp");
    glColor3f(0,1,0);
    printTTF(font,-1.0f,-0.65f,-2.0f,0.004f,buf);
/*    sprintf(buf,"kinfu");
    glColor3f(1,0,1);
    printTTF(font,-1.0f,-0.75f,-2.0f,0.004f,buf);
    sprintf(buf,"ref");
    glColor3f(1,1,1);
    printTTF(font,-1.0f,-0.85f,-2.0f,0.004f,buf);
*/

/*
    sprintf(buf,"mode: %d, nkeys: %d, mem: %d",trackingMode,gputracker::get_keyframe_count(),gputracker::get_free_gpu_memory());

    glColor3f(1,1,1);
    printTTF(font,-0.7f,0.75f,-2.0f,0.004f,buf);
*/

    glEnable(GL_TEXTURE_2D);
    glPopAttrib();
}

void renderFunc4() {

    glColor4f(1.0f,1.0f,1.0f,1.0f);
    /*if (keyFrameMode && similarKeyFrame != NULL) {
        drawTexture(similarKeyFrame->grayPyramid.getImageRef(multiResolutionLayer),-1,-1,2.0f);      
        //float relativeT[16]; identity4x4(&relativeT[0]); drawGrid(gridObject, nGridX, nGridY, nGridZ, calib, &relativeT[0],projectedGridObject);
    } else {
        drawTexture(keyFrameRing.getGray(0).getImageRef(multiResolutionLayer),-1,-1,2.0f);
    }*/

    glColor4f(1.0f,1.0f,1.0f,1.0f);
    static float zeta = -1.0f;
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LEQUAL);
    glDepthRange(0.0f, 100000.0f);
    if (gputracker::get_mode() == INCREMENTAL) {
        gputracker::render_icp_ref_tex(zeta);
    } else {
        gputracker::render_rgb_tex(zeta,true);
    }
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);


//    gputracker::render_ref_points();

    char buf[512];
    glColor4f(1.0f,1.0f,1.0f,1.0f);
    sprintf(buf,"photometric reference");
    printTTF(font,-0.9f,0.9f,-0.1f,0.004f,buf);
    sprintf(buf,"edge pixels: %d",pixelSelectionAmount);
    printTTF(font,-0.9f,0.80f,-2.0f,0.004f,buf);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
}

void renderFunc5()
{
    glPushAttrib(GL_CURRENT_BIT);
    glLineWidth(2.0f);
    glDisable(GL_TEXTURE_2D);
    glBegin(GL_LINES);
    glColor3f(1,0,0);
    glVertex3f(0,0,0); glVertex3f(1000,0,0);
    glColor3f(0,1,0);
    glVertex3f(0,0,0); glVertex3f(0,1000,0);
    glColor3f(0,0,1);
    glVertex3f(0,0,0); glVertex3f(0,0,1000);
    glEnd();
    glColor3f(1,0,0);
    glPointSize(2.0f);

    float lightPos[4],clightPos[4];
    clightPos[0] = 0; clightPos[1] = 100.0f; clightPos[2] = 0; clightPos[3] = 1.0f;
    gputracker::render_trimesh(&clightPos[0]);

//    gputracker::render_active_key();


    char buf[512];
    sprintf(buf,"ICP reference");
    glColor4f(1,1,1,1);
    printTTF(font,-0.7f,0.85f,-2.0f,0.004f,buf);
    glEnable(GL_TEXTURE_2D);
    glPopAttrib();
}

void renderFunc6() {  
    glPushAttrib(GL_CURRENT_BIT);

    glColor4f(1.0f,1.0f,1.0f,1.0f);
    static float zeta = -1.0f;

    glDisable(GL_DEPTH_TEST);
    gputracker::render_depth(zeta,depthLayer);

    /*
    glLineWidth(2.0f);
    glDisable(GL_TEXTURE_2D);

    if (cameraTrajectoryKinfu != NULL) {
        float T[16];
        int lastKinfuFrame  = kinfuTrajectory->getMaxPointCount()/2-5;
        int cFrame      = videoStream->getFrame();
        if (cFrame >= lastKinfuFrame) cFrame = lastKinfuFrame;
        transpose4x4(&cameraTrajectoryKinfu[cFrame*16],&T[0]);
        drawBase(&T[0],1,0,0,0,1,0,0,0,1);
        kinfuTrajectory->nPoints = cFrame*2;
        renderTrajectory(kinfuTrajectory);
    }
    renderTrajectory(groundTruthTrajectory);
*/
    char buf[512];
    sprintf(buf,"depth residual (layer: %d)",depthLayer);
    glColor3f(1,1,1);
    printTTF(font,-0.7f,0.75f,-2.0f,0.004f,buf);

    glEnable(GL_TEXTURE_2D);
    glPopAttrib();
}


int TestApplication::init(int argc, char **argv, const std::string &name, int resx, int resy, int nCol, int nRow)
{
    this->resx = resx; this->resy = resy;
    this->nCol = nCol; this->nRow = nRow;

    screenShot = new unsigned char[resx*resy*nCol*nRow*3];
    flippedImage = Mat::zeros(resy*nRow,resx*nCol,CV_8UC3);

    printf("initializing SDL...\n");
    if ( SDL_Init(SDL_INIT_VIDEO) < 0 ) {
        fprintf(stderr, "Unable to initialize SDL: %s\n", SDL_GetError());
        return 0;
    }

    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    if ( (sdlScreen = SDL_SetVideoMode(resx*nCol, resy*nRow, 0, SDL_OPENGL)) == NULL ) {
        fprintf(stderr, "Unable to create OpenGL screen: %s\n", SDL_GetError());
        return 0;
    }
    SDL_WM_SetCaption(name.c_str(), NULL);

    gputracker::initialize("config/gputracker_config.xml");

    printf("initializing OpenGL...\n");
    initGL(resx*nCol, resy*nRow);

    printf("initializing fonts...\n");
    font.init("fonts/blackWolf.ttf",16);

    printf("initializing textures...\n");
    loadImage("textures/reddot.png",&reddot,ONLY_GPU_TEXTURE,false);

    camera.reset();
    camera.setPerspective(62.58660f,float(resx)/float(resy),0.1f,100000.0f); camera.setOrbitingOrigin(0,0,-1000);
    camera.setCameraMaxSpeed(20.0f); camera.setFriction(1.0f); camera.setRotFriction(3e-4f*10.0f);
    camera.update();


    camera2.reset();
    camera2.setPerspective(62.58660f,float(resx)/float(resy),0.1f,100000.0f); camera.setOrbitingOrigin(0,0,-1000);
    camera2.setCameraMaxSpeed(20.0f); camera.setFriction(1.0f); camera.setRotFriction(3e-4f*10.0f);
    //camera2.setAngleX(3.141592653f/16);
    //camera2.setAngleY(3.141592653f/16);
    camera2.update();

    icpTrajectory = new LineBuffer(5000*2);

    GLWindow *window = NULL;

    window = new GLWindow(0*resx,1*resy,resx,resy,renderFunc1); glWindows.push_back(window);
    window = new GLWindow(1*resx,1*resy,resx,resy,renderFunc2); glWindows.push_back(window); window->setCamera(&camera2);
    window = new GLWindow(2*resx,1*resy,resx,resy,renderFunc3); glWindows.push_back(window); window->setCamera(&camera);
    window = new GLWindow(0*resx,0*resy,resx,resy,renderFunc4); glWindows.push_back(window);
    window = new GLWindow(1*resx,0*resy,resx,resy,renderFunc5); glWindows.push_back(window); window->setCamera(&camera2);//window->setCamera(&camera2);
    window = new GLWindow(2*resx,0*resy,resx,resy,renderFunc6); glWindows.push_back(window); //window->setCamera(&camera);

    sprintf(videoStreamPath,"%s/%d",videoStreamBasePath,2);
    kinect = new Kinect(videoStreamPath);
    for (int i = 0; i < 10; i++) {
        char buf[512]; sprintf(buf,"%s/%d",videoStreamBasePath,i);
        videoStreamLib[i] = new FileSource(buf,false);
    }

    setupVideoStream(videoStreamIndex);

    // initialize keyframe model
    char buf[512];
    sprintf(buf,"%s/%s",keyframeBasePath,keyFrameSet);

    //gputracker::set_keyframe_model(buf);
    if (trackingMode == INCREMENTAL) refUpdateFreq = gputracker::get_update_freq();
    else if (trackingMode == KEYFRAME) gputracker::set_keyframe_mode();
    else gputracker::set_hybrid_mode();

//    gputracker::set_depth_filtering(true);

    // ask how many points are selected based on config file
    pixelSelectionAmount = gputracker::get_selected_points();
    maxResidualSize = gputracker::get_max_residual_length();

    identity4x4(currentPose); identity4x4(currentPoseICP);
    initTeapot("scratch/teapot.txt");

    return 1;
}

void bubbleSort(float *vals, int *indices, int array_size)
{
    for (int i = 0; i < array_size; i++) indices[i] = i;
    for (int i = (array_size - 1); i > 0; i--) {
        for (int j = 1; j <= i; j++) {
          if (vals[indices[j-1]] > vals[indices[j]]) {
              int temp = indices[j-1];
              indices[j-1] = indices[j];
              indices[j] = temp;
      }
    }
  }
/*    for (int i = 0; i < array_size; i++) {
        printf("%d, %f %d\n",i,vals[indices[i]],indices[i]);
    }
    printf("median: %f, medianIndex: %d\n",vals[indices[array_size/2]],indices[array_size/2]);*/
}

float median(float* vals, int *index, int size) {
    int *indices = new int[size];
    bubbleSort(vals,indices,size);
    int medianIndex = size/2;
    *index = indices[medianIndex];
    delete[] indices;
    return vals[*index];
}

void writeTrajectoryDifferenceTxt(float *ref, float *cur,int numFrames, const char *outFile, bool flipMatrix=false) {
    float *distArray = new float[numFrames-1];
    float *angleArray = new float[numFrames-1];
    FILE *f = fopen(outFile,"wb");
    int ii = 0;
    fprintf(f,"frame millimeters degrees\n");
    for (int i = 0; i < numFrames-1; i++) {
        if (i > 650 && i < 1350) continue;
        float relativeT[16];
        float invCur[16];
        float mRef[16],mCur[16];
        //transpose4x4(&ref[16*i],&mRef[0]);
        //transpose4x4(&cur[16*i],&mCur[0]);
        memcpy(&mRef[0],&ref[16*i],sizeof(float)*16);
        memcpy(&mCur[0],&cur[16*i],sizeof(float)*16);

        float convMatrix[16]; identity4x4(&convMatrix[0]); convMatrix[5] = -1;  convMatrix[10] = -1;
        if (flipMatrix) {
            mRef[1] *= -1.0f; mRef[2] *= -1.0f;
            mRef[5] *= -1.0f; mRef[6] *= -1.0f;
            mRef[9] *= -1.0f; mRef[10] *= -1.0f;
        }

        invertRT4(&mCur[0],&invCur[0]);
        matrixMult4x4(&invCur[0], &mRef[0], &relativeT[0]);
        float dist,angle;
        poseDistance(&relativeT[0], &dist,&angle);
        fprintf(f,"%d %e %e\n",ii, dist,angle);
        distArray[ii] = dist; angleArray[ii] = angle;
        ii++;
    }
    fclose(f);

    int medianIndex = 0;
    float medianDist = median(distArray,&medianIndex,ii);
    float medianAngle = angleArray[medianIndex];
    printf("median trans error: %e\n",medianDist);
    printf("median angle error: %e\n",medianAngle);
    delete[] distArray;
    delete[] angleArray;
}

float average(float *arr, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += arr[i];
    return float(sum/n);
}

void writeTrajectoryDeltaDifferenceTxt(float *ref, float *cur,int numFrames, const char *outFile, bool flipMatrix=false) {
    float *distArray = new float[numFrames-1];
    float *angleArray = new float[numFrames-1];
    int ii = 0;
    float deltaM[16]; identity4x4(&deltaM[0]);
    FILE *f = fopen(outFile,"wb");
    int frameDelta = 300;
    fprintf(f,"frame trans_error angle_error\n");
    for (int i = 0; i < numFrames-frameDelta; i+=frameDelta) {
        if (i+frameDelta > 650 && i < 1350) continue;
        float relativeT[16];
        float invCur[16];
        float mRef1[16],mInvRef1[16],mInvRef2[16],mRef2[16],mRefDelta[16],mError1[16];
        float mCur1[16],mInvCur1[16],mInvCur2[16],mCur2[16],mCurDelta[16],mError2[16];

        float convMatrix[16]; identity4x4(&convMatrix[0]); convMatrix[5] = -1;  convMatrix[10] = -1;
        memcpy(&mRef1[0],&ref[16*i],sizeof(float)*16);
        if (flipMatrix) {
            mRef1[1] *= -1.0f; mRef1[2] *= -1.0f;
            mRef1[5] *= -1.0f; mRef1[6] *= -1.0f;
            mRef1[9] *= -1.0f; mRef1[10] *= -1.0f;
        }
        memcpy(&mCur1[0],&cur[16*i],sizeof(float)*16);

        invertRT4(&mRef1[0],&mInvRef1[0]);
        invertRT4(&mCur1[0],&mInvCur1[0]);

        memcpy(&mRef2[0],&ref[16*(i+frameDelta)],sizeof(float)*16);
        if (flipMatrix) {
            mRef2[1] *= -1.0f; mRef2[2] *= -1.0f;
            mRef2[5] *= -1.0f; mRef2[6] *= -1.0f;
            mRef2[9] *= -1.0f; mRef2[10] *= -1.0f;
        }
        memcpy(&mCur2[0],&cur[16*(i+frameDelta)],sizeof(float)*16);
        invertRT4(&mRef2[0],&mInvRef2[0]);
        invertRT4(&mCur2[0],&mInvCur2[0]);

        matrixMult4x4(&mInvRef1[0], &mRef2[0], &mRefDelta[0]);
        matrixMult4x4(&mInvCur2[0], &mCur1[0], &mCurDelta[0]);

        matrixMult4x4(&mCurDelta[0], &mRefDelta[0], &relativeT[0]);
        float dist,angle;
        poseDistance(&relativeT[0], &dist,&angle); //dist /= frameDelta; angle /= frameDelta;
        distArray[ii] = dist; angleArray[ii] = angle;
        fprintf(f,"%d %e %e\n",ii, dist,angle);
        ii++;
    }
    fclose(f);
    float resultScale = 30.0f/frameDelta;
    float stdevDist = 0;
    float stdevAngle = 0;
    int medianIndex = 0;
    float medianDist = median(distArray,&medianIndex,ii);
    float medianAngle = angleArray[medianIndex];
    printf("median trans drift: %e\n",medianDist*resultScale);
    printf("median angle drift: %e\n",medianAngle*resultScale);
    //printf("average trans drift: %e\n",avgDist);
   // printf("average angle drift: %e\n",avgAngle);

    for (int i = 0; i < ii; i++) {
        float distError = distArray[i]-medianDist;
        stdevDist += distError*distError;
        float angleError = angleArray[i]-medianAngle;
        stdevAngle += angleError*angleError;
    }
    stdevDist = sqrtf(stdevDist/ii);
    stdevAngle = sqrtf(stdevAngle/ii);
    printf("drift dist stdev: %e\n",stdevDist*resultScale);
    printf("drift angle stdev: %e\n",stdevAngle*resultScale);

    delete[] distArray;
    delete[] angleArray;
}

void writeTrajectoryRefDifferenceTxt(float *ref, int numFrames, const char *outFile) {
    float *distArray = new float[numFrames-1];
    float *angleArray = new float[numFrames-1];

    float deltaM[16]; identity4x4(&deltaM[0]);
    FILE *f = fopen(outFile,"wb");
    fprintf(f,"frame trans_error angle_error\n");
    for (int i = 0; i < numFrames-1; i++) {
        //printf("delta diff %d\n",i);
        float relativeT[16];
        float mRef1[16],mInvRef1[16],mInvRef2[16],mRef2[16],mRefDelta[16];
        memcpy(&mRef1[0],&ref[16*i],sizeof(float)*16);
        invertRT4(&mRef1[0],&mInvRef1[0]);
        memcpy(&mRef2[0],&ref[16*(i+1)],sizeof(float)*16);
        invertRT4(&mRef2[0],&mInvRef2[0]);

        matrixMult4x4(&mInvRef1[0], &mRef2[0], &mRefDelta[0]);

        float dist,angle;
        poseDistance(&mRefDelta[0], &dist,&angle); distArray[i] = dist; angleArray[i] = angle;
        fprintf(f,"%d %e %e\n",i, dist,angle);
    }
    fclose(f);
    printf("ref median trans delta: %e\n",quickMedian(distArray,numFrames-1));
    printf("ref median angle delta: %e\n",quickMedian(angleArray,numFrames-1));

    delete[] distArray;
    delete[] angleArray;
}


void TestApplication::shutDown()
{
    if (groundTruthTrajectory != NULL) delete groundTruthTrajectory;
    if (kinfuTrajectory != NULL) delete kinfuTrajectory;
    if (icpTrajectory != NULL) delete icpTrajectory;

    resultTxt.save();

    int tmpRows,nDiff=0;
    /*
        printf("win1:\n");
        float *cameraTrajectoryWin1 = loadCameraMatrices("../sequences/kinectScratch/0/cameraMatrixEst-win1.txt",&tmpRows); nDiff = tmpRows;
        writeTrajectoryDifferenceTxt(cameraTrajectoryWin1,cameraTrajectoryRef,nDiff,"../sequences/kinectScratch/0/trajectoryDiff-win1.txt");
        writeTrajectoryDeltaDifferenceTxt(cameraTrajectoryWin1,cameraTrajectoryRef,nDiff,"../sequences/kinectScratch/0/trajectoryDeltaDiff-win1.txt");
        delete[] cameraTrajectoryWin1;

        printf("win2:\n");
        float *cameraTrajectoryWin2 = loadCameraMatrices("../sequences/kinectScratch/0/cameraMatrixEst-win2.txt",&tmpRows);
        writeTrajectoryDifferenceTxt(cameraTrajectoryWin2,cameraTrajectoryRef,nDiff,"../sequences/kinectScratch/0/trajectoryDiff-win2.txt");
        writeTrajectoryDeltaDifferenceTxt(cameraTrajectoryWin2,cameraTrajectoryRef,nDiff,"../sequences/kinectScratch/0/trajectoryDeltaDiff-win2.txt");
        delete[] cameraTrajectoryWin2;
*/
    /*
        printf("cur:\n");
        float *cameraTrajectoryCur = loadCameraMatrices("../sequences/kinectScratch/1/cameraMatrixEst.txt",&tmpRows); nDiff = tmpRows;
        writeTrajectoryDifferenceTxt(cameraTrajectoryCur,cameraTrajectoryRef,nDiff,"../sequences/kinectScratch/1/trajectoryDiff.txt");
        writeTrajectoryDeltaDifferenceTxt(cameraTrajectoryCur,cameraTrajectoryRef,nDiff,"../sequences/kinectScratch/1/trajectoryDeltaDiff.txt");

        printf("kinfu:\n");
        writeTrajectoryDifferenceTxt(cameraTrajectoryKinfu,cameraTrajectoryRef,nDiff,"../sequences/kinectScratch/0/trajectoryDiff-kinfu.txt",true);
        writeTrajectoryDeltaDifferenceTxt(cameraTrajectoryKinfu,cameraTrajectoryRef,nDiff,"../sequences/kinectScratch/0/trajectoryDeltaDiff-kinfu.txt",true);

*/

/*
    float *dense = loadCameraMatrices("scratch/dense.txt",&tmpRows); nDiff = tmpRows;
    float *keyframe = loadCameraMatrices("scratch/keyframe.txt",&tmpRows); nDiff = tmpRows;

    float key0[] = { -408.870850, -12.341110, 0.315760};
    float key1[] = { 3679.837891, 111.070007, -2.841837 };
    float keym[] = { 1635.483521, 49.364449, -1.263039 };

    float u1 = key1[0] - key0[0];
    float u2 = key1[1] - key0[1];
    float u3 = key1[2] - key0[2];
    float len = sqrt(u1*u1+u2*u2+u3*u3);
    u1 /= len;
    u2 /= len;
    u3 /= len;

    FILE *f = fopen("scratch/rail-compare.txt","wb");
    fprintf(f,"frame millimeters millimeters\n");

    for (int i = 0; i < tmpRows; i++) {
        float v1,v2,v3;

        float x1 = dense[i*16+3]  - key0[0];
        float x2 = dense[i*16+7]  - key0[1];
        float x3 = dense[i*16+11] - key0[2];
        float dot;
        dot = x1*u1+x2*u2*x3*u3;
        x1 -= dot*u1;
        x2 -= dot*u2;
        x3 -= dot*u3;
        float e1 = sqrt(x1*x1+x2*x2+x3*x3);

        float y1 = keyframe[i*16+3]  - key0[0];
        float y2 = keyframe[i*16+7]  - key0[1];
        float y3 = keyframe[i*16+11] - key0[2];
        dot = y1*u1+y2*u2*y3*u3;
        y1 -= dot*u1;
        y2 -= dot*u2;
        y3 -= dot*u3;
        float e2 = sqrt(y1*y1+y2*y2+y3*y3);
        fprintf(f,"%d %e %e\n",i,e1,e2);
    }

    fclose(f);

    delete[] dense;
    delete[] keyframe;
*/

/*
    float *cameraTrajectory27 = loadCameraMatrices("../sequences/kinectSequences/keyframes.aure10/cameraMatrixEst-keys27.txt",&tmpRows); nDiff = tmpRows;
//    float *cameraTrajectory27 = loadCameraMatrices("../sequences/kinectSequences/keyframes.aure10/cameraMatrixEst-keys14ba.txt",&tmpRows); nDiff = tmpRows;

    float *cameraTrajectory14 = loadCameraMatrices("../sequences/kinectSequences/keyframes.aure10/cameraMatrixEst-keys14.txt",&tmpRows); nDiff = tmpRows;
    float *cameraTrajectory9 = loadCameraMatrices("../sequences/kinectSequences/keyframes.aure10/cameraMatrixEst-keys9.txt",&tmpRows); nDiff = tmpRows;
    float *cameraTrajectory7 = loadCameraMatrices("../sequences/kinectSequences/keyframes.aure10/cameraMatrixEst-keys7.txt",&tmpRows); nDiff = tmpRows;
    float *cameraTrajectory4 = loadCameraMatrices("../sequences/kinectSequences/keyframes.aure10/cameraMatrixEst-keys5.txt",&tmpRows); nDiff = tmpRows;

    writeTrajectoryDifferenceTxt(cameraTrajectory14,cameraTrajectory27,nDiff,"../sequences/kinectSequences/keyframes.aure10/diff-keys27-14.txt",false);
    writeTrajectoryDifferenceTxt(cameraTrajectory9,cameraTrajectory27,nDiff,"../sequences/kinectSequences/keyframes.aure10/diff-keys27-9.txt",false);
    writeTrajectoryDifferenceTxt(cameraTrajectory7,cameraTrajectory27,nDiff,"../sequences/kinectSequences/keyframes.aure10/diff-keys27-7.txt",false);
    writeTrajectoryDifferenceTxt(cameraTrajectory4,cameraTrajectory27,nDiff,"../sequences/kinectSequences/keyframes.aure10/diff-keys27-5.txt",false);

    delete[] cameraTrajectory27;
    delete[] cameraTrajectory14;
    delete[] cameraTrajectory9;
    delete[] cameraTrajectory7;
    delete[] cameraTrajectory4;
*/


/*
        delete[] cameraTrajectoryCur;
*/
    delete[] cameraTrajectoryRef;
    delete[] cameraTrajectoryKinfu;
    /*
        printf("win1:\n");
        float *cameraTrajectoryWin1 = loadCameraMatrices("../sequences/kinectScratch/1/cameraMatrixEst-win1.txt",&tmpRows); nDiff = tmpRows;
        writeTrajectoryDifferenceTxt(cameraTrajectoryWin1,cameraTrajectoryRef,nDiff,"../sequences/kinectScratch/1/trajectoryDiff-win1.txt");
        writeTrajectoryDeltaDifferenceTxt(cameraTrajectoryWin1,cameraTrajectoryRef,nDiff,"../sequences/kinectScratch/1/trajectoryDeltaDiff-win1.txt");
        delete[] cameraTrajectoryWin1;

        float *cameraTrajectoryCur = loadCameraMatrices("../sequences/kinectScratch/1/cameraMatrixEst.txt",&tmpRows);
        writeTrajectoryDifferenceTxt(cameraTrajectoryCur,cameraTrajectoryRef,nDiff,"../sequences/kinectScratch/1/trajectoryDiff.txt");
        writeTrajectoryDeltaDifferenceTxt(cameraTrajectoryCur,cameraTrajectoryRef,nDiff,"../sequences/kinectScratch/1/trajectoryDeltaDiff.txt");

        printf("win3:\n");
        float *cameraTrajectoryWin3 = loadCameraMatrices("../sequences/kinectScratch/0/cameraMatrixEst-win3.txt",&tmpRows); nDiff = tmpRows;
        writeTrajectoryDifferenceTxt(cameraTrajectoryWin3,cameraTrajectoryRef,nDiff,"../sequences/kinectScratch/0/trajectoryDiff-win3.txt");
        writeTrajectoryDeltaDifferenceTxt(cameraTrajectoryWin3,cameraTrajectoryRef,nDiff,"../sequences/kinectScratch/0/trajectoryDeltaDiff-win3.txt");
        delete[] cameraTrajectoryWin3;

        delete[] cameraTrajectoryCur;
        delete[] cameraTrajectoryRef;
*/
    /*frame3C.releaseData();
    frame1C.releaseData();
    depth1C.releaseData();
    gradX.releaseData();
    gradY.releaseData();*/
    reddot.releaseData();

    if (screenShot != NULL) delete[] screenShot; screenShot = NULL;


    for (size_t i = 0; i < glWindows.size(); i++) {
        GLWindow *window = glWindows[i]; delete window; glWindows[i] = NULL;
    }
    for (int i = 0; i < 10; i++) {
        delete videoStreamLib[i]; videoStreamLib[i] = NULL;
    }
    if (kinect != NULL) delete kinect; kinect = NULL;

    glWindows.clear();
    font.clean();

    ///if (imDepthDev != NULL) cudaFree(imDepthDev); imDepthDev = NULL;
    ///if (vbuffer != NULL) { vbuffer->release(); delete vbuffer; vbuffer = NULL; }
    gputracker::release();
    saveTeapot("scratch/teapot.txt");
    SDL_Quit();
}

void VSyncOn(char On) 
{ 
#ifdef WIN32
	typedef void (APIENTRY * WGLSWAPINTERVALEXT) (int); 
	WGLSWAPINTERVALEXT wglSwapIntervalEXT = (WGLSWAPINTERVALEXT) 
		wglGetProcAddress("wglSwapIntervalEXT"); 
	if (wglSwapIntervalEXT) 
	{ 
		wglSwapIntervalEXT(On); // set vertical synchronisation 
	} 
#endif

} 

void TestApplication::initGL( int width, int height ) 
{
	glViewport(0, 0, width, height);
	glEnable(GL_TEXTURE_2D);                    // Enable texture mapping.
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);	// This Will Clear The Background Color To Black
	glClearDepth(1.0);				// Enables Clearing Of The Depth Buffer
	glDepthFunc(GL_LESS);			// The Type Of Depth Test To Do
	glEnable(GL_DEPTH_TEST);			// Enables Depth Testing
	glShadeModel(GL_SMOOTH);			// Enables Smooth Color Shading

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();				// Reset The Projection Matrix

    gluPerspective(62.58660f,(GLfloat)width/(GLfloat)height,0.1f,100000.0f);	// Calculate The Aspect Ratio Of The Window
	glMatrixMode(GL_MODELVIEW);

     GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 1.0 };
     GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
     GLfloat light_specular[] = { 1.0, 1.0, 1.0, 1.0 };
     GLfloat light_position[] = { 0.0, 0.0, -500.0, 1.0 };

     glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
     glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
     glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
     glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	glDisable(GL_LIGHTING);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//	glDisable(GL_DEPTH_TEST);
	glEnable(GL_ALPHA_TEST);
	glAlphaFunc( GL_GREATER, 0.0f);

	glDisable(GL_CULL_FACE);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);      // 4-byte pixel alignment
//	initGlew();
	VSyncOn(0);
}

void TestApplication::saveScreenShot() {
/*
    // original
    static int shotIndex = 0;
    char buf[512];
    sprintf(buf,"scratch/screenshot%04d.ppm",shotIndex);
    glReadPixels(0,0,resx*nCol,resy*nRow,GL_BGR,GL_UNSIGNED_BYTE,screenShot);
    Mat rgbHeader(resy*nRow, resx*nCol, CV_8UC3, screenShot);
    flip(rgbHeader,flippedImage,0);
    imwrite(buf,flippedImage);
    shotIndex++;
*/
    static int shotIndex = 0;

    char buf[512];
    sprintf(buf,"/home/tommi/Downloads/www.tml.tkk.fi/~hthartik/freiburg1_desk_8m/frame%04d.png.ppm",shotIndex);
    Mat kinfuImage = imread(buf);

    sprintf(buf,"scratch/screenshot%04d.ppm",shotIndex);
    glReadPixels(0,0,resx*nCol,resy*nRow,GL_BGR,GL_UNSIGNED_BYTE,screenShot);
    Mat rgbHeader(resy*nRow, resx*nCol, CV_8UC3, screenShot);
    flip(rgbHeader,flippedImage,0);

    Mat composite(resy*2, resx*2, CV_8UC3);
    unsigned char *ptr = composite.ptr();
    unsigned char *src = flippedImage.ptr();
    unsigned char *kinfuPtr = kinfuImage.ptr();
    for (int y = 0; y < resy; y++) {
        memcpy(&ptr[y*resx*2*3],&src[y*resx*nCol*3],resx*3);
        memcpy(&ptr[y*resx*2*3+resx*3],&src[y*resx*nCol*3+resx*3*2],resx*3);
        memcpy(&ptr[(y+resy)*resx*2*3+resx*3],&src[(y+resy)*resx*nCol*3+resx*3*2],resx*3);
        memcpy(&ptr[(y+resy)*resx*2*3],&kinfuPtr[y*resx*3],resx*3);
    }
    imwrite(buf,composite);
    shotIndex++;
}

void TestApplication::saveScreenShot(unsigned char *index, int nIndices) {
    static int shotIndex = 0;
    char buf[512];
    sprintf(buf,"scratch/screenshot%04d.ppm",shotIndex);
    glReadPixels(0,0,resx*nCol,resy*nRow,GL_BGR,GL_UNSIGNED_BYTE,screenShot);
    Mat rgbHeader(resy*nRow, resx*nCol, CV_8UC3, screenShot);
    flip(rgbHeader,flippedImage,0);

    unsigned char *src[6] = {NULL,NULL,NULL,NULL,NULL,NULL};
    src[0] = flippedImage.ptr();
    src[1] = flippedImage.ptr() + resx*3;
    src[2] = flippedImage.ptr() + 2*resx*3;
    src[3] = flippedImage.ptr() + resy*resx*3*nCol;
    src[4] = flippedImage.ptr() + resy*resx*3*nCol+resx*3;
    src[5] = flippedImage.ptr() + resy*resx*3*nCol+2*resx*3;
/*
    char buff[512]; unsigned int width,height,nChannels,pitch;
    sprintf(buff,"scratch/pandaframe%04d.png",shotIndex+3);
    unsigned char *data = (unsigned char*)loadPNG(&buff[0], &width,&height,&nChannels,&pitch,false);
//    printf("w: %d, h: %d channels:%d\n",width,height,nChannels); fflush(stdin); fflush(stdout);

    Mat pandaShotBig(480,640,CV_8UC3);
    unsigned char *dst = pandaShotBig.ptr();

    unsigned char *srcA = data;
    for (int j=0; j < 480; j++)
    for (int i = 0; i < 640; i++) {
        int srcOffset = i*4+j*pitch;
        int dstOffset = i*3+j*3*640;
        dst[dstOffset+0] = srcA[srcOffset+2];
        dst[dstOffset+1] = srcA[srcOffset+1];
        dst[dstOffset+2] = srcA[srcOffset+0];
    }
    unsigned char *pandaPtr = pandaShotBig.ptr();
*/
   /*
    char buff[512]; unsigned int width,height,nChannels,pitch;
    int modelFrame = shotIndex-52; if (modelFrame<1) modelFrame = 1;
    sprintf(buff,"scratch/modelshot%04d.ppm",modelFrame);
    Mat modelShot = imread(buff,-1);
    unsigned char *pandaPtr = modelShot.ptr();
*/

    if (nIndices == 6) {
        // save full image as is
        imwrite(buf,flippedImage);
    } else if (nIndices == 4) {
        Mat composite(resy*2, resx*2, CV_8UC3);
        unsigned char *ptr = composite.ptr();
        for (int y = 0; y < resy; y++) {
            memcpy(&ptr[y*composite.cols*3],&src[index[0]][y*resx*3*nCol],resx*3);
 //           memcpy(&ptr[y*composite.cols*3+resx*3],&pandaPtr[y*resx*3],resx*3);
            memcpy(&ptr[y*composite.cols*3+resx*3],&src[index[1]][y*resx*3*nCol],resx*3);
            memcpy(&ptr[(y+resy)*composite.cols*3],&src[index[2]][y*resx*3*nCol],resx*3);
            memcpy(&ptr[(y+resy)*composite.cols*3+resx*3],&src[index[3]][y*resx*3*nCol],resx*3);
        }
        imwrite(buf,composite);
    } else if (nIndices == 2) {
        Mat composite(resy, resx*2, CV_8UC3);
        unsigned char *ptr = composite.ptr();
        for (int y = 0; y < resy; y++) {
            memcpy(&ptr[y*composite.cols*3],&src[index[0]][y*resx*3*nCol],resx*3);
            memcpy(&ptr[y*composite.cols*3+resx*3],&src[index[1]][y*resx*3*nCol],resx*3);
        }
        imwrite(buf,composite);
    }
    //delete[] data;
    shotIndex++;
}


TestApplication::TestApplication()
{
	videoStream = NULL;
}

TestApplication::~TestApplication()
{

}

void TestApplication::renderScene()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    for (size_t i = 0; i < glWindows.size(); i++) glWindows[i]->render();

	glFlush(); // NOTE: important to tell CUDA rendering is done!
	SDL_GL_SwapBuffers();
}

void TestApplication::handleKeyDown(int key, int &done) {
	switch (key) 
	{	
	case SDLK_0:
		setupVideoStream(0);
		break;
	case SDLK_1:
		setupVideoStream(1);
		break;
	case SDLK_2:
		setupVideoStream(2);
		break;
	case SDLK_3:
		setupVideoStream(3);
		break;
	case SDLK_4:
		setupVideoStream(4);
		break;
	case SDLK_5:
		setupVideoStream(5);
		break;
	case SDLK_6:
		setupVideoStream(6);
		break;
	case SDLK_7:
		setupVideoStream(7);
		break;
	case SDLK_8:
		setupVideoStream(8);
		break;
	case SDLK_9:
		setupVideoStream(9);
		break;
    case SDLK_LSHIFT:
        shiftPressed = true;
        break;
	case SDLK_t:
		toggleKinect = !toggleKinect;
		//printf("toggle kinect: %d\n",int(toggleKinect));	
		setupVideoStream(videoStreamIndex);
		break;
	case SDLK_r:
		videoStream->setRecording(videoStreamPath,!videoStream->isRecording());
		break;
	case SDLK_k:
		sprintf(videoStreamPath,"%s/keyframes/keyframe-%04d",videoStreamBasePath,keyFrameIndex);
		videoStream->setRecording(videoStreamPath,true,true,30,true,false);
		keyFrameIndex++;
		break;
	case SDLK_l:
		kinect->record();
		break;

	case SDLK_p:
        gputracker::set_camera_tracking(!gputracker::playing());
        printf("P pressed!\n");
		break;
    case SDLK_x:
        overlayCross = !overlayCross;
        break;
    case SDLK_SPACE:
        saveScreenShot();
        printf("saving screenshot at frame %d\n!\n",videoStream->getFrame());
        break;
    }
}

void TestApplication::handleKeyUp(int key, int &done) {
    switch(key)
	{
	case SDLK_ESCAPE:
		done = 1;
		break;
	case SDLK_u:
		multiResolutionLayer = (multiResolutionLayer+1)%nMultiResolutionLayers;
		break;
        case SDLK_i:
		multiResolutionLayer = (multiResolutionLayer+nMultiResolutionLayers-1)%nMultiResolutionLayers;
		break;
        case SDLK_e:
                if (kinect != NULL) {
                    kinectExposure -= 0.5f;
                    if (kinectExposure < 0.0f) kinectExposure = 1.0f;
                    kinect->setExposure(kinectExposure);
                }
                break;
    case SDLK_LSHIFT:
        shiftPressed = false;
        break;
        case SDLK_s:
                // toggle SLAM on/off
                trackingMode = (trackingMode+1)%3;
                if (trackingMode == INCREMENTAL) gputracker::set_incremental_mode();
                else if (trackingMode == KEYFRAME) { gputracker::set_hybrid_mode(); trackingMode = HYBRID;} //gputracker::set_keyframe_mode();}
                else gputracker::set_hybrid_mode();
                break;
        case SDLK_a:
                gputracker::set_camera_tracking(!gputracker::playing());
                printf("a pressed!\n");
                break;
        case SDLK_q:
                toggleAR = !toggleAR;
                printf("AR toggled: %d\n",int(toggleAR));
                break;
        case SDLK_w:
                toggleAR2 = !toggleAR2;
                break;
        case SDLK_d:
                depthLayer = (depthLayer+1)%3;
                break;
	case SDLK_c:
        resetDataSet();
		break;
    case SDLK_z:
            camera.saveCameraMatrix("cam.txt");
            break;
    case SDLK_x:
            camera.loadCameraMatrix("cam.txt");
            break;
	}
}

void TestApplication::handlePressedKeys() {
	/* Check current key state for special commands */
	Uint8 *keys = SDL_GetKeyState(NULL);

	if ( keys[SDLK_KP7] == SDL_PRESSED ) 
	{
                camera.updateYawSpeed(0.01);
	}
	if ( keys[SDLK_KP9] == SDL_PRESSED ) 
	{
                camera.updateYawSpeed(-0.01);
	}
	if ( keys[SDLK_KP8] == SDL_PRESSED ) 
	{
		camera.updateSpeed(1000.0f);
	}
	if ( keys[SDLK_KP5] == SDL_PRESSED ) 
	{
		camera.updateSpeed(-1000.0f);
	}
	if ( keys[SDLK_KP1] == SDL_PRESSED ) 
	{
		camera.updatePitchSpeed(-0.01);
	}
	if ( keys[SDLK_KP3] == SDL_PRESSED ) 
	{
		camera.updatePitchSpeed(0.01);
	}

    if ( keys[SDLK_m] == SDL_PRESSED) teapotScale += 10.0f;
    if ( keys[SDLK_n] == SDL_PRESSED) {
           teapotScale -= 10.0f;
            if (teapotScale < 100.0f) teapotScale = 100.0f;
    }

    if (keys[SDLK_UP] == SDL_PRESSED) { if (!shiftPressed) arPos[2] -= 10.0f; else arRot[2] -= 1.0f; }
    if (keys[SDLK_DOWN] == SDL_PRESSED) { if (!shiftPressed) arPos[2] += 10.0f; else arRot[2] += 1.0f; }
    if (keys[SDLK_LEFT] == SDL_PRESSED) { if (!shiftPressed) arPos[0] -= 10.0f; else arRot[0] -= 1.0f; }
    if (keys[SDLK_RIGHT] == SDL_PRESSED) { if (!shiftPressed) arPos[0] += 10.0f; else arRot[0] += 1.0f; }
    if (keys[SDLK_PAGEUP] == SDL_PRESSED) { if (!shiftPressed) arPos[1] += 10.0f; else arRot[1] -= 1.0f; }
    if (keys[SDLK_PAGEDOWN] == SDL_PRESSED) { if (!shiftPressed) arPos[1] -= 10.0f; else arRot[1] += 1.0f; }

    if ( keys[SDLK_KP_PLUS] == SDL_PRESSED && trackingMode == INCREMENTAL) {
        pixelSelectionAmount = MIN(pixelSelectionAmount + 1024,maxResidualSize);
        gputracker::set_selected_points(pixelSelectionAmount);
    }
    if ( keys[SDLK_KP_MINUS] == SDL_PRESSED && trackingMode == INCREMENTAL) {
        pixelSelectionAmount = MAX(pixelSelectionAmount - 1024,1024);
        gputracker::set_selected_points(pixelSelectionAmount);
    }
}

int TestApplication::run(int fps)
{
    int frame = 0;
    double totalTime = 0;
    int done=0;
    PerformanceCounter timer;
    PerformanceCounter timer2;

    while ( !done ) {
        timer.StartCounter();
        // process input
        SDL_Event event;
        while ( SDL_PollEvent(&event)) {
            if ( event.type == SDL_QUIT ) done = 1;
            if ( event.type == SDL_KEYDOWN)
                handleKeyDown(event.key.keysym.sym,done);
            if (event.type == SDL_KEYUP)
                handleKeyUp(event.key.keysym.sym,done);
        }
        handlePressedKeys();

/*
        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,0);
        float delay = 0;
        float delays[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        static float delaysCumulated[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        static int numExec = 0;*/

        //timer2.StartCounter();
        int ret = gputracker::track_frame();
        //cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); printf("track time: %3.1f\n",delay);


        //timer2.StopCounter(); printf("track time: %3.1f\n",timer2.GetElapsedTime());

        if (ret && gputracker::playing()) {
            float p[3]; p[0] = currentPoseICP[12]; p[1] = currentPoseICP[13]; p[2] = currentPoseICP[14];
            gputracker::get_pose(&currentPose[0],&currentPoseICP[0]); resultTxt.addPose(&currentPoseICP[0]);
            icpTrajectory->addLine(p[0],p[1],p[2],currentPoseICP[12],currentPoseICP[13],currentPoseICP[14], 0,255,0);
            icpTrajectory->upload();
        }

        //cudaEventRecord(start,0);
        renderScene();
        //cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); printf("render time: %3.1f\n",delay);

        //cudaEventDestroy(start);
        //cudaEventDestroy(stop);

//        timer2.StopCounter(); printf("render time: %3.1f\n",timer2.GetElapsedTime());

        camera.addFriction();
        camera.update();

        timer.StopCounter();
        double elapsedTime = timer.GetElapsedTime();
        totalTime += elapsedTime;
        frame++;
        if (frame%fps==0) {
            frameMillis = (totalTime*1000.0f)/fps;
            totalTime = 0.0;
        }
        if (gputracker::playing()) {
            subIndex[0] = 0;
            subIndex[1] = 2;
            subIndex[2] = 3;
            subIndex[3] = 4;
           // saveScreenShot(subIndex,4);
        }

        int msRemaining = MAX(33 - int(elapsedTime*1000.0f),0);
        SDL_Delay(msRemaining);
    }
    shutDown();
    return 1;
}

