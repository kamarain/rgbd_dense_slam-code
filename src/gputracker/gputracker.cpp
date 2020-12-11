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
#include "gputracker.h"
#include <GL/glew.h>
#include <cudakernels/cuda_funcs.h>
#include <helper_cuda_gl.h>
#include <opencv2/opencv.hpp>
#include <calib/calib.h>
#include <capture/VideoPreProcessorGPU.h>
#include <capture/VideoPreProcessorCPU.h>
#include <reconstruct/KeyFrameModel.h>
#include <warping/KeyFrameRing.h>
#include <timer/performanceCounter.h>
#include <calib/ResultTxt.h>
#include <reconstruct/basic_math.h>
#include <imagematch/ICP.h>
#include <tinyxml.h>
#include <rendering/TriangleBuffer2.h>
#include <rendering/shader.h>

#define PERFORMANCE_TEST

using namespace cv;

static Calibration calibKinect;
static VideoPreProcessorGPU *videoPreprocessorGPU = NULL;
static VideoPreProcessorCPU *videoPreprocessorCPU = NULL;
static ImagePyramid2 frame1C;
static Image2 frame3C;
static Image2 depth1C;
static Image2 depthCur1C[3];  // only for ICP visualization
static Image2 normalMap3C[3]; // only for ICP visualization
static VertexBuffer2 *vbuffer = NULL;
static TriangleBuffer2 *trimesh = NULL;
static Shader *phongShader = NULL;
static float *imDepthDev = NULL;
static KeyFrameModel *keyFrameModel = NULL;
static KeyFrameModel *incrementalModel = NULL;
static int pointSelectionAmount = 8192;//320*240; must be divisible by 1024 for matching 1024 threads (max count)
static int keyFramePointSelectionAmount = 7*1024;//320*240; must be divisible by 1024 for matching 1024 threads (max count)
static KeyFrameRing keyFrameRing;
static bool firstRefFrameUpdated = false;
static bool addKeyframe = true;
static KeyFrame *similarKeyFrame = NULL;
static KeyFrame *previousSimilarKeyFrame = NULL;
static int trackingMode = INCREMENTAL;
static std::vector<KeyFrame*> similarKeys;
//static PoseFilter poseFilter;
static int maxEstimationKeys = 1;
static float cpuPose[16*256];
static float cpuWeights[256];
static int poseCount = 0;
//static bool estimationOn = true;
static int photometricUpdateFreq = 1;
static int nPhotometricReferences = 3;
static int icpUpdateFreq = 1;
static PerformanceCounter timer2;
static float poseMatrix[16];
static float icpPoseMatrix[16];
static bool filterDepthFlag = false;
static bool blankImagesFlag = false;
static bool cameraTrackingEnabled = false;
static float *greenScreenDev = NULL;
static float planeMean[3] = {0,0,0};
static float planeNormal[3] = {0,0,1};
static int currentFrame = 0;
static int slamDistTol = 200;
static int slamAngleTol = 20;
static int slamMaxKeys = 50;
static int nIterations[3] = {10,3,3};
static ICP icpMatcher;
cv::Mat icpSelectionImage(240,320,CV_8UC3);
static Image2 icpSelection3C;
static BaseBuffer2 baseBuffer;
bool verboseOpt = false;
bool useParallelDICP = true;
static int icpReferenceResoX = 320;
static int icpReferenceResoY = 240;

void gputracker::render_active_key() {
    if (similarKeyFrame != NULL)
        similarKeyFrame->vbuffer.render();
}

void gputracker::get_first_plane(float *mean, float *normal) {
    memcpy(mean,&planeMean[0],sizeof(float)*3);
    memcpy(normal,&planeNormal[0],sizeof(float)*3);
}

int gputracker::get_mode() {
    return trackingMode;
}

int gputracker::get_keyframe_count() {
    if (trackingMode == KEYFRAME) {
        if (keyFrameModel != NULL) return keyFrameModel->getKeyFrameCount();
    } else if (trackingMode == HYBRID) {
        if (incrementalModel != NULL) return incrementalModel->getKeyFrameCount();
    } else return 0;
}

void gputracker::render_keyframes() {
    if (incrementalModel == NULL) return;
    int cnt = incrementalModel->getKeyFrameCount();
    for (int i = 0; i < cnt; i++) {
        KeyFrame *kf = incrementalModel->getKeyFrame(i);
        if (kf) {
            glPushMatrix();
            glMultMatrixf(&kf->T[0]);
            kf->vbuffer.render();
            glPopMatrix();
        }
    }
}

int gputracker::get_max_residual_length() {
    return maxResidualSize;
}

void gputracker::set_blank_images(bool flag) {
    blankImagesFlag = flag;
}

int gputracker::get_free_gpu_memory() {
    // note: this requires one texture allocation before gives values back!
    #define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
    #define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

    int cur_avail_mem_kb = 0;
    glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX,
              &cur_avail_mem_kb);
    return cur_avail_mem_kb/1024;
}

void gputracker::render_vertices(bool allvertices) {
    if (vbuffer == NULL) return;
    if (allvertices) vbuffer->renderAll();
    else vbuffer->render();
}


void gputracker::render_trimesh(float *clightPos) {
    if (trimesh == NULL || !useParallelDICP) return;
    trimesh->render(phongShader,clightPos);
    /*glPushMatrix();
    float mtx1[16],mtx2[16],mtxT[16];
    matrixMult4x4(icpMatcher.getBaseTransform(),&calibKinect.getCalibData()[TRL_OFFSET],&mtx1[0]);
    matrixMult4x4(&calibKinect.getCalibData()[TLR_OFFSET],&mtx1[0],&mtx2[0]);
    transpose4x4(&mtx2[0],&mtxT[0]);
    glMultMatrixf(&mtxT[0]);
    vbuffer->renderColor(1,0,0,1);
    glPopMatrix();*/
}


void gputracker::render_keys()
{
    if (trackingMode == KEYFRAME) {
        if (keyFrameModel == NULL) return;
        keyFrameModel->renderCameraFrames(&similarKeys);

    } else if (trackingMode == HYBRID) {
        if (incrementalModel == NULL) return;
        incrementalModel->renderCameraFrames(&similarKeys);
    }
}

void gputracker::render_base()
{
    baseBuffer.renderBase();
}

void gputracker::set_depth_filtering(bool mode) {
    filterDepthFlag = mode;
}
/*
void gputracker::set_estimation(bool mode) {
    estimationOn = mode;
}*/

int gputracker::get_num_est_poses() {
    return poseCount;
}

int gputracker::get_max_est_poses() {
    return maxEstimationKeys;
}

void drawTextureLIB(Image2 &t, float x0, float y0, float size, float z = -1.0f) {
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

void drawTextureDepthLIB(Image2 &t, float x0, float y0, float size, float z = -1.0f) {
    if (t.data == NULL || videoPreprocessorCPU == NULL) return;
    t.bind();
    glBegin(GL_QUADS);		                // begin drawing a cube
    glNormal3f( 0.0f, 0.0f, 1.0f);                              // front face points out of the screen on z.
    float stepU = 1.0f / (t.width-1);
    float stepV = 1.0f / (t.height-1);
    float *dptr = (float*)videoPreprocessorCPU->getDepthImageR().ptr();
    float *K = &calibKinect.getCalibData()[KR_OFFSET];
    float iK[9]; inverse3x3(K,&iK[0]);
    for (int j = 0; j < t.height-1; j++) {
        for (int i = 0; i < t.width-1; i++) {
            float us = i*stepU;
            float vs = j*stepV;
            int off = i+j*t.width;
            float p0[3],p1[3],p2[3],p3[3];
            get3DPoint(float(i),float(j),dptr[off],iK,&p0[0],&p0[1],&p0[2]);
            get3DPoint(float(i+1),float(j),dptr[off+1],iK,&p1[0],&p1[1],&p1[2]);
            get3DPoint(float(i+1),float(j+1),dptr[off+1+t.width],iK,&p2[0],&p2[1],&p2[2]);
            get3DPoint(float(i),float(j+1),dptr[off+t.width],iK,&p3[0],&p3[1],&p3[2]);
            glTexCoord2f(us, vs); glVertex3fv(p0);	// Bottom Left Of The Texture and Quad
            glTexCoord2f(us+stepU, vs); glVertex3fv(p1);	// Bottom Right Of The Texture and Quad
            glTexCoord2f(us+stepU, vs+stepV); glVertex3fv(p2);	// Top Right Of The Texture and Quad
            glTexCoord2f(us, vs+stepV); glVertex3fv(p3);	// Top Left Of The Texture and Quad
        }
    }
    glEnd();                                    // done with the polygon.
}

void gputracker::render_icp_ref_tex(float z) {
    if (videoPreprocessorCPU == NULL) return;
    glColor4f(1.0f,1.0f,1.0f,1.0f);
    drawTextureLIB(icpSelection3C,-1,-1,2.0f,z);
}

void gputracker::render_rgbd()
{
    if (videoPreprocessorGPU == NULL) return;
    glColor4f(1.0f,1.0f,1.0f,1.0f);
    drawTextureDepthLIB(frame3C,-1,-1,2.0f);
}

void gputracker::render_rgb_tex(float z, bool overlayPoints)
{
    if (videoPreprocessorGPU == NULL) return;
    glColor4f(1.0f,1.0f,1.0f,1.0f);
    drawTextureLIB(frame3C,-1,-1,2.0f,z);
    if (overlayPoints) {
        glPushAttrib(GL_LIST_BIT | GL_CURRENT_BIT | GL_ENABLE_BIT | GL_TRANSFORM_BIT | GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_DEPTH_BUFFER_BIT);
        glDisable(GL_TEXTURE_2D);
        glDisable(GL_DEPTH_TEST);
        glColor4f(0.0f,1.0f,0.0f,1.0f);
        glPushMatrix();
        glScalef(2.0f/320.0f,2.0f/240.0f,1);
        glTranslatef(-160,-120,z+0.01f);
        if (cameraTrackingEnabled) {
            if (trackingMode == INCREMENTAL) {
                baseBuffer.renderDstPoints(pointSelectionAmount);
            } else {
                baseBuffer.renderDstPoints(keyFramePointSelectionAmount);
            }
        }
        glPopMatrix();
        glEnable(GL_DEPTH_TEST);
        glPopAttrib();
    }
}

void gputracker::render_depth(float z,int layer) {
    if (videoPreprocessorCPU == NULL) return;
    glColor4f(1.0f,1.0f,1.0f,1.0f);
    drawTextureLIB(depthCur1C[layer],-1,-1,2.0f,z);
}


void gputracker::render_ref_points() {
    /*glEnable(GL_TEXTURE_2D);
    glColor4f(1.0f,1.0f,1.0f,1.0f);
    drawTextureLIB(frame3C,-1,-1,2.0f,-3.0f);
*/
    glPushAttrib(GL_LIST_BIT | GL_CURRENT_BIT | GL_ENABLE_BIT | GL_TRANSFORM_BIT | GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glColor4f(0.0f,1.0f,0.0f,1.0f);
    glPushMatrix();
    glScalef(2.0f/320.0f,2.0f/240.0f,1);
    glTranslatef(-160,-120,-1.0f);

    if (cameraTrackingEnabled) {
        if (trackingMode == INCREMENTAL) {
            baseBuffer.renderSrcPoints(pointSelectionAmount);
        } else {
            baseBuffer.renderSrcPoints(keyFramePointSelectionAmount);
        }
    }

    glPopMatrix();
    glEnable(GL_DEPTH_TEST);
    glPopAttrib();
}


void gputracker::set_selected_points(int nPoints) {
//    if (trackingMode != INCREMENTAL) return;
    pointSelectionAmount = nPoints;
    if (videoPreprocessorCPU != NULL ) {
        videoPreprocessorCPU->setPixelSelectionAmount(pointSelectionAmount);
    }    
    keyFrameRing.setPointSelectionAmount(pointSelectionAmount);
}

int gputracker::get_selected_points() {
    return pointSelectionAmount;
}

void gputracker::set_camera_tracking(bool flag) {
    if (videoPreprocessorGPU == NULL) return;
    if (videoPreprocessorCPU == NULL) return;

    bool isPlaying = !videoPreprocessorGPU->isPaused();
    if (isPlaying != flag) videoPreprocessorGPU->pause();

    bool isPlayingCPU = !videoPreprocessorCPU->isPaused();
    if (isPlayingCPU != flag) videoPreprocessorCPU->pause();

    cameraTrackingEnabled = flag;
}

bool gputracker::playing() {
    bool isPlayingA = !videoPreprocessorGPU->isPaused();
    bool isPlayingB = !videoPreprocessorCPU->isPaused();
    cameraTrackingEnabled = isPlayingA && isPlayingB;
    return cameraTrackingEnabled;
}

void initialize_cuda()
{
    // Number of CUDA devices
    int devCount;
    cudaGetDeviceCount(&devCount);
    if (devCount == 0) { printf("no cuda devices found!\n"); exit(0); }
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);
    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }
//    cudaGLSetGLDevice(0);
    // Pick the device with highest Gflops/s
//    int devID = gpuGetMaxGflopsDeviceId();
    cudaGLSetGLDevice(0);
//    cudaSetDevice(0);
    //cudaGLSetGLDevice(0);
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaTest();

    if (greenScreenDev == NULL) cudaMalloc( (void **)&greenScreenDev, 320*240*3*sizeof(float));

    printFreeDeviceMemory();
    printf("gputracker::cuda initialized.\n");
}

void initialize_glew()
{
    glewInit();
    if (!glewIsSupported( "GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object" )) {
        printf("Error: GL_ARB_vertex_buffer_object or GL_ARB_pixel_buffer_object not supported.\n");
        exit(-1);
    }
    printf("gputracker::glew %s initialized.\n",glewGetString(GLEW_VERSION));
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
        parseIntegers(pElem->GetText(),5,&val[0]);
        printf("    updateFreqPhotometric  : %d, updateFreqICP : %d, nPoints: %d, icpRefReso: %d x %d\n",val[0],val[1],val[2]*1024,val[3],val[4]);
        photometricUpdateFreq = val[0];
        icpUpdateFreq = val[1];
        pointSelectionAmount = val[2]*1024;
        icpReferenceResoX = val[3];
        icpReferenceResoY = val[4];
        icpMatcher.setReferenceResolution(icpReferenceResoX,icpReferenceResoY);
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

int gputracker::initialize(const char *xmlConfigFile) {
    printf("gputracker init based on %s!\n",xmlConfigFile);
    load_parameters(xmlConfigFile);
    initialize_cuda();
    initialize_glew();
//    poseFilter.init(maxEstimationKeys);

    Mat flippedImage = Mat::zeros(480,640,CV_8UC3);
    Mat greenImage = Mat::zeros(480,640,CV_32FC3);

    float *ptr = (float*)greenImage.ptr();
    for (int i = 0; i < 640*480; i++) {
        ptr[i*3+0] = 0;
        ptr[i*3+1] = 1.0f;
        ptr[i*3+2] = 0;
    }
    if (greenScreenDev) {
        cudaMemcpyAsync(greenScreenDev,ptr,320*240*3*sizeof(float),cudaMemcpyHostToDevice);
    }

}

void gputracker::reset() {
    baseBuffer.reset();
    keyFrameRing.resetTransforms();
    videoPreprocessorGPU->reset();
    videoPreprocessorCPU->reset();
    if (keyFrameModel != NULL) keyFrameModel->resetTransforms();
    if (incrementalModel != NULL) {
        incrementalModel->removeKeyframes();

    }
    similarKeys.clear();
    currentFrame = 0;
    firstRefFrameUpdated = false;
    addKeyframe = true;
    similarKeyFrame = NULL;
    previousSimilarKeyFrame = NULL;
    icpMatcher.reset();
    identity4x4(&poseMatrix[0]);
    identity4x4(&icpPoseMatrix[0]);
    cameraTrackingEnabled = false;
}

void gputracker::set_calibration(const char *xmlCalibFileName) {
    calibKinect.init(xmlCalibFileName,false);
    calibKinect.setupCalibDataBuffer(320,240);
    icpMatcher.setCalib(&calibKinect.getCalibData()[KL_OFFSET],&calibKinect.getCalibData()[KR_OFFSET],&calibKinect.getCalibData()[TLR_OFFSET],&calibKinect.getCalibData()[KcR_OFFSET]);
    keyFrameRing.updateCalibration();
    if (videoPreprocessorGPU != NULL)
        videoPreprocessorGPU->updateCalibration();

    if (incrementalModel == NULL) {
        incrementalModel = new KeyFrameModel(slamDistTol,slamAngleTol,slamMaxKeys,&calibKinect,3,keyFramePointSelectionAmount,keyFramePointSelectionAmount,false,true,true);
        incrementalModel->setIterationCounts(&nIterations[0]);
    }
}

void gputracker::set_keyframe_model(const char *keyframeModelPath) {
    if (keyFrameModel != NULL)  {
        keyFrameModel->release();
        delete keyFrameModel; keyFrameModel = NULL;
    }
    keyFrameModel = new KeyFrameModel(keyframeModelPath,3,keyFramePointSelectionAmount,keyFramePointSelectionAmount,true,false,false);
    keyFrameModel->setIterationCounts(&nIterations[0]);
}

void selectXYZRange(cv::Mat &xyzMap,float minDepth,float maxDepth,Image2 &depthMap) {
    int sz = xyzMap.cols*xyzMap.rows;
    float *xyzPtr = (float*)xyzMap.ptr();
    float *zptr = (float*)depthMap.data;
    for (int i = 0; i < sz; i++) {
        float dval = fabs(xyzPtr[i*3+2]);
        if (dval < minDepth) dval = minDepth;
        if (dval > maxDepth) dval = maxDepth;
        zptr[i] = float(dval - minDepth)/float(maxDepth-minDepth);
    }
   depthMap.updateTexture(zptr);
}

// upload depth map to GPU with given dynamic color range
void selectRange(Mat &dmap,float minDepth,float maxDepth,Image2 &depthMap) {
    int sz = depthMap.width*depthMap.height;
    float *ptr = (float*)dmap.ptr();
    float *fptr = (float*)depthMap.data;
    for (int i = 0; i < sz; i++)
    {
        float dval = fabs(ptr[i]);
        if (dval < minDepth) dval = minDepth;
        if (dval > maxDepth) dval = maxDepth;
        fptr[i] = float(dval - minDepth)/float(maxDepth-minDepth);
    }
   depthMap.updateTexture(fptr);
}


void gputracker::set_source(VideoSource *stream) {
    if (videoPreprocessorGPU == NULL) {
        videoPreprocessorGPU = new VideoPreProcessorGPU(stream,3,&calibKinect);
        videoPreprocessorCPU = new VideoPreProcessorCPU(stream,3,&calibKinect);
        vbuffer = new VertexBuffer2(videoPreprocessorGPU->getDepthWidth()*videoPreprocessorGPU->getDepthHeight(),true,VERTEXBUFFER_STRIDE,"current");
//        trimesh = new TriangleBuffer2(NULL,videoPreprocessorGPU->getDepthWidth()*videoPreprocessorGPU->getDepthHeight()*2,NULL,"trimesh buffer");
        printf("resogun %d x %d\n",icpReferenceResoX,icpReferenceResoY);
        trimesh = new TriangleBuffer2(NULL,icpReferenceResoX*icpReferenceResoY*2,NULL,"trimesh buffer");
        phongShader = new Shader("shaders/phongVS.glsl","shaders/phongPS.glsl");
        createHdrImage(NULL,videoPreprocessorGPU->getWidth(),videoPreprocessorGPU->getHeight(),3,&frame3C,ONLY_GPU_TEXTURE, false); frame3C.setName("frame3C");
        createImage(NULL,videoPreprocessorGPU->getWidth(),videoPreprocessorGPU->getHeight(),3,videoPreprocessorGPU->getWidth()*3,&icpSelection3C,ONLY_GPU_TEXTURE, true); icpSelection3C.setName("icpSelection3C");
        createHdrImage(NULL,videoPreprocessorGPU->getWidth(),videoPreprocessorGPU->getHeight(),1,&depth1C,ONLY_GPU_TEXTURE, false); depth1C.setName("depth1C");
        frame1C.createHdrPyramid(videoPreprocessorGPU->getWidth(),videoPreprocessorGPU->getHeight(),1,3,false,ONLY_GPU_TEXTURE);  frame1C.setName("frame1C");
        // initialize ICP visualization images
        for (int i = 0; i < 3; i++) {
            // only for ICP visualization :
            char buf[512];
            sprintf(buf, "depthCur1C-%d",i);
            unsigned int w = videoPreprocessorGPU->getDepthWidth();  w = w >> i;
            unsigned int h = videoPreprocessorGPU->getDepthHeight(); h = h >> i;
            createHdrImage(NULL,w,h,1,&depthCur1C[i],CREATE_GPU_TEXTURE);  depthCur1C[i].setName(buf);
            sprintf(buf, "normalMap3C-%d",i);
            createHdrImage(NULL, w,h, 3,&normalMap3C[i],CREATE_GPU_TEXTURE);  normalMap3C[i].setName(buf);
        }
        cudaMalloc((void **)&imDepthDev,videoPreprocessorGPU->getWidth()*videoPreprocessorGPU->getHeight()*sizeof(float));
        keyFrameRing.init(1,videoPreprocessorGPU->getWidth(),videoPreprocessorGPU->getHeight(),3,calibKinect);
        baseBuffer.initialize();
    }
    videoPreprocessorGPU->setVideoSource(stream);
    videoPreprocessorCPU->setVideoSource(stream);
    reset();
}

float gputracker::get_fov_x() {
    return calibKinect.getFovX();
}

float gputracker::get_fov_y() {
    return calibKinect.getFovY();
}

int gputracker::get_frame_index() {
    if (videoPreprocessorGPU == NULL) return 0;
    return videoPreprocessorGPU->getFrame();
}

void gputracker::set_keyframe_mode() {
    trackingMode = KEYFRAME;
    printf("keyframe mode set!\n");
    fflush(stdin); fflush(stdout);
    similarKeys.clear();
    useParallelDICP = false;
    gputracker::set_selected_points(0);
}

int gputracker::get_update_freq() {
    return photometricUpdateFreq;
}

void gputracker::set_incremental_mode() {
    trackingMode = INCREMENTAL;
    printf("incremental mode set!\n");
    useParallelDICP = true;
    fflush(stdin); fflush(stdout);
    similarKeys.clear();
    firstRefFrameUpdated = false;
    keyFrameRing.setTransforms(baseBuffer.getCurrentPose());
}

void gputracker::set_hybrid_mode() {
    if (incrementalModel != NULL) {
        trackingMode = HYBRID;
        similarKeys.clear();
        useParallelDICP = false;
        gputracker::set_selected_points(keyFramePointSelectionAmount);
        printf("hybrid mode set!\n");
        fflush(stdin); fflush(stdout);
    } else {
        printf("calibration not set, could not set hybrid mode!\n");
        fflush(stdin); fflush(stdout);
    }
    firstRefFrameUpdated = false;
    addKeyframe = true;
}

void gputracker::release() {
    if (videoPreprocessorGPU != NULL) {
        videoPreprocessorGPU->release(); delete videoPreprocessorGPU; videoPreprocessorGPU = NULL;
        videoPreprocessorCPU->release(); delete videoPreprocessorCPU; videoPreprocessorCPU = NULL;
        printf("videpreprocessor release!\n");
        fflush(stdin);
        fflush(stdout);

        if (vbuffer != NULL) { vbuffer->release(); delete vbuffer; vbuffer = NULL; }        
        printf("vbuffer release!\n");        
        if (trimesh != NULL) { trimesh->release(); delete trimesh; trimesh = NULL; }
        printf("trimesh release!\n");
        if (phongShader != NULL) { phongShader->release(); delete phongShader; phongShader = NULL; }
        printf("phong shader release!\n");
        fflush(stdin);
        fflush(stdout);


        frame3C.releaseData();
        icpSelection3C.releaseData(); icpSelectionImage.release();        
        for (int i=0; i < 3; i++) {
            // only for ICP visualization:
            depthCur1C[i].releaseData();
            normalMap3C[i].releaseData();
        }

        depth1C.releaseData();
        frame1C.releaseData();
        if (imDepthDev != NULL) cudaFree(imDepthDev); imDepthDev = NULL;
        if (greenScreenDev != NULL) cudaFree(greenScreenDev); greenScreenDev = NULL;

        printf("images release!\n");
        fflush(stdin);
        fflush(stdout);

        keyFrameRing.release();
        printf("keyframering release!\n");
        fflush(stdin);
        fflush(stdout);

        if (keyFrameModel != NULL) {
            keyFrameModel->release();
            delete keyFrameModel; keyFrameModel = NULL;
            printf("keyframemodel release!\n");
            fflush(stdin);
            fflush(stdout);
        }

        if (incrementalModel != NULL) {
            printf("trying incremental model release!\n");
            fflush(stdin);
            fflush(stdout);
            incrementalModel->release();
            delete incrementalModel; incrementalModel = NULL;
            printf("incremental model release!\n");
            fflush(stdin);
            fflush(stdout);
        }

        icpMatcher.release();
        fflush(stdin);
        fflush(stdout);

        baseBuffer.release();        
        //poseFilter.release();
        printf("misc release!\n");
        fflush(stdin);
        fflush(stdout);

    }
    fflush(stdin);
    fflush(stdout);
    printf("gputracker release!\n");
    fflush(stdin);
    fflush(stdout);

    checkCudaError("gputracker::cuda shutdown error");
    cudaDeviceReset();
}

void lockGPUBuffers() {
    frame1C.lock(); frame3C.lock(); vbuffer->lock(); vbuffer->lockIndex(); depth1C.lock();
}

void unlockGPUBuffers() {
    vbuffer->unlockIndex(); vbuffer->unlock(); frame3C.unlock(); frame1C.unlock(); depth1C.unlock();
}

// unauthorized OpenGL copy to Panda3D texture
void gputracker::fill_rgbtex(unsigned int texID) {
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, frame3C.pbo);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 320,240, 0, GL_RGB, GL_FLOAT, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);
    glBindTexture(GL_TEXTURE_2D,0);
}

void gputracker::fill_depthtex(unsigned int depthID) {
    glBindTexture(GL_TEXTURE_2D, depthID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, depth1C.pbo);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, 320,240, 0, GL_LUMINANCE, GL_FLOAT, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);
    glBindTexture(GL_TEXTURE_2D,0);
}


KeyFrame *updateNearestKeyframes(float *mtx, std::vector<KeyFrame*> &similarKeys, int maxKeys) {

    similarKeys.clear();
    if (trackingMode == KEYFRAME)
        keyFrameModel->findSimilarKeyFrames(mtx,similarKeys);
    else if (trackingMode == HYBRID) {
        incrementalModel->findSimilarKeyFrames(mtx,similarKeys);
        // if keyframe does not exist closeby add new one
        if (similarKeys.size() < 1) {
            addKeyframe = true;
            incrementalModel->findSimilarKeyFramesLarge(mtx,similarKeys);
        }
    }

    if (similarKeys.size() < 1) {
        printf("ref not found!\n"); fflush(stdin); fflush(stdout);
        similarKeyFrame = NULL;
        return NULL;
    }

    // remove limit exceeding keyframes (the worst hits)
    while (similarKeys.size() > maxKeys) similarKeys.pop_back();

    if (previousSimilarKeyFrame != similarKeys[0] && previousSimilarKeyFrame != NULL) {
        // initialize delta transformations for the found keys
        for (size_t k = 0; k < similarKeys.size(); k++) {
            KeyFrame *kf = similarKeys[k];
            kf->setupRelativeCPUTransform(mtx);
        }
    }

    similarKeyFrame = similarKeys[0];
    return similarKeyFrame;
}

void lockKeys(std::vector<KeyFrame *> &keys) {
    for (size_t k = 0; k < keys.size(); k++) {
        KeyFrame *kf = keys[k];
        kf->lock();
    }
}

void unlockKeys(std::vector<KeyFrame *> &keys) {
    for (size_t k = 0; k < keys.size(); k++) {
        KeyFrame *kf = keys[k];
        kf->unlock();
    }
}

void optimizeKeys(std::vector<KeyFrame *> &keys, ImagePyramid2 &frame1C, VertexBuffer2 *vbuffer) {
    for (size_t k = 0; k < keys.size(); k++) {
        KeyFrame *kf = keys[k];

/*        cudaEvent_t start,stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,0);
        float delay = 0;*/
        kf->optimizePose(frame1C,vbuffer,filterDepthFlag);
/*        cudaEventRecord(stop,0); cudaThreadSynchronize(); cudaEventElapsedTime(&delay, start, stop); printf("opt time: %3.1f\n",delay);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);*/
        // TODO: is it possible to initialize relative transforms (similarKey[k]) using the similarKey[0] result?
        //kf->setupRelativeCPUTransform(baseBuffer.getCurrentPose());
    }
}

/*
void filterKeys(std::vector<KeyFrame *> &keys) {
    poseFilter.clear();
    float maxWeight = 0;
    for (size_t k = 0; k < keys.size(); k++) {
        KeyFrame *kf = keys[k];
        poseFilter.addPose(kf->getNextBaseDev(),kf->weight);
        cpuWeights[k] = kf->weight;
        if (kf->weight > maxWeight) maxWeight = kf->weight;
    }

    for (size_t k = 0; k < keys.size(); k++) {
        cpuWeights[k] /= maxWeight;
    }
}*/


void gputracker::get_pose(float *poseMatrixDst, float *icpPose) {
    memcpy(poseMatrixDst,poseMatrix,sizeof(float)*16);
    if (icpPose != NULL) {
        // convert from IR coordinate system to RGB camera coordinate system
        float *TRL = &calibKinect.getCalibData()[TRL_OFFSET];
        float *TLR = &calibKinect.getCalibData()[TLR_OFFSET];
        float T[16];
        // inv(TLR*inv(icpPoseMat)) = icpPoseMat * TRL
        matrixMult4x4(&icpPoseMatrix[0],TRL,&T[0]);
        matrixMult4x4(TLR,&T[0],&T[0]);
        transpose4x4(&T[0],icpPose);
    }
}

void gatherNearestKeys() {
    if (!cameraTrackingEnabled) return;
    static int runTimesSincePreviousKeyAddition = 0;
    runTimesSincePreviousKeyAddition++;
    if (runTimesSincePreviousKeyAddition > 15 || !firstRefFrameUpdated) {
        runTimesSincePreviousKeyAddition = 0;
        updateNearestKeyframes(&poseMatrix[0],similarKeys,maxEstimationKeys);
        if (similarKeys.size() > 0) {
            firstRefFrameUpdated = true;
        }
    }
}

int gputracker::track_frame() {
    if (videoPreprocessorGPU == NULL || videoPreprocessorCPU == NULL) return 0;

//    cameraTrackingEnabled = videoPreprocessorGPU->isPlaying() && videoPreprocessorCPU->isPlaying();
    // preprocess rgb-d input
    // - correct format: 320x240x3 rgb, 320x240x1 depth
    // - reconstruct 3D point cloud into gpu memory
    // - also cpu depth map available
    // - additional cuda operations on current image possible (buffers locked to cuda)

    lockGPUBuffers();

    int retA = 1;
    if (useParallelDICP) {
        retA = videoPreprocessorCPU->preprocess();
        // also update depth map on CPU:
        icpMatcher.setDepthMap(videoPreprocessorCPU->getDepthImageL(),videoPreprocessorCPU->getGrayImage(),pointSelectionAmount, nPhotometricReferences);
        videoPreprocessorCPU->getPlane(&planeMean[0],&planeNormal[0]);
    }
    int retB = videoPreprocessorGPU->preprocess(frame1C,frame3C,vbuffer,imDepthDev,depth1C,trackingMode==KEYFRAME);
    if (!retA || !retB) {
        unlockGPUBuffers();
        reset();
        return 0;
    }

    if (blankImagesFlag) {
        cudaMemcpyAsync(frame3C.devPtr,greenScreenDev,frame3C.height*frame3C.pitch,cudaMemcpyDeviceToDevice);
        //printf("joo\n"); fflush(stdin); fflush(stdout);
       // cudaMemset(frame3C.devPtr,0,frame3C.height*frame3C.pitch);
        cudaMemsetAsync(depth1C.devPtr,0,depth1C.height*depth1C.pitch);
    }

    if (trackingMode != INCREMENTAL) {
        gatherNearestKeys();
    }

    bool updatePhotometricRef = !firstRefFrameUpdated && cameraTrackingEnabled;
    bool updateICPRef = updatePhotometricRef;
    if (trackingMode == INCREMENTAL && (currentFrame % photometricUpdateFreq == 0)) updatePhotometricRef =  true;
    if (trackingMode == INCREMENTAL && (currentFrame % icpUpdateFreq == 0)) updateICPRef =  true;
    if (trackingMode == HYBRID && addKeyframe) updatePhotometricRef = true;

    lockKeys(similarKeys);
    if (cameraTrackingEnabled && firstRefFrameUpdated && (similarKeys.size() > 0)) {
        // optimize transformation params for mapping ref -> cur
        /////////////////////////// resultTxt.addPose(baseBuffer.getCurrentPose());
        optimizeKeys(similarKeys,frame1C,vbuffer); previousSimilarKeyFrame = similarKeys[0];
        similarKeys[0]->updateBase(baseBuffer,frame1C);
        memcpy(&poseMatrix[0],baseBuffer.getCurrentPose(),sizeof(float)*16);
        if (useParallelDICP) {
            // also align depth maps on CPU:
            icpMatcher.optimize(&nIterations[0],verboseOpt);
            matrixMult4x4(&icpPoseMatrix[0],icpMatcher.getIncrement(),&icpPoseMatrix[0]);
        } else identity4x4(&icpPoseMatrix[0]);
        currentFrame++;
    }

    if (updatePhotometricRef) {
        if (trackingMode == INCREMENTAL) {
            keyFrameRing.updateSingleReference(currentFrame, keyFrameRing.getKeyFrame(0),frame1C,frame3C,vbuffer,imDepthDev,pointSelectionAmount);
            similarKeys.clear(); similarKeys.push_back(keyFrameRing.getKeyFrame(0));
            firstRefFrameUpdated = true;            
        } else if (trackingMode == HYBRID && videoPreprocessorGPU->isPlaying()) {
            KeyFrame *newKey = incrementalModel->extendMap(currentFrame,frame1C,frame3C,vbuffer,imDepthDev,keyFramePointSelectionAmount,&poseMatrix[0],previousSimilarKeyFrame);
            if (newKey != NULL) {
                similarKeys.insert(similarKeys.begin(), newKey); previousSimilarKeyFrame = newKey;
                // remove limit exceeding keyframes (the worst hits)
                while (similarKeys.size() > maxEstimationKeys) similarKeys.pop_back();
                addKeyframe = false;
            }
        }
    }
    unlockKeys(similarKeys);

    if (trackingMode == INCREMENTAL && useParallelDICP) {
        icpMatcher.updatePhotoReference();
        if (updateICPRef) {
            // also update reference depth map on CPU:
            icpMatcher.updateReference();
        }
        // update trimesh using ICP reference
        cv::Mat *xyzImage; int stride;
        icpMatcher.getReferenceCloud(&xyzImage,&stride);
        if (trimesh!=NULL) trimesh->update(xyzImage,stride);
        icpMatcher.markPhotometricSelection(icpSelectionImage,0,255,0);
        icpSelection3C.updateTexture(icpSelectionImage.ptr());
    }
    unlockGPUBuffers();
    return 1;
}

