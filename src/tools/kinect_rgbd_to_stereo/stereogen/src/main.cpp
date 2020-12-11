/*
stereo-gen
Copyright (c) 2014, Tommi Tykkälä, All rights reserved.

This source code is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3.0 of the License, or (at your option) any later version.

This source code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this source code.
*/

#include <math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>
//#include <helper_functions.h> // Helper functions (utilities, parsing, timing)
#include "GLSLProgram.h"
#include "SmokeShaders.h"
#include "keyframe.h"
#include "basic_math.h"
#include "screenshot.h"
#include "calib.h"
#include "fileSource.h"
#include "VideoPreProcessorCPU.h"
#include "SmokeRenderer.h"
#include "GroundTruth.h"

void initDisplayThread();
bool g_killThreads = false;

SmokeRenderer *renderer   = 0;
ScreenShot    *screenShot = 0;
const int texWidth = 640,        texHeight = 480;
int winWidth       = 2*texWidth, winHeight = texHeight;
int currentFrame = 0;
int layer = 0;
bool saveNextFrame = false;
float *camTrack = NULL;
int numCameraPoses = 0;
// view params
int ox, oy;
int buttonState = 0;
bool keyDown[256];
bool saving = true;

nv::vec3f cameraPos(0, 0, 1000);
nv::vec3f cameraRot(0, 0, 0);
nv::vec3f cameraPosLag(cameraPos);
nv::vec3f cameraRotLag(cameraRot);
nv::vec3f cursorPos(0, 1, 0);
nv::vec3f cursorPosLag(cursorPos);
float fov = 40.0f;
nv::vec3f lightPos(5.0, 5.0, -5.0);
nv::vec3f trackPosSpeed;
nv::vec3f trackAngleSpeed;

const float inertia = 0.1f;
const float translateSpeed = 100.000f;
const float cursorSpeed = 0.01f;
const float rotateSpeed = 0.2f;
const float walkSpeed = 10.00f;

// simulation parameters
float currentTime = 0.0f;
Calibration calibKinect;
FileSource *fileSource = NULL;
float modelView[16];
float baseModelView[16];
Keyframe keyframe;
GLuint framebuffer = 0;     // to bind the proper targets
GLuint depth_buffer = 0;    // for proper depth test while rendering the scene
GLuint inputTexture = 0;      // where we render the image
GLuint sensorRGBTex = 0;      // where we render the image
GLuint sensorDepthTex = 0;      // where we render the image
GLuint debugTexture = 0;
GLuint debugTexture1C[4] = {0,0,0,0};
GLuint fbo_source = 0;
GLuint outputTexture;  // where we will copy the CUDA result
struct cudaGraphicsResource *cudaInputTexture = NULL;
struct cudaGraphicsResource *cudaOutputTexture = NULL;
struct cudaGraphicsResource *cudaDebugTexture = NULL;
struct cudaGraphicsResource *cudaDebugTexture1C[4] = {NULL,NULL,NULL,NULL};

cudaArray *rgbdFrame = NULL;
VideoPreProcessorCPU *preprocessor = NULL;

const char *sRefBin[]  =
{
    "ref_smokePart_pos.bin",
    "ref_smokePart_vel.bin",
    NULL
};
float TtmpB[16];
float *convertT(float *Tin) {
    invertRT4(&Tin[0],&TtmpB[0]);
    return &TtmpB[0];
}

void createCudaTexture(GLuint *tex_cudaResult, bool gray, unsigned int size_x, unsigned int size_y, cudaGraphicsResource **cudaTex)
{
    // create a texture
    glGenTextures(1, tex_cudaResult);
    glBindTexture(GL_TEXTURE_2D, *tex_cudaResult);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    if (!gray) {
      //  printf("Creating a Texture GL_RGBA32F_ARB\n");
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, size_x, size_y, 0, GL_RGBA, GL_FLOAT, NULL);
    } else {
      //  printf("Creating a Texture GL_LUMINANCE32F_ARB\n");
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, size_x, size_y, 0, GL_LUMINANCE, GL_FLOAT, NULL);
    }

    SDK_CHECK_ERROR_GL();
    // register this texture with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterImage(cudaTex, *tex_cudaResult, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
}

void createTextureSrc(GLuint *tex_screen, unsigned int size_x, unsigned int size_y)
{
    // create a texture
    glGenTextures(1, tex_screen);
    glBindTexture(GL_TEXTURE_2D, *tex_screen);
    // buffer data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, size_x, size_y, 0, GL_RGBA, GL_FLOAT, NULL);
    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);//GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);//GL_NEAREST);
    SDK_CHECK_ERROR_GL();
    // register this texture with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaInputTexture, *tex_screen, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));    

    // allocate extra rgbdFrame for producing delay - 1 to syntetic inputs:
    cudaChannelFormatDesc channelDesc;
    channelDesc = cudaCreateChannelDesc(32,32,32,32, cudaChannelFormatKindFloat);
    checkCudaErrors(cudaMallocArray(&rgbdFrame, &channelDesc, size_x, size_y));
}

void createSensorTextures(GLuint *sensorRGBTex, GLuint *sensorDepthTex, int size_x, int size_y) {
    // create a texture
    glGenTextures(1, sensorDepthTex);
    glBindTexture(GL_TEXTURE_2D, *sensorDepthTex);
    // buffer data
  //  printf("Creating a Texture GL_LUMINANCE32F_ARB\n");

    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, size_x, size_y, 0, GL_LUMINANCE, GL_FLOAT, NULL);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    SDK_CHECK_ERROR_GL();

    // create a texture
    glGenTextures(1, sensorRGBTex);
    glBindTexture(GL_TEXTURE_2D, *sensorRGBTex);
    // buffer data
  //  printf("Creating a Texture GL_RGB\n");

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, size_x, size_y, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    SDK_CHECK_ERROR_GL();
}


void createDepthBuffer(GLuint *depth, unsigned int size_x, unsigned int size_y)
{
    // create a renderbuffer
    glGenRenderbuffersEXT(1, depth);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, *depth);

    // allocate storage
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, size_x, size_y);

    // clean up
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);

    SDK_CHECK_ERROR_GL();
}

void createFramebuffer(GLuint *fbo, GLuint color, GLuint depth)
{    
    // create and bind a framebuffer
    glGenFramebuffersEXT(1, fbo);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, *fbo);

    // attach images
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, color, 0);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depth);

    // clean up
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    SDK_CHECK_ERROR_GL();

}

void deleteTexture(GLuint *tex)
{
    glDeleteTextures(1, tex);
    SDK_CHECK_ERROR_GL();

    *tex = 0;
}

void deleteDepthBuffer(GLuint *depth)
{
    glDeleteRenderbuffersEXT(1, depth);
    SDK_CHECK_ERROR_GL();

    *depth = 0;
}

void deleteFramebuffer(GLuint *fbo)
{
    glDeleteFramebuffersEXT(1, fbo);
    SDK_CHECK_ERROR_GL();

    *fbo = 0;
}

void resetVariables(int frame) {
    identity4x4(&modelView[0]);
    if (frame < 0) frame = 0; if (frame > numCameraPoses-1) frame = numCameraPoses-1;
    memcpy(&baseModelView[0],&camTrack[frame*16],sizeof(float)*16);
    identity4x4(&TtmpB[0]);
    nv::vec3f zeroVec(0,0,0);
    cursorPos    = zeroVec;
    cursorPosLag = zeroVec;
    cameraPos    = zeroVec;
    cameraPosLag = cameraPos;
    cameraRot    = zeroVec;
    cameraRotLag = zeroVec;
    renderer->setCameraMatrix(modelView);
    currentFrame = 0;
}


void release()
{
    if (camTrack != NULL) delete[] camTrack;
    if(renderer!=NULL)    delete renderer;
    if (screenShot!=NULL) delete screenShot;
    if(fileSource!=NULL)  delete fileSource;
     // unregister this buffer object with CUDA
    if (cudaInputTexture != NULL)  checkCudaErrors(cudaGraphicsUnregisterResource(cudaInputTexture));
    if (cudaOutputTexture != NULL) checkCudaErrors(cudaGraphicsUnregisterResource(cudaOutputTexture));
    if (cudaDebugTexture != NULL)  checkCudaErrors(cudaGraphicsUnregisterResource(cudaDebugTexture));
    for (int i = 0; i < 4; i++) {
        if (cudaDebugTexture1C[i] != NULL) checkCudaErrors(cudaGraphicsUnregisterResource(cudaDebugTexture1C[i]));
        if (debugTexture1C[i])  deleteTexture(&debugTexture1C[i]);
    }
    if (outputTexture)  deleteTexture(&outputTexture);
    if (debugTexture)   deleteTexture(&debugTexture);
    if (inputTexture)   deleteTexture(&inputTexture);
    if (sensorRGBTex)   deleteTexture(&sensorRGBTex);
    if (sensorDepthTex) deleteTexture(&sensorDepthTex);
    if (depth_buffer)   deleteDepthBuffer(&depth_buffer);
    if (framebuffer)    deleteFramebuffer(&framebuffer);
    if (rgbdFrame)      cudaFreeArray(rgbdFrame);

    if (preprocessor) { preprocessor->release(); delete preprocessor; }
    cudaDeviceReset();
}

// GLUT callback functions
void reshape(int w, int h)
{
      winWidth = w;
      winHeight = h;
}

void mouse(int button, int state, int x, int y)
{
    int mods;

    if (state == GLUT_DOWN)
    {
        buttonState |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    mods = glutGetModifiers();

    if (mods & GLUT_ACTIVE_SHIFT)
    {
        buttonState = 2;
    }
    else if (mods & GLUT_ACTIVE_CTRL)
    {
        buttonState = 3;
    }

    ox = x;
    oy = y;

    glutPostRedisplay();
}

// transfrom vector by matrix
void xform(nv::vec3f &v, nv::vec3f &r, float *m)
{
    r.x = v.x*m[0] + v.y*m[4] + v.z*m[8] + m[12];
    r.y = v.x*m[1] + v.y*m[5] + v.z*m[9] + m[13];
    r.z = v.x*m[2] + v.y*m[6] + v.z*m[10] + m[14];
}

// transform vector by transpose of matrix (assuming orthonormal)
void ixform(nv::vec3f &v, nv::vec3f &r, float *m)
{
    r.x = v.x*m[0] + v.y*m[4] + v.z*m[8];
    r.y = v.x*m[4] + v.y*m[5] + v.z*m[9];
    r.z = v.x*m[8] + v.y*m[6] + v.z*m[10];
}

void motion(int x, int y)
{
     float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 1)
    {
        // left = rotate
        cameraRot[0] += dy * rotateSpeed;
        cameraRot[1] += dx * rotateSpeed;
    }

    if (buttonState == 2)
    {
        // middle = translate
        nv::vec3f v = nv::vec3f(dx*translateSpeed, -dy*translateSpeed, 0.0f);
        nv::vec3f r;
        ixform(v, r, modelView);
        cameraPos += r;
    }

    if (buttonState == 3)
    {
        // left+middle = zoom
        nv::vec3f v = nv::vec3f(0.0, 0.0, dy*translateSpeed);
        nv::vec3f r;
        ixform(v, r, modelView);
        cameraPos += r;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
    case '\033':
        g_killThreads = true;
        break;
    case ' ':
        saveNextFrame = true;
        break;
    case 'n':
        keyframe.setPose(convertT(&modelView[0]),fov,(float)texWidth/(float)texHeight);
        break;
    case 'r':
        resetVariables(currentFrame);
        break;
    case '+':
        currentFrame = (currentFrame+1)%numCameraPoses; break;
    case '-':
        currentFrame = (currentFrame+numCameraPoses-1)%numCameraPoses; break;
    }
    keyDown[key] = true;
    glutPostRedisplay();
}

void keyUp(unsigned char key, int /*x*/, int /*y*/)
{
    keyDown[key] = false;
}


void idle(void)
{
    // move camera in view direction
    /*
        0   4   8   12  x
        1   5   9   13  y
        2   6   10  14  z
    */
    if (keyDown['w'])
    {
        cameraPos[0] += -modelView[8]  * walkSpeed;
        cameraPos[1] += -modelView[9]  * walkSpeed;
        cameraPos[2] += -modelView[10] * walkSpeed;
    }

    if (keyDown['s'])
    {
        cameraPos[0] -= -modelView[8] * walkSpeed;
        cameraPos[1] -= -modelView[9] * walkSpeed;
        cameraPos[2] -= -modelView[10] * walkSpeed;
    }

    if (keyDown['a'])
    {
        cameraPos[0] -= modelView[0] * walkSpeed;
        cameraPos[1] -= modelView[1] * walkSpeed;
        cameraPos[2] -= modelView[2] * walkSpeed;
    }

    if (keyDown['d'])
    {
        cameraPos[0] += modelView[0] * walkSpeed;
        cameraPos[1] += modelView[1] * walkSpeed;
        cameraPos[2] += modelView[2] * walkSpeed;
    }

    if (keyDown['e'])
    {
        cameraPos[0] += modelView[4] * walkSpeed;
        cameraPos[1] += modelView[5] * walkSpeed;
        cameraPos[2] += modelView[6] * walkSpeed;
    }

    if (keyDown['q'])
    {
        cameraPos[0] -= modelView[4] * walkSpeed;
        cameraPos[1] -= modelView[5] * walkSpeed;
        cameraPos[2] -= modelView[6] * walkSpeed;
    }
//    glutPostRedisplay();
}

void mainMenu(int i)
{
    key((unsigned char) i, 0, 0);
}

// initialize OpenGL
void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(winWidth, winHeight);
    glutCreateWindow("stereogen");

    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5"))
    {
        fprintf(stderr, "The following required OpenGL extensions missing:\n\tGL_VERSION_2_0\n\tGL_VERSION_1_5\n");
        exit(EXIT_SUCCESS);
    }

    if (!glewIsSupported("GL_ARB_multitexture GL_ARB_vertex_buffer_object GL_EXT_geometry_shader4"))
    {
        fprintf(stderr, "The following required OpenGL extensions missing:\n\tGL_ARB_multitexture\n\tGL_ARB_vertex_buffer_object\n\tGL_EXT_geometry_shader4.\n");
        exit(EXIT_SUCCESS);
    }
//GEOMETRY_VERTICES_OUT_EXT
#if defined (_WIN32)

    if (wglewIsSupported("WGL_EXT_swap_control"))
    {
        // disable vertical sync
        wglSwapIntervalEXT(0);
    }

#endif

    glEnable(GL_DEPTH_TEST);


    glutReportErrors();
}

void printDevProp( cudaDeviceProp devProp )
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %u\n",  (unsigned int)devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n",  (unsigned int)devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %u\n",  (unsigned int)devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %u\n",  (unsigned int)devProp.totalConstMem);
    printf("Texture alignment:             %u\n",  (unsigned int)devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multi-processors:    %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    printf("Can map host memory:           %s\n",  (devProp.canMapHostMemory ? "Yes" : "No"));
    return;
}


void writePPM(const char *fn, int dimx, int dimy, float *src) {
    float zcap = 50.0f;
    float minZ = 1e8f;
    float maxZ = 0.0f;
    FILE *fp = fopen(fn, "wb"); /* b - binary mode */
    fprintf(fp, "P6\n%d %d\n255\n", dimx, dimy);
    unsigned char *buf = new unsigned char[dimx*dimy*3];
    int off4 = 0;
    for (int j = 0; j < dimy; j++) {
        int off3 = (dimy-1-j)*dimx*3;
        for (int i = 0; i < dimx; i++,off3+=3,off4+=4) {
            float zval = -src[off4+3];
            if (zval < minZ) minZ = zval;
            if (zval > maxZ) maxZ = zval;
            if (zval < 0.0f) zval = 0;
            if (zval > zcap) zval = zcap;
            zval = 255*zval/zcap;
            buf[off3+0] = zval;//src[off4+0]*255.0f;
            buf[off3+1] = zval;//src[off4+1]*255.0f;
            buf[off3+2] = zval;//src[off4+2]*255.0f;
        }
    }
    printf("minZ: %f maxZ: %f\n",minZ,maxZ);
    fwrite(buf,dimx*dimy*3,1,fp);
    delete[] buf;
    fclose(fp);
}

void captureScreen() {

    // map buffer objects to get CUDA device pointers
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaInputTexture, 0));
    //printf("Mapping tex_in\n");
    cudaArray_t inArray;
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&inArray, cudaInputTexture, 0, 0));

//    unsigned char *dst = new unsigned char[winWidth*winHeight*4];
    float *dst = new float[texWidth*texHeight*4];
    checkCudaErrors(cudaMemcpy2DFromArray(dst, texWidth*16, inArray, 0, 0, texWidth*16, texHeight,cudaMemcpyDeviceToHost));
    writePPM("testi.ppm",texWidth,texHeight,dst);
    delete[] dst;

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaInputTexture, 0));
}

void lockRGBD(cudaGraphicsResource_t &cuda_tex, cudaArray_t &inArray) {
    // map buffer objects to get CUDA device pointers
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex, 0));
    //printf("Mapping tex_in\n");
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&inArray, cuda_tex, 0, 0));
}

void unlockRGBD(cudaGraphicsResource_t  &cuda_tex) {
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex, 0));
}

void blitToPrimaryScreen() {
    //sdkStartTimer(&timer);
    glBindFramebufferEXT( GL_READ_FRAMEBUFFER_EXT, framebuffer ); // set target as primary backbuffer
    glBindFramebufferEXT( GL_DRAW_FRAMEBUFFER_EXT, 0 ); // set target as primary backbuffer
    glBlitFramebufferEXT(0, 0, texWidth,texHeight, 0, 0, winWidth/2, winHeight, GL_COLOR_BUFFER_BIT , GL_LINEAR);
}

void uploadCudaTexture(void *cudaDevData, int sizeTexBytes, cudaGraphicsResource *cudaTexture) {
    cudaArray *texture_ptr;
    cudaThreadSynchronize();
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaTexture, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cudaTexture, 0, 0));
    checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, cudaDevData, sizeTexBytes, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaTexture, 0));

}


void updateRGBDTextures() {
    preprocessor->preprocess(true,false,false);
    Mat &depthImage = preprocessor->getDepthImageL();
    if (depthImage.cols != texWidth || depthImage.rows != texHeight) { printf("updateSensorTextures: dimension mismatch!\n"); return;}
    glBindTexture(GL_TEXTURE_2D,sensorDepthTex);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,texWidth,texHeight,GL_LUMINANCE,GL_FLOAT,depthImage.ptr());
    glBindTexture(GL_TEXTURE_2D,sensorRGBTex);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,texWidth,texHeight,GL_RGB,GL_UNSIGNED_BYTE,preprocessor->getDistortedRGBImage().ptr());
    glBindTexture(GL_TEXTURE_2D,0);
}

void updateFrame() {

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuffer);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);    
    // note: this function does not correctly align the images, merely combines rgb-d data from 2 source textures
    //renderer->displayRGBDImage(sensorRGBTex,sensorDepthTex, 0,0, texWidth,texHeight,calibKinect.getMinDist(),calibKinect.getMaxDist());
    renderer->renderPointCloud(preprocessor->getPointCloud(),texWidth,texHeight,0,0,texWidth,texHeight,calibKinect.getMinDist(),calibKinect.getMaxDist(),sensorRGBTex,"");
    // copy the rgb data onto primary screen:
    blitToPrimaryScreen();
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    cudaStreamSynchronize(0);
    cudaArray *mappedFrame = NULL;
    lockRGBD(cudaInputTexture,mappedFrame);
    // update rgbdFrame as the "current frame"
    cudaMemcpy2DArrayToArray(rgbdFrame,0,0,mappedFrame,0,0,texWidth*sizeof(float4),texHeight,cudaMemcpyDeviceToDevice);
//    cudaStreamSynchronize(0);
    unlockRGBD(cudaInputTexture);    
}

void drawCameraEstimationView(float *cameraPoseA, float *cameraPoseB, float x0, float y0, float width, float height) {
    float modelView1[16];  memcpy(&modelView1[0], renderer->getCameraMatrix(),sizeof(float)*16);
    float modelView2[16];
    identity4x4(&modelView2[0]);
    modelView2[3]  +=   0;
    modelView2[7]  +=   0;
    modelView2[11] -= 750;
    // render RGB-D cloud farther away to show camera trajectories in front of the view:
    renderer->setCameraMatrix(&modelView2[0]);
    renderer->renderPointCloud(preprocessor->getPointCloud(),texWidth,texHeight,x0,y0,width,height,calibKinect.getMinDist(),calibKinect.getMaxDist(),sensorRGBTex,"");
    renderer->setCameraMatrix(&modelView1[0]);

    float modelView2T[16]; transpose4x4(&modelView2[0],&modelView2T[0]);
    renderer->resetView3d(x0,y0,width,height);
    glPushMatrix();
    glLoadMatrixf(&modelView2T[0]);
    renderer->renderPose(&cameraPoseA[0],45,1,255,0,0,300.0f);
    renderer->renderPose(&cameraPoseB[0],45,1,0,255,0,300.0f);
    float cA[16];  transpose4x4(cameraPoseA,&cA[0]);
    float cB[16];  transpose4x4(cameraPoseB,&cB[0]);
    float cAi[16]; invertRT4(&cA[0],&cAi[0]);
    float dT[16]; matrixMult4x4(&cB[0],&cAi[0],&dT[0]);
    double angleDiff=0,posDiff=0;
    poseDistance(&dT[0],&posDiff,&angleDiff);
    glPopMatrix();

    char buf[512];
    sprintf(buf,"pos err: %3.3fmm",posDiff);
    renderer->drawText(x0,y0,width,height, buf,0.45f,0.9f-float(0)*0.1f,0.5f,0.5f,0.5f);
    sprintf(buf,"rot err: %3.3fdeg",angleDiff);
    renderer->drawText(x0,y0,width,height, buf,0.45f,0.9f-float(1)*0.1f,0.5f,0.5f,0.5f);
}

void setupCamera() {
    // view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    float baseModelViewT[16];
    memcpy(&baseModelView[0],&camTrack[currentFrame*16],sizeof(float)*16);
    transpose4x4(&baseModelView[0],&baseModelViewT[0]);
    // move camera
    cameraPosLag += (cameraPos - cameraPosLag) * inertia;
    cameraRotLag += (cameraRot - cameraRotLag) * inertia;
    cursorPosLag += (cursorPos - cursorPosLag) * inertia;
    glRotatef(cameraRotLag[0], 1.0, 0.0, 0.0);
    glRotatef(cameraRotLag[1], 0.0, 1.0, 0.0);
    glTranslatef(-cameraPosLag[0], -cameraPosLag[1], -cameraPosLag[2]);
    glMultMatrixf(&baseModelViewT[0]);

    float modelViewT[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewT); transpose4x4(modelViewT,modelView);
    nv::vec3f cameraDirLag; cameraDirLag.x = -modelView[8]; cameraDirLag.y = -modelView[9]; cameraDirLag.z = -modelView[10];
    renderer->setCameraMatrix(&modelView[0]);
}

// main rendering loop
void display()
{
    static bool firstTime = true;
    if (g_killThreads) return;
    initDisplayThread();
    if (firstTime) {
        resetVariables(currentFrame);
    }
    setupCamera();
    renderer->resetView3d(0,0,texWidth,texHeight);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    updateFrame();
    float poseA[16];  invertRT4(modelView,&poseA[0]);
    float poseAT[16]; transpose4x4(&poseA[0],&poseAT[0]);
    float poseBT[16]; identity4x4(&poseBT[0]);
    drawCameraEstimationView(&poseAT[0],&poseBT[0],winWidth/2,0,winWidth/2,winHeight);
    glutSwapBuffers();
    if (screenShot && saveNextFrame) screenShot->save();
    firstTime     = false;
    saveNextFrame = false;
    if (saving) { currentFrame++; saveNextFrame=true; if (currentFrame == numCameraPoses) { saving=false; resetVariables(0);}}
}

void initDisplayThread() {
    static int firstTime = true;
    if (firstTime) {
        renderer   = new SmokeRenderer();
        calibKinect.setupCalibDataBuffer(texWidth,texHeight);
        renderer->setProjectionMatrix(&calibKinect.getCalibData()[KL_OFFSET],texWidth,texHeight,0.01f, 1e5f);
        screenShot = new ScreenShot("scratch",0,0,winWidth/2,winHeight);
        // create opengl texture that will receive the result of CUDA
        createCudaTexture(&outputTexture, false, texWidth, texHeight,&cudaOutputTexture);
        createCudaTexture(&debugTexture,  false, texWidth, texHeight, &cudaDebugTexture);
        for (int i = 0; i < 4; i++) {
            createCudaTexture(&debugTexture1C[i], true, texWidth>>i, texHeight>>i, &cudaDebugTexture1C[i]);
        }
        // create texture for blitting onto the screen
        createTextureSrc(&inputTexture, texWidth,texHeight);
        createSensorTextures(&sensorRGBTex, &sensorDepthTex, texWidth, texHeight);
        // create a depth buffer for offscreen rendering
        createDepthBuffer(&depth_buffer, texWidth,texHeight);
        // create a framebuffer for offscreen rendering
        createFramebuffer(&framebuffer, inputTexture, depth_buffer);
        updateRGBDTextures();
        firstTime = false;
    }

}

bool setupCuda()  {
    int numCudaDevices = 0;
    cudaGetDeviceCount(&numCudaDevices);
    if (numCudaDevices == 0) { printf("no cuda devices found!\n"); return false; }
    printf("There are %d CUDA devices.\n", numCudaDevices);
    // initialize first GPU device to be CUDA interop device:
    checkCudaErrors(cudaGLSetGLDevice(0));
    return true;
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    char calibDir[512];
    char fileRGB[512];
    char fileDepth[512];
    char camPoses[512];
    if (argc < 5) {
        printf("no arguments!\n");
        printf("stereogen <calib.xml> <filergb.ppm> <filedepth.ppm> <cameraposes.txt>\n");
        return 1;
    }
    strcpy(calibDir, argv[1]);
    strcpy(fileRGB,  argv[2]);
    strcpy(fileDepth,argv[3]);
    strcpy(camPoses, argv[4]);

    camTrack = loadCameraMatrices(camPoses,&numCameraPoses);
    printf("%d camera poses found\n",numCameraPoses);

    printf("calib: %s\n",calibDir);
    printf("image: %s\n",fileRGB);
    printf("depthmap: %s\n",fileDepth);
    printf("cameraposes: %s\n",camPoses);

    // 1st initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is needed to achieve optimal performance with OpenGL/CUDA interop.
    initGL(&argc, argv);

    if (!setupCuda()) return 0;

    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(key);
    glutKeyboardUpFunc(keyUp);

    identity4x4(&modelView[0]);    
    // initialize calibration data:
    calibKinect.init(calibDir,true);
    calibKinect.setMinDist(500.0f);
    calibKinect.setMaxDist(8000.0f);
    calibKinect.setupCalibDataBuffer(texWidth,texHeight);

    fileSource   = new FileSource(fileRGB,fileDepth);
    preprocessor = new VideoPreProcessorCPU(fileSource,3,&calibKinect,texWidth,texHeight);
    preprocessor->pause();
    preprocessor->setFrame(0);
    while (!g_killThreads) {
        glutMainLoopEvent();
        idle();
        display();
    }
    release();
    return 1;
}

