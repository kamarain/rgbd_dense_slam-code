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
#include <stdlib.h>
#include "SmokeRenderer.h"
#include "SmokeShaders.h"
#include <GL/freeglut.h>
#include <time.h>
#include <timer.h>
#include <string.h>
using namespace nv;
//#define GRID_RESO 32


#define MIN(x,y) ( x < y ? x : y)
#define MAX(x,y) ( x > y ? x : y)

GLuint createTexture(GLenum target, GLint internalformat, GLenum format, int w, int h, void *data)
{
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(target, tex);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(target, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_UNSIGNED_BYTE, data);
    return tex;
}

SmokeRenderer::SmokeRenderer() :
    mNumVertices(0),
    mGridReso(0),
    mPosVbo(0),
    mPos2dVbo(0),
    mColorVbo(0),
    mIndexBuffer(0),
    mPointSize(0.005f),
    mWindowW(0),
    mWindowH(0),
    m_downSample(2),
    m_spriteAlpha(0.1f),
    m_lightPos(5.0f, 5.0f, -5.0f),
    m_lightTarget(0.0f, 0.0f, 0.0f),
    m_lightColor(1.0f, 1.0f, 0.5f),
    m_colorAttenuation(0.1f, 0.2f, 0.3f),
    m_cameraPos(0,0,0),
    m_imageTex(0),
    m_depthTex(0),
    m_imageFbo(0)
{
    identity3x3(&K[0]);
    identity3x3(&Knew[0]);
    identity4x4(&cameraTransform[0]);
    identity4x4(&cameraPose[0]);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  //  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16.0f);

    m_textureProg = new GLSLProgram(floorVS, floorPS);


    glutReportErrors();
}


SmokeRenderer::~SmokeRenderer()
{   
    delete m_textureProg;
    glDeleteTextures(1, &m_imageTex);
    glDeleteTextures(1, &m_depthTex);
}
// display texture to screen
void SmokeRenderer::displayTexture(GLuint tex)
{
    m_displayTexProg->enable();
    m_displayTexProg->bindTexture("tex", tex, GL_TEXTURE_2D, 0);
    drawQuad();
    m_displayTexProg->disable();
}

void displayText( float x, float y, float r, float g, float b, const char *string ) {
    glPushMatrix();
    glLoadIdentity();
    glEnable(GL_COLOR_MATERIAL);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_2D);
    int j = strlen( string );
    glColor4f( r, g, b, 1.0f );
    glRasterPos2f( x, y );
    for( int i = 0; i < j; i++ ) {
        glutBitmapCharacter( GLUT_BITMAP_TIMES_ROMAN_24, string[i] );
    }
    glPopMatrix();
    glEnable(GL_DEPTH_TEST);
}

void SmokeRenderer::drawText(int x0, int y0, int width, int height, char *text, float xPos, float yPos, float r, float g, float b) {
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_LIGHTING);
    glDisable(GL_ALPHA_TEST);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1e8f);
    glViewport(x0, y0, width,height);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glPopMatrix();

    if (text != NULL) {
        displayText( xPos, yPos, r,g,b, text);
    }
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
}

// display image to the screen as textured quad
void SmokeRenderer::displayImage(GLuint texture, int x0, int y0, int width, int height, float r, float g, float b, float a, char *text)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_LIGHTING);
    glEnable(GL_ALPHA_TEST);

    //glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1e8f);
    glViewport(x0, y0, width,height);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    //if (flip)
    glScalef(1,-1,1);

    glEnable(GL_COLOR_MATERIAL);
    glColor4f(r,g,b,a);

    float t = 1.0f;
    float s = 1.0f;
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex3f(-s, -s, 0.5);
    glTexCoord2f(t, 0.0);
    glVertex3f(s, -s, 0.5);
    glTexCoord2f(t, t);
    glVertex3f(s, s, 0.5);
    glTexCoord2f(0.0, t);
    glVertex3f(-s, s, 0.5);
    glEnd();

    glPopMatrix();
    if (text != NULL) {
       // resetView2d(x0,y0,width,height);
        displayText( -0.8, 0.9f, 1.0f,1.0f,1.0f, text);
    }

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
}

void SmokeRenderer::displayRGBDImage(GLuint rgbTex, GLuint depthTex, int x0, int y0, int width, int height, float minDepth, float maxDepth)
{
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_LIGHTING);
    glDisable(GL_ALPHA_TEST);
    //glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1e8f);
    glViewport(x0, y0, width,height);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    m_depth2DProg->enable();
    m_depth2DProg->bindTexture("rgbTex", rgbTex, GL_TEXTURE_2D, 0);
    m_depth2DProg->bindTexture("depthTex", depthTex, GL_TEXTURE_2D, 1);
    m_depth2DProg->setUniform1f("minDepth", minDepth);
    m_depth2DProg->setUniform1f("maxDepth", maxDepth);
    m_depth2DProg->bindTexture("depthTex", depthTex, GL_TEXTURE_2D, 1);

    float s = 1.0f;
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 1.0f);
    glVertex3f(-s, -s, 0.5);
    glTexCoord2f(1.0f, 1.0);
    glVertex3f(s, -s, 0.5);
    glTexCoord2f(1.0f, 0);
    glVertex3f(s, s, 0.5);
    glTexCoord2f(0.0, 0.0f);
    glVertex3f(-s, s, 0.5);
    glEnd();

    m_depth2DProg->disable();

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
}

void SmokeRenderer::setCameraMatrix(float *modelView) {
    memcpy(&cameraTransform[0],modelView,sizeof(float)*16);
    invertRT4(modelView,&cameraPose[0]);
    copy3x3(&cameraPose[0],&R[0]);
    m_cameraPos.x = cameraPose[3];
    m_cameraPos.y = cameraPose[7];
    m_cameraPos.z = cameraPose[11];
}

float *SmokeRenderer::getCameraMatrix() {
    return &cameraTransform[0];
}

float *SmokeRenderer::getPoseMatrix() {
    return &cameraPose[0];
}

void SmokeRenderer::render(int offX, int offY, int viewportW, int viewportH, char *text)
{
    resetView3d(offX,offY,viewportW,viewportH);
    glColor3f(1.0, 0.0, 0.0);

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glColor4f(1,1,1,1);
    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    drawBox(m_cubeOrigin,m_cubeDim);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);

    if (text != NULL) {
        resetView2d(offX,offY,viewportW,viewportH);
        displayText( viewportW*0.1f,viewportH*0.9f, 1.0f,1.0f,1.0f, text);
    }

    glEnable(GL_LIGHTING);
    resetView3d(0,0,mWindowW,mWindowH);
    glutReportErrors();
}


void SmokeRenderer::renderPointCloud(ProjectData *pointCloud, int texWidth, int texHeight, int offX, int offY, int viewportW, int viewportH, float minDist, float maxDist, GLuint rgbTex, char *text)
{
    float w = float(texWidth); float h = float(texHeight);
    resetView3d(offX,offY,viewportW,viewportH);
    glColor3f(1.0, 0.0, 0.0);

    glPushMatrix();
    float mtx[16]; transpose4x4(&cameraTransform[0],&mtx[0]);
    glLoadMatrixf(&mtx[0]);
    glEnable(GL_TEXTURE_2D);
 //   glDisable(GL_TEXTURE_2D);
    glDisable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
    glColor4f(1,1,1,1);
//    glEnable(GL_BLEND);
//    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);



    m_textureProg->enable();
    m_textureProg->setUniform1f("minDepth", minDist);
    m_textureProg->setUniform1f("maxDepth", maxDist);
    m_textureProg->bindTexture("tex", rgbTex, GL_TEXTURE_2D, 0);

    glBegin(GL_QUADS);
    for (int j = 1; j < (texHeight-1); j++) {
        for (int i = 1; i < (texWidth-1); i++) {
            int offset = i+j*texWidth;
            ProjectData &p0 = pointCloud[offset]; //float n0 = randGrid[offset];
            ProjectData &p1 = pointCloud[offset+texWidth]; //float n1 = randGrid[offset+texWidth];
            ProjectData &p2 = pointCloud[offset+1+texWidth]; //float n2 = randGrid[offset+1+texWidth];
            ProjectData &p3 = pointCloud[offset+1]; //float n3 = randGrid[offset+1];

            if (p0.magGrad < 0) continue;
            if (p1.magGrad < 0) continue;
            if (p2.magGrad < 0) continue;
            if (p3.magGrad < 0) continue;

            float minZ = MIN(MIN(MIN(p0.rz,p1.rz),p2.rz),p3.rz);
            float maxZ = MAX(MAX(MAX(p0.rz,p1.rz),p2.rz),p3.rz);

            if (maxZ >= -300.0f || fabs(maxZ-minZ) > 150.0f) continue;

            float alpha = 1.0f;//MAX(1-fabs(maxZ-minZ)/10.0f,0);

//            glColor4f(p0.colorR,p0.colorG,p0.colorB,alpha);
            glTexCoord2f(p0.px/w,p0.py/h);
            glVertex3f(p0.rx,p0.ry,p0.rz);

//            glColor4f(p1.colorR,p1.colorG,p1.colorB,alpha);
            glTexCoord2f(p1.px/w,p1.py/h);
            glVertex3f(p1.rx,p1.ry,p1.rz);

//            glColor4f(p2.colorR,p2.colorG,p2.colorB,alpha);
            glTexCoord2f(p2.px/w,p2.py/h);
            glVertex3f(p2.rx,p2.ry,p2.rz);

//            glColor4f(p3.colorR,p3.colorG,p3.colorB,alpha);
            glTexCoord2f(p3.px/w,p3.py/h);
            glVertex3f(p3.rx,p3.ry,p3.rz);
        }
    }
    glEnd();

    m_textureProg->disable();
    glPopMatrix();
  //  glDisable(GL_BLEND);
    if (text != NULL) {
        resetView2d(offX,offY,viewportW,viewportH);
        displayText( viewportW*0.01f,viewportH*0.95f, 1.0f,1.0f,1.0f, text);
    }
    glutReportErrors();
}
// create an OpenGL texture
GLuint SmokeRenderer::createTexture(GLenum target, int w, int h, GLint internalformat, GLenum format)
{
    GLuint texid;
    glGenTextures(1, &texid);
    glBindTexture(target, texid);

    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(target, 0, internalformat, w, h, 0, format, GL_FLOAT, 0);
    return texid;
}

// create buffers for off-screen rendering
void SmokeRenderer::createBuffers(int w, int h)
{
    if (w == mWindowW && h == mWindowH) return;

    if (m_imageFbo)
    {
        glDeleteTextures(1, &m_imageTex);
        glDeleteTextures(1, &m_depthTex);
        delete m_imageFbo;
    }

    // create fbo for image buffer
    GLint format = GL_RGBA16F_ARB;
    //GLint format = GL_LUMINANCE16F_ARB;
    //GLint format = GL_RGBA8;
    m_imageTex = createTexture(GL_TEXTURE_2D, m_imageW, m_imageH, format, GL_RGBA);
    m_depthTex = createTexture(GL_TEXTURE_2D, m_imageW, m_imageH, GL_DEPTH_COMPONENT24_ARB, GL_DEPTH_COMPONENT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    m_imageFbo = new FramebufferObject();
    m_imageFbo->AttachTexture(GL_TEXTURE_2D, m_imageTex, GL_COLOR_ATTACHMENT0_EXT);
    m_imageFbo->AttachTexture(GL_TEXTURE_2D, m_depthTex, GL_DEPTH_ATTACHMENT_EXT);
    m_imageFbo->IsValid();
}

void SmokeRenderer::setProjectionMatrix(float *Kin, float texWidth, float texHeight, float nearZ, float farZ) {
    float P[16],PT[16];
    memcpy(this->Knew,Kin,sizeof(float)*9);
    this->zNear    = nearZ;
    this->zFar     = farZ;
    this->mWindowW = texWidth;
    this->mWindowH = texHeight;
   // dumpMatrix("Kin",Kin,3,3);
    P[0]  = -2*Kin[0]/texWidth;  P[1]  = 0;                   P[2]  = 1-(2*Kin[2]/texWidth);        P[3] = 0;
    P[4]  = 0.0f;                P[5]  = 2*Kin[4]/texHeight;  P[6]  = -1+(2*Kin[5]+2)/texHeight;    P[7] = 0;
    P[8]  = 0.0f;                P[9]  = 0.0f;                P[10] = (farZ+nearZ)/(nearZ-farZ);    P[11] = 2*nearZ*farZ/(nearZ-farZ);
    P[12] = 0.0f;                P[13] = 0.0f;                P[14] = -1;                           P[15] = 0;
    transpose4x4(&P[0],&PT[0]);
   // dumpMatrix("P",P,4,4);
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(&PT[0]);
    glMatrixMode(GL_MODELVIEW);
}


void SmokeRenderer::resetView3d(int x0, int y0, int w, int h) {
    createBuffers(w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    setProjectionMatrix(Knew,mWindowW,mWindowH,zNear,zFar);
    glMatrixMode(GL_MODELVIEW);
    glViewport(x0, y0, w, h);
}

void SmokeRenderer::resetView2d(int x0, int y0, int w, int h) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0,w,0,h,0.1f,1e9f);
    glMatrixMode(GL_MODELVIEW);
    glViewport(x0,y0,w,h);
}

float *SmokeRenderer::getK() {
    return &Knew[0];
}


void SmokeRenderer::drawQuad()
{
    float s=1.0f;
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(-s, -s);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(s, -s);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(s,  s);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(-s,  s);
    glEnd();
}

void SmokeRenderer::drawBox(vec3f origin, vec3f dim)
{
    float hx = dim.x+1e-4f;
    float hy = dim.y+1e-4f;
    float hz = dim.z+1e-4f;
    float x0 = origin.x;
    float y0 = origin.y;
    float z0 = origin.z;

    glLineWidth(2.0f);

    glBegin(GL_QUADS);
    // Front Face
    glTexCoord2f(0.0f, 0.0f); glVertex3f(-hx+x0, -hy+y0,  hz+z0);  // Bottom Left Of The Texture and Quad
    glTexCoord2f(1.0f, 0.0f); glVertex3f( hx+x0, -hy+y0,  hz+z0);  // Bottom Right Of The Texture and Quad
    glTexCoord2f(1.0f, 1.0f); glVertex3f( hx+x0,  hy+y0,  hz+z0);  // Top Right Of The Texture and Quad
    glTexCoord2f(0.0f, 1.0f); glVertex3f(-hx+x0,  hy+y0,  hz+z0);  // Top Left Of The Texture and Quad
    // Back Face
    glTexCoord2f(1.0f, 0.0f); glVertex3f(-hx+x0, -hy+y0, -hz+z0);  // Bottom Right Of The Texture and Quad
    glTexCoord2f(1.0f, 1.0f); glVertex3f(-hx+x0,  hy+y0, -hz+z0);  // Top Right Of The Texture and Quad
    glTexCoord2f(0.0f, 1.0f); glVertex3f( hx+x0,  hy+y0, -hz+z0);  // Top Left Of The Texture and Quad
    glTexCoord2f(0.0f, 0.0f); glVertex3f( hx+x0, -hy+y0, -hz+z0);  // Bottom Left Of The Texture and Quad
    // Top Face
    glTexCoord2f(0.0f, 1.0f); glVertex3f(-hx+x0,  hy+y0, -hz+z0);  // Top Left Of The Texture and Quad
    glTexCoord2f(0.0f, 0.0f); glVertex3f(-hx+x0,  hy+y0,  hz+z0);  // Bottom Left Of The Texture and Quad
    glTexCoord2f(1.0f, 0.0f); glVertex3f( hx+x0,  hy+y0,  hz+z0);  // Bottom Right Of The Texture and Quad
    glTexCoord2f(1.0f, 1.0f); glVertex3f( hx+x0,  hy+y0, -hz+z0);  // Top Right Of The Texture and Quad
    // Bottom Face
    glTexCoord2f(1.0f, 1.0f); glVertex3f(-hx+x0, -hy+y0, -hz+z0);  // Top Right Of The Texture and Quad
    glTexCoord2f(0.0f, 1.0f); glVertex3f( hx+x0, -hy+y0, -hz+z0);  // Top Left Of The Texture and Quad
    glTexCoord2f(0.0f, 0.0f); glVertex3f( hx+x0, -hy+y0,  hz+z0);  // Bottom Left Of The Texture and Quad
    glTexCoord2f(1.0f, 0.0f); glVertex3f(-hx+x0, -hy+y0,  hz+z0);  // Bottom Right Of The Texture and Quad
    // Right face
    glTexCoord2f(1.0f, 0.0f); glVertex3f( hx+x0, -hy+y0, -hz+z0);  // Bottom Right Of The Texture and Quad
    glTexCoord2f(1.0f, 1.0f); glVertex3f( hx+x0,  hy+y0, -hz+z0);  // Top Right Of The Texture and Quad
    glTexCoord2f(0.0f, 1.0f); glVertex3f( hx+x0,  hy+y0,  hz+z0);  // Top Left Of The Texture and Quad
    glTexCoord2f(0.0f, 0.0f); glVertex3f( hx+x0, -hy+y0,  hz+z0);  // Bottom Left Of The Texture and Quad
    // Left Face
    glTexCoord2f(0.0f, 0.0f); glVertex3f(-hx+x0, -hy+y0, -hz+z0);  // Bottom Left Of The Texture and Quad
    glTexCoord2f(1.0f, 0.0f); glVertex3f(-hx+x0, -hy+y0,  hz+z0);  // Bottom Right Of The Texture and Quad
    glTexCoord2f(1.0f, 1.0f); glVertex3f(-hx+x0,  hy+y0,  hz+z0);  // Top Right Of The Texture and Quad
    glTexCoord2f(0.0f, 1.0f); glVertex3f(-hx+x0,  hy+y0, -hz+z0);  // Top Left Of The Texture and Quad
    glEnd();
}

void SmokeRenderer::renderPose(float *T, float fovY, float aspect, float r, float g, float b, float len) {
    glPushAttrib(GL_LIST_BIT | GL_CURRENT_BIT | GL_ENABLE_BIT | GL_TRANSFORM_BIT | GL_LIGHTING_BIT | GL_TEXTURE_BIT | GL_DEPTH_BUFFER_BIT);
    glLineWidth(2.0f);
    glDisable(GL_TEXTURE_2D);
    glBegin(GL_LINES);
//    printf("aspect:%f\n",aspect);
    float o[3]; o[0] =      T[12]; o[1] =     T[13]; o[2] =      T[14];
    float z[3]; z[0] =  -len*T[8]; z[1] = -len*T[9]; z[2] = -len*T[10];
    float sX = aspect*tan(3.141592653f*fovY/360.0f)*len;
    float sY = tan(3.141592653f*fovY/360.0f)*len;

    float u[3]; u[0] = sX*T[0]; u[1] = sX*T[1]; u[2] = sX*T[2];
    float v[3]; v[0] = sY*T[4]; v[1] = sY*T[5]; v[2] = sY*T[6];

    float x0[3]; x0[0] = o[0]+z[0]-u[0]-v[0]; x0[1] = o[1]+z[1]-u[1]-v[1]; x0[2] = o[2]+z[2]-u[2]-v[2];
    float x1[3]; x1[0] = o[0]+z[0]+u[0]-v[0]; x1[1] = o[1]+z[1]+u[1]-v[1]; x1[2] = o[2]+z[2]+u[2]-v[2];
    float x2[3]; x2[0] = o[0]+z[0]+u[0]+v[0]; x2[1] = o[1]+z[1]+u[1]+v[1]; x2[2] = o[2]+z[2]+u[2]+v[2];
    float x3[3]; x3[0] = o[0]+z[0]-u[0]+v[0]; x3[1] = o[1]+z[1]-u[1]+v[1]; x3[2] = o[2]+z[2]-u[2]+v[2];

    glColor3f(r,g,b);

    glVertex3f(x0[0],x0[1],x0[2]); glVertex3f(x1[0],x1[1],x1[2]);
    glVertex3f(x1[0],x1[1],x1[2]); glVertex3f(x2[0],x2[1],x2[2]);
    glVertex3f(x2[0],x2[1],x2[2]); glVertex3f(x3[0],x3[1],x3[2]);
    glVertex3f(x3[0],x3[1],x3[2]); glVertex3f(x0[0],x0[1],x0[2]);

    glVertex3f(o[0],o[1],o[2]); glVertex3f(x0[0],x0[1],x0[2]);
    glVertex3f(o[0],o[1],o[2]); glVertex3f(x1[0],x1[1],x1[2]);
    glVertex3f(o[0],o[1],o[2]); glVertex3f(x2[0],x2[1],x2[2]);
    glVertex3f(o[0],o[1],o[2]); glVertex3f(x3[0],x3[1],x3[2]);

    glEnd();
    glPopAttrib();

}
