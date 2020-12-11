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
// Smoke particle renderer with volumetric shadows

#ifndef SMOKE_RENDERER_H
#define SMOKE_RENDERER_H

#include <GL/glew.h>
#include "framebufferObject.h"
#include "GLSLProgram.h"
#include "nvMath.h"
//#include <helper_functions.h> // Helper functions (utilities, parsing, timing)
//using namespace nv;
#include <basic_math.h>
class SmokeRenderer
{
    public:
        SmokeRenderer();
        ~SmokeRenderer();

        void setCube(nv::vec3f origin, nv::vec3f dim) {
            m_cubeOrigin = origin;
            m_cubeDim    = dim;
        }

        void setCameraPos(float x, float y, float z) {
            m_cameraPos.x = x;
            m_cameraPos.y = y;
            m_cameraPos.z = z;
            printf("setPos: %f %f %f\n",m_cameraPos.x,m_cameraPos.y,m_cameraPos.z);
        }

        void setGridResolution(unsigned int x)
        {
            mNumVertices = x*x*x;
            mGridReso = x;
        }
        void setPositionBuffer(GLuint vbo)
        {
            mPosVbo = vbo;
        }
        void setCloudBuffer(GLuint vbo)
        {
            mCloudVbo = vbo;
        }
        void setPosition2dBuffer(GLuint vbo)
        {
            mPos2dVbo = vbo;
        }
        void setColorBuffer(GLuint vbo)
        {
            mColorVbo = vbo;
        }
        void setIndexBuffer(GLuint ib)
        {
            mIndexBuffer = ib;
        }

        void setPointSize(nv::vec3f sz)
        {
            mPointSize = sz.x;
        }
        void resetView3d(int x0, int y0, int w, int h);
        void resetView2d(int x0, int y0, int w, int h);
        void setCalib(float fov, float w, float h);


        void drawText(int x0, int y0, int width, int height, char *text,float xPos = -0.8f, float yPos = 0.9f, float r = 1.0f, float g = 1.0f, float b = 1.0f);
        void render(int offX, int offY, int viewportW, int viewportH, char *text = NULL);        
        void renderPose(float *cameraPose, float fovX, float aspectRatio,float r, float g, float b, float len);
        float *getK();
        void drawBox(nv::vec3f origin,nv::vec3f dim);
        void setCameraMatrix(float *modelView);
        float *getCameraMatrix();
        float *getPoseMatrix();
        void displayImage(GLuint texture, int x0, int y0, int width, int height, float r, float g, float b, float a, char *text=NULL);
        void displayRGBDImage(GLuint rgbTex, GLuint depthTex, int x0, int y0, int width, int height, float minDepth=0.0f, float maxDepth=10000.0f);
        void renderPointCloud(ProjectData *pointCloud, int texWidth, int texHeight, int offX, int offY, int viewportW, int viewportH, float minDist, float maxDist, GLuint rgbTex, char *text);
        void setProjectionMatrix(float *K, float texWidth, float texHeight, float nearZ, float farZ);
private:       
        void renderRays();
        void traverseVoxels(const nv::vec3f &origin,const nv::vec3f &rayDir, const nv::vec3f &invDir, int *sign, float enterT, float exitT, const nv::vec3f *bounds, const nv::vec3f &voxelDim);
        void calcRayDir(int xi, int yi, float *K, float *cameraPose, nv::vec3f &rayDir, nv::vec3f &invDir, int *sign);
        void displayTexture(GLuint tex);
        void compositeResult();
        void blurLightBuffer();
        void setPerspective(float fovY, float aspectratio, float znear, float zfar);

        GLuint createTexture(GLenum target, int w, int h, GLint internalformat, GLenum format);
        void createBuffers(int w, int h);
        void createLightBuffer();
        void drawQuad();
        float K[9],Knew[9]; float zNear,zFar;
        float cameraPose[16];
        float cameraTransform[16];
        float R[9];

        // particle data
        unsigned int        mNumVertices;
        unsigned int        mGridReso;

        GLuint              mPosVbo;
        GLuint              mCloudVbo;
        GLuint              mPos2dVbo;
        GLuint              mColorVbo;
        GLuint              mIndexBuffer;

        float               mPointSize;

        // window
        unsigned int        mWindowW, mWindowH;
        int                 m_downSample;
        int                 m_imageW, m_imageH;

        float               m_spriteAlpha;

        nv::vec3f               m_cubeOrigin,m_cubeDim;
        nv::vec3f               m_lightVector, m_lightPos, m_lightTarget;
        nv::vec3f               m_lightColor;
        nv::vec3f               m_colorAttenuation;
        nv::vec3f               m_cameraPos;
        float               m_lightDistance;

        nv::matrix4f            m_modelView, m_lightView, m_lightProj, m_shadowMatrix;
        nv::vec3f               m_viewVector, m_halfVector;
        bool                m_invertedView;
        nv::vec4f               m_eyePos;
        nv::vec4f               m_halfVectorEye;
        nv::vec4f               m_lightPosEye;

        // programs
        GLSLProgram         *m_particleProg;
        GLSLProgram         *m_depthProg;
        GLSLProgram         *m_simpleProg;
        GLSLProgram         *m_displayTexProg;
        GLSLProgram         *m_depth2DProg;
        GLSLProgram         *m_textureProg;
        GLSLProgram         *m_colorProg;


        GLuint floorTex;
        GLuint boxTex;

        GLuint              m_imageTex, m_depthTex;
        FramebufferObject   *m_imageFbo;
};

#endif
