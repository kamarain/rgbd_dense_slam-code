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

#ifndef SHADER_H
#define SHADER_H

#include <GL/glew.h> // GLEW Library
#include <GL/gl.h>	// OpenGL32 Library
#include <GL/glu.h>	// GLU32 Library
#include <fstream>

using namespace std;

class Shader {
public:
    Shader(const char *vsFilename, const char *fsFilename);
    void bind();
    void unbind();
    void release();
    int getAttrib(const char *name);
    void setUniformVec4(const char *name, const float *vec4);
    ~Shader();
private:
    GLhandleARB vs, fs, program; // handles to objects
    char *loadSource(const char* filename);
    int unloadSource(GLcharARB** ShaderSource);
};


#endif // SHADER_H
