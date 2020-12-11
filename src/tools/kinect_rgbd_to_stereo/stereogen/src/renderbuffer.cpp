/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include "renderbuffer.h"
#include <iostream>
using namespace std;

Renderbuffer::Renderbuffer()
    : m_bufId(_CreateBufferId())
{}

Renderbuffer::Renderbuffer(GLenum internalFormat, int width, int height)
    : m_bufId(_CreateBufferId())
{
    Set(internalFormat, width, height);
}

Renderbuffer::~Renderbuffer()
{
    glDeleteRenderbuffersEXT(1, &m_bufId);
}

void Renderbuffer::Bind()
{
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, m_bufId);
}

void Renderbuffer::Unbind()
{
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
}

void Renderbuffer::Set(GLenum internalFormat, int width, int height)
{
    int maxSize = Renderbuffer::GetMaxSize();

    if (width > maxSize || height > maxSize)
    {
        cerr << "Renderbuffer::Renderbuffer() ERROR:\n\t"
             << "Size too big (" << width << ", " << height << ")\n";
        return;
    }

    // Guarded bind
    GLint savedId = 0;
    glGetIntegerv(GL_RENDERBUFFER_BINDING_EXT, &savedId);

    if (savedId != (GLint)m_bufId)
    {
        Bind();
    }

    // Allocate memory for renderBuffer
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, internalFormat, width, height);

    // Guarded unbind
    if (savedId != (GLint)m_bufId)
    {
        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, savedId);
    }
}

GLuint Renderbuffer::GetId() const
{
    return m_bufId;
}

GLint Renderbuffer::GetMaxSize()
{
    GLint maxAttach = 0;
    glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE_EXT, &maxAttach);
    return maxAttach;
}

GLuint Renderbuffer::_CreateBufferId()
{
    GLuint id = 0;
    glGenRenderbuffersEXT(1, &id);
    return id;
}

