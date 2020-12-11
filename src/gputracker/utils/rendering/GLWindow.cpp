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

#include <GL/glew.h> // GLEW Library
#include "GLWindow.h"
#include <camera/Camera.hpp>

using namespace customCameraTools;

GLWindow::GLWindow( int x0, int y0, int w, int h,	void (*renderFunc)())
{
	this->x0 = x0;
	this->y0 = y0;
	this->w = w;
	this->h = h;
	this->camera = NULL;
	this->renderFunc = renderFunc;
}

GLWindow::~GLWindow()
{

}

void GLWindow::render() {
	if (camera == NULL) {
		glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-1.0f,1.0f,-1.0f,1.0f, 0.1f, 20000.0f);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glScalef(1,-1,1);
	} else {
		camera->activate();
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(camera->getProjectionMatrix());
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(camera->getModelViewMatrix());
	}
	glViewport(x0, y0, w, h);

	renderFunc();
}

void GLWindow::setCamera( Camera *camera )
{
	this->camera = camera;
}
