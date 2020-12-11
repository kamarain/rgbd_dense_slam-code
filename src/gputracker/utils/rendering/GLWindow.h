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

#include "camera/Camera.hpp"

class GLWindow {
public:
	GLWindow(int x0, int y0, int w, int h, void (*renderFunc)());
    void setCamera(customCameraTools::Camera *camera);
	~GLWindow();
	void render();
private:
	int x0,y0,w,h;
	void (*renderFunc)();
    customCameraTools::Camera *camera;
};
