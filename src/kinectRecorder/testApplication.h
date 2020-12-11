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

#include <string>
#include <vector>
#include <image2/Image2.h>
using namespace std;

struct TrueTypeText;
struct SDL_Surface;

class OrbitingCamera;
class GLWindow;

class TestApplication {
public:
	TestApplication();
	~TestApplication();
    int init(int argc, char **argv, const std::string &name, int resx, int resy, int nCol, int nRow);
	int run(int fps=30);
	void renderScene();
private:
	void initGL(int width, int height);
	void shutDown();
	void updateCalib(double *kcR,float kcPhase1, float kcPhase2, float kcAmplitude, double viewDistanceMin, double viewDistanceMax);
	int resx,resy;
	int nCol,nRow;
	SDL_Surface *sdlScreen;
	std::vector<GLWindow *> glWindows;
    //VideoPreProcessor *videoPreprocessor;
	void handleKeyDown(int key, int &done);
	void handleKeyUp(int key, int &done);
	void handlePressedKeys();
	void setupVideoStream(int index);
    void saveScreenShot();
    void saveScreenShot(unsigned char *index, int nIndices);
    unsigned char *screenShot;
};
