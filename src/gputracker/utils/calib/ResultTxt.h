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

#define MAX_POSE_COUNT 10000

class ResultTxt {
public:
	ResultTxt();
	~ResultTxt();
        void reset(bool addIdentityFlag=false);
        void init(const char *fn,bool addIdentityFlag=false);
        void save(int transpose=0, int inverse=0);
        void addPose(float *m4x4);
        void canonize();
private:
    int nRows;
    float *data;
    char fn[512];
};
