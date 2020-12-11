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

#include "VertexBuffer2.h"

/*
BASEBUFFER VERTEX FORMAT (scratch bufferi, 2d overlay-graffaa varten kuten basebuffer + 3d trajectory)
0 x
1 y
2 z
3 r
4 g
5 b
*/

class BaseBuffer2 {
private:
    VertexBuffer2 *vbuffer;
    float T[16];
    float invT[16];
public:
      BaseBuffer2();
     ~BaseBuffer2();
     void release();
     void reset();
     void initialize();
     void renderBase();
     void renderSrcPoints(int cnt);
     void renderDstPoints(int cnt);
     VertexBuffer2 *getVBuffer() { return vbuffer; }
     float *getCurrentPose();
     void downloadBaseCPU(float *devT);
};
