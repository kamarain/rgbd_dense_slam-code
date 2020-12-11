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

#include <capture/videoSource.h>

namespace cputracker {
    int initialize(const char *xmlConfigFile);
    void set_calibration(const char *xmlCalibFileName);
    void set_source(VideoSource *stream);
    void set_selected_points(int nPoints);
    int get_selected_points();
    void set_camera_tracking(bool flag);
    bool playing();
    int track_frame();
    float get_fov_x();
    float get_fov_y();
    void release();
    void reset();
    void set_camera_tracking(bool flag);
    void get_pose(float *poseMatrixDst);
    int get_update_freq();
    void set_estimation(bool mode);
    void fill_rgbtex(unsigned int texID);
    void fill_depthtex(unsigned int depthID);
    unsigned char *get_rgb_ptr();
    float *get_depth_ptr();
    void get_first_plane(float *mean, float *normal);
}
