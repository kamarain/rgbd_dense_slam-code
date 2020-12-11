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

enum TrackingMode {
    INCREMENTAL, KEYFRAME, HYBRID
};

namespace gputracker {
    int initialize(const char *xmlConfigFile);
    void set_calibration(const char *xmlCalibFileName);
    void set_source(VideoSource *stream);
    void set_selected_points(int nPoints);
    int get_selected_points();
    void set_camera_tracking(bool flag);
    bool playing();
    int track_frame();
    void fill_rgbtex(unsigned int texID);
    void fill_depthtex(unsigned int depthID);
    float get_fov_x();
    float get_fov_y();
    void release();
    void reset();
    void set_depth_filtering(bool mode);
    void set_keyframe_mode();
    void set_incremental_mode();
    int get_update_freq();
    void set_hybrid_mode();
    void set_keyframe_model(const char *keyFrameModelPath);
    int get_num_est_poses();
    int get_max_est_poses();
    int get_keyframe_count();
    int get_max_residual_length();
    int get_mode();
    // debug only:
    void render_rgb_tex(float z=-1.0f, bool overlayPoints=false);
    void render_icp_ref_tex(float z = -1.0f);
    void render_ref_points();
    void render_depth(float z = -1.0f, int layer=0);
    void render_trimesh(float *clightPos); // render a rgb-d frame using triangle mesh. good for validating reconstruction quality
    //void set_estimation(bool mode);
    void get_pose(float *poseMatrix, float *icpPose=NULL);
    void render_vertices(bool allvertices);
    void render_keyframes();
    void render_keys();
    void render_base();
    void render_rgbd();
    void render_active_key();
    int get_frame_index();
    int get_free_gpu_memory();
    void set_blank_images(bool flag);
    void get_first_plane(float *mean, float *normal);    
}
