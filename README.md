   RGB-D Tracking Library and Tools V0.1 (Draft)
   =============================================

This library contains contains main functionality for
RGB-D (Kinect) based real-time (dense) simultaneous
localisation and mapping (SLAM). Various visualisation
and reconstruction tools are included.

For detailed installation instructions and usage, see the
Wiki page: https://bitbucket.org/kamarain/rgbd-tracker/wiki/Home

Copyright: 2012-2014 Tommi Tykkala (ttykkala@gmail.com)

License:
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


I. Directory structure
----------------------
   - CMake     : cmake compilation helpers
   - ext       : directory to download external libs/tools
   - fonts     : freetype fonts used by the lib
   - sequences : place to store captured sequences (inc. one example seq)
   - shaders   : shaders used by the lib
   - src       : source code
   - textures  : textures used by the lib

II. Installation
----------------

See the Wiki: https://bitbucket.org/kamarain/rgbd-tracker/wiki/Home


III. Utilities and tools in realtime -directory
-----------------------------------------------

- kinectRecorder
	- for storing kinect sequences and keyframes
	- for running dense slam + visualization

NOTE: Binaries may require you to run them from the main directory!


IV. Sequences
-------------

When recording image data using kinectRecorder,
the target directory is always
	sequences/kinectScratch/<slotNumber>
where <slotNumber> is in range [0,9] and set by pressing [0..9]


**RGBD-TRACKER WIKI - **
This Wiki contains step by step instructions to use our code for RGBD-sensor based tracking and reconstruction.

[TOC]

# Introduction

This library contains main functionality for
RGB-D (Kinect) based real-time (dense) simultaneous
localisation and mapping (SLAM). Various visualisation
and reconstruction tools are included.

Please, if you use this software in your publications, cite our journal paper:

* Live RGB-D Camera Tracking for Television Production Studios (T. Tykkälä, A.I. Comport, J.-K. Kämäräinen, H. Hartikainen), In Journal of Visual Communication and Image Representation, volume 25, 2014. ( [bibtex](http://vision.cs.tut.fi/show_bib.php?key=TykComKam%3A2014&bib=data%2Fbibliography.bib),  [pdf](http://vision.cs.tut.fi/data/publications/JVCIR2014_accepted.pdf) )

More detailed information about the theory behind the method, you can find from [Tommi Tykkala's PhD thesis](http://vision.cs.tut.fi/data/publications/tykkala_phd_2013.pdf).

# License

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

# Installation and Requirements

You need the following components:

* CUDA 5.5 or later (NVidia installation script)
* OpenCV (tested with 2.4.4 local install)
* GLEW OpenGL extension (CUDA5.5 bundles this, also tested with 1.10.0 local install)
* TinyXML (tested with 2.6.2 local install)
* FreeNect (tested with 0.3.0 local install)
* [http://www.libsdl.org/](SDL) (tested with Ubuntu 12.04 default libsdl1.2-dev)
* [http://www.freetype.org/](FreeType) (tested with Ubuntu 12.04 default libfreetype6-dev)
* [http://libpng.org/](libPNG) (tested with Ubuntu 12.04 default libpng12-dev)
* [http://www.libusb.org/](libusb) (tested with Ubuntu 12.04 default libusb-1.0-0-dev)
* [http://www.opengl.org/](OpenGL) (comes with CUDA installation)
* [http://openmp.org](OpenMP) (comes with Ubuntu)

For all these components you should check which version is provided by your distribution directly
and if it is recent enough then just *sudo apt-get install <PACKAGE>*, but otherwise we have
preferred local manual installation, i.e. "make", but no "make install" and in the following we have collected some hopefully helpful hints.

## Ubuntu 12.04 with NVidia Graphics Card

### Necessary Ubuntu packages and Packages Available via Package Manager

```
$ sudo apt-get install cmake cmake-qt-gui cmake-curses-gui
$ sudo apt-get install libsdl1.2-dev
$ sudo apt-get install libfreetype6-dev
```

### CUDA and NVidia Drivers (inc. OpenGL)

These are needed in many algorithms in the code to achieve real-time performance (on a commodity laptop).

Web is full of dodgy installation instruction, but by far the best is [Installing CUDA on Ubuntu 12.04](http://sn0v.wordpress.com/2012/12/07/installing-cuda-5-on-ubuntu-12-04/). An error not mentioned in
the page is dependencies of the installed Ubuntu packages. If that happens, just add the dependencies
to the apt-get line and that's it - however don't boot at that point or you may loose your Desktop.

NOTE: It is preffered to use CUDA 5.5 version for this project.

### OpenCV

OpenCV library is fastly developing and thus always outdated in Ubuntu packages. Therefore, we recommend building it locally and not installing it system wide. Detailed description is given at the [opencv.org](http://opencv.org). Read Documentation -> Turorials -> Introduction -> Installation in Linux. You should just download the latest source package, put it somewhere (e.g. /home/<USER>/Work/external/) and then

```
$ cd <OPENCV_EXTRACTION_DIR>
$ mkdir build ; cd build
$ cmake ..
$ make 
```
NOTE: If there are errors in compilation related to .png versions, do following steps:
$ ccmake .. 
Keep the
BUILD_PNG         OFF (if it is ON) 
and WITH_PNG      OFF (if it is ON)
BUILD_SHARED_LIBS OFF

### GLEW

The project is CUDA 5.5 compatible which contains a version of GLEW too.
This version is used by the project for maximum compliancy. 
The latest version of GLEW is found at [glew.sourceforge.net](http://glew.sourceforge.net/).

### TinyXML

TinyXML source files are bundled in ext/tinyxml. A static library is built 
and linked to some of the applications. 

The latest version is found at [SourceForge](http://sourceforge.net/projects/tinyxml/),
but TinyXML does not currently install as Ubuntu package so it is included
as static library. The library is used by calibration module to store calibration
datas in xml format. TODO: replace xml by txt for avoiding this library completely. 

### FreeNect (OpenKinect)

This library provides tools to access your Kinect device. All software is provide by the [OpenKinect](http://openkinect.org) project. In particular, check *Install* and then *Ubuntu Manual Install*. Follow the steps (except "make install") and run the *bin/glview* to test that your system works ok.


### PoissonRecon - Polygon Surface Reconstruction 

You need to download the source code from [http://www.cs.jhu.edu/~misha/Code/PoissonRecon](http://www.cs.jhu.edu/~misha/Code/PoissonRecon) and put it rgbd-tracker/src/external/ or use our installation script:
```
$ cd ext/
$ source get_PoissonRecon.sh
```

### Compiling the Project

In the following, we assume that you have fetched the project to <RGBD-TRACKER_DIR> and you a
local installation of OpenCV in <OPENCV_DIR>/build/ and GLEW in <GLEW_DIR> and TinyXML in <TINYXML_DIR>
(note that our cmake assumes full path names and trailing slash at the end):

```
$ cd <RGBD-TRACKER_DIR>/realtime
$ mkdir build
$ cd build
$ cmake -DOpenCV_DIR=<OPENCV_DIR>/build/ -DGLEW_DIR=<GLEW_DIR>/ -DTINYXML_DIR=<TINYXML_DIR>..
$ make
```

Example true cmake command:
```
$  cmake  -DOpenCV_DIR=/home/kamarain/Work/ext/opencv-2.4.4/build/ -DGLEW_DIR=/home/kamarain/Work/ext/glew-1.10.0 -DTINYXML_DIR=/home/kamarain/Work/ext/tinyxml/ -DFreeNect_DIR=/home/kamarain/Work/ext/libfreenect/ ..
```

# Running Test Tracker (bin/kinectRecorder)

Note that the current test tracker is very picky since many paths are hard coded (we will change this to utilise Boost command line arguments soon). Moreover, the GUI contains magic key bindings to run the code with different parameters and inputs and to demonstrate augmented graphics.

Key  Action

-----------------------

1) "ESC" quit 

2) "t" toggle Kinect input

3) "a" start tracking with pre-recorded video  

4) "k" store one keyframe 

5) "r" record sequence (to memory) 

6) "l" store recorded sequence: before you need to  use "r" to record a sequence, then select a "folder name" eg. "1..9" prior to click "l". 

7) "p" pause 

8) "q" augmented a teapot

9) "c" reset the data

10) Left_SHIFT + "s" GPU-based tracking

-----------------------

Note that recording stores the sequence under sequences/kinectScratch/<slotNumber> where the
slotNumbers is between 0-9.

## Offline (with pre-recorded sequences)

When your system compiles, you may run the test tracker GUI from the main directory:
```
$ ./build/bin/kinectRecorder
```
By default this opens the GUI shown below and reads a pre-recorded sequence from *sequences/kinectScratch/0/*. The calibration information for the Kinect used to capture the sample sequence is loaded from *sequences/scratch_0_calib/* (symbolic link set from *0/*). This is a short sequence, but
you can test that your system works and the tracker runs (press 'a' to start tracking). 

![kinectRecorder_gui_web.png](https://bitbucket.org/repo/jxqz5n/images/4063631907-kinectRecorder_gui_web.png)

# RGB-D sensor (Kinect) Calibration

You need to calibrate the intrinsic camera parameters of your Kinect device. Camera calibration ready-made tool is available at [CalibrationTool](http://www.ee.oulu.fi/~dherrera/kinect/)

# stereogen: Single RGB-D to Multi-view Stereo Images

Directory src/tools contains stereogen program for generating multi-view stereo images
from a single RGB-D measurement. 

Compilation:
cd build; ccmake ..; <configure with c, generate with g>; make 

Example execution:
stereogen example/calib.xml example/lanxurgb.ppm example/lanxudepth.ppm example/cameraposes.txt

The program takes calibration information, one RGB image and one disparity map as input.
The formats are compatible with kinectRecorder. cameraposes.txt is a list of 4x4 matrices
which determine viewpoints that are generated to scratch/

# Useful links

* [RGBDToolkit](http://www.rgbdtoolkit.com/) - Open source for making 3D videos
* [RGB+D sensor comparison](http://wiki.ipisoft.com/Depth_Sensors_Comparison) - See what's the best device
* [OpenNI](http://www.openni.org/) - Open source SDK for the development of 3D sensing middleware libraries and applications
