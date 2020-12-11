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
