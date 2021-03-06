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

#include "testApplication.h"

const int resx = 640;
const int resy = resx*3/4;

int main(int argc, char **argv) 
{  
	TestApplication testApp;

    testApp.init(argc, argv, "cudaTester",resx,resy,3,2);
	testApp.run(30);
	return 1;
}
