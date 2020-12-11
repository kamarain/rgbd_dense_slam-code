/*
stereo-gen
Copyright (c) 2014, Tommi Tykkälä, All rights reserved.

This source code is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3.0 of the License, or (at your option) any later version.

This source code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this source code.
*/

#define STRINGIFY(A) #A

// floor shader
const char *floorVS = STRINGIFY(
            varying vec4 vertexPosEye;  // vertex position in eye space  \n
void main()                                                  \n
{
    \n
             gl_Position    = gl_ModelViewProjectionMatrix*vec4(gl_Vertex.xyz,1);  \n
             gl_TexCoord[0] = gl_MultiTexCoord0;                      \n
             vertexPosEye   = gl_ModelViewMatrix *vec4(gl_Vertex.xyz,1);           \n
//             gl_FrontColor = gl_Color;                                \n
}                                                            \n
);

const char *floorPS = STRINGIFY(
            uniform sampler2D tex; \n
            uniform float minDepth; \n
            uniform float maxDepth; \n
            varying vec4 vertexPosEye;                                                    \n
            void main() {                                                                 \n
                vec4 colorMap  = texture2D(tex, gl_TexCoord[0].xy);                       \n
                gl_FragColor.xyz = colorMap.xyz;                                          \n
                gl_FragColor.w = clamp((-vertexPosEye.z-minDepth)/(maxDepth-minDepth),0,1.0f); \n
            }                                                                             \n
);
