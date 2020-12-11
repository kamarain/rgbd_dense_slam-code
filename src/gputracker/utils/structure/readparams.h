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

#ifndef _READPARAMS_H_
#define _READPARAMS_H_


#define FULLQUATSZ     4


extern void readInitialSBAEstimate(char *camsfname, char *ptsfname, int cnp, int pnp, int mnp, 
                                   void (*infilter)(double *pin, int nin, double *pout, int nout), int cnfp,
                                   int *ncams, int *n3Dpts, int *n2Dprojs,
                                   double **motstruct, double **initrot, double **imgpts, double **covimgpts, char **vmask);
extern void readCalibParams(char *fname, double ical[9]);
extern int readNumParams(char *fname);

extern void printSBAMotionData(FILE *fp, double *motstruct, int ncams, int cnp,
                               void (*outfilter)(double *pin, int nin, double *pout, int nout), int cnop);
extern void printSBAStructureData(FILE *fp, double *motstruct, int ncams, int n3Dpts, int cnp, int pnp);
extern void printSBAData(FILE *fp, double *motstruct, int cnp, int pnp, int mnp, 
                         void (*outfilter)(double *pin, int nin, double *pout, int nout), int cnop,
                         int ncams, int n3Dpts, double *imgpts, int n2Dprojs, char *vmask);

extern void saveSBAStructureDataAsPLY(char *fname, double *motstruct, int ncams, int n3Dpts, int cnp, int pnp, int withrgb);

#endif /* _READPARAMS_H_ */
