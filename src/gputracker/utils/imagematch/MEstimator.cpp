#include "MEstimator.h"
#include "../timer/performanceCounter.h"
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

extern "C" void initHistogram64();
extern "C" void closeHistogram64();
extern "C" void histogram64(unsigned int *hist,void *d_Data,unsigned int length);
extern "C" void initHistogram256();
extern "C" void closeHistogram256();
extern "C" void histogram256(unsigned int *hist,void *d_Data,unsigned int length);
extern "C" void convertResidualCuda(float *residF, unsigned char *residC, int length);
extern "C" void initScan(void);
extern "C" void closeScan(void);
extern "C" size_t scanExclusiveShort(unsigned int *d_Dst,unsigned int *d_Src,unsigned int batchSize,unsigned int arrayLength);
extern "C" void medianCuda(unsigned int *cumuHistDev, unsigned int *cumuHistPlusDev, unsigned int *medianDev, float *tukeyThresholdDev);
extern "C" void tukeyThresholdCuda(unsigned int *cumuHistDev, unsigned int *cumuHistPlusDev, float *tukeyThresholdDev);
extern "C" void absDiffCuda(unsigned char *residualByteDev,unsigned int *medianDev,unsigned int length, unsigned char *residualStdev);
extern "C" void tukeyWCuda(float *tukeyThresholdDev, unsigned char *residualStdev, float *weights, unsigned int length);
 
void medianCuda(unsigned char *dataDev, unsigned int length, unsigned int *histDev, unsigned int *cumuHistDev, unsigned int *cumuHistPlusDev, unsigned int *hist, unsigned int *medianDev, float *tukeyThresholdDev) 
{
	// alignment to int size
	// note: 1-3 additional data values from previous time instants are also accumulated
	// this is assumed not to be a problem for median values!
	int length4 = (length/sizeof(int)+1)*sizeof(int);
	histogram256(histDev,(void*)dataDev,length4);
	cudaMemset( cumuHistDev, 0, sizeof(unsigned int)*256);
	scanExclusiveShort(cumuHistDev,histDev,1,256);
	cudaMemcpy( cumuHistPlusDev,&cumuHistDev[1],sizeof(unsigned int)*255,cudaMemcpyDeviceToDevice);
	medianCuda(cumuHistDev,cumuHistPlusDev,medianDev,tukeyThresholdDev);
} 

void medianStdevCuda(unsigned char *dataDev, unsigned int length, unsigned int *histDev, unsigned int *cumuHistDev, unsigned int *cumuHistPlusDev, unsigned int *hist, unsigned int *medianDev, float *tukeyThresholdDev) 
{
	// alignment to int size
	// note: 1-3 additional data values from previous time instants are also accumulated
	// this is assumed not to be a problem for median values!
	int length4 = (length/(sizeof(int)*4)+1)*(sizeof(int)*4);
	histogram64(histDev,(void*)dataDev,length4);
	cudaMemset( cumuHistDev, 0, sizeof(unsigned int)*64);
	scanExclusiveShort(cumuHistDev,histDev,1,256);
	cudaMemcpy( cumuHistPlusDev,&cumuHistDev[1],sizeof(unsigned int)*63,cudaMemcpyDeviceToDevice);
	tukeyThresholdCuda(cumuHistDev,cumuHistPlusDev,tukeyThresholdDev);
} 

// convert residual values into positive range [0,511] and divide bin resolution into [0,255]
// for matching gpu histogram routine
void MEstimator::convertResidual(float *residF, unsigned char *residC, unsigned int length) {
	for (unsigned int i = 0; i < length; i++) {
		float r = residF[i]; 
		if (r < -128) r = -128; if (r > 127) r = 127;
		residC[i] = unsigned int(r+128);
	}
}

void MEstimator::histogram256CPU(unsigned int *hist, unsigned char *data, unsigned int length, unsigned int *totalMass) {
	memset(hist,0,sizeof(unsigned int)*256);
	int mass = 0;
	for (unsigned int i = 0; i < length; i++) {hist[data[i]]++; mass++;}
	*totalMass = mass;
}

unsigned char MEstimator::median(unsigned char *data, unsigned int length, unsigned int *hist) 
{
	unsigned int totalMass;
	histogram256CPU(hist,data,length,&totalMass);
	unsigned int thresholdMass = totalMass>>1;
	unsigned int mass = 0;
	for (int i = 0; i < 256; i++){
		mass += hist[i];
		if (mass > thresholdMass) { return (unsigned char)i; }
	}
	return 255;
} 

/*
void huberW(int *val, float *weights, int length) {
	float threshold = 1.2107f;	
	for (int i = 0; i < length; i++) {
		if (val[i] < threshold) {
			weights[i] = 1.0f;
		} else {
			weights[i] = threshold/float(val[i]);
		}
	}
}
*/
void MEstimator::tukeyW(float deviation, unsigned char *val, float *weights, unsigned int length) {
	float thresdev = 4.6851f*deviation;
	for (unsigned int i = 0; i < length; i++) {
		float valf = float(val[i]);
		if (valf < thresdev) {
			float v = valf/thresdev;
			v *= v; v = 1-v; v *= v;
			weights[i] = v;
		} else {
			weights[i] = 0;
		}
	}
}

void MEstimator::generateWeights(float *residual, float *weights, unsigned int length) {
	//PerformanceCounter timer;
	//timer.StartCounter();
	convertResidual(residual,residualByte,length);
	// median of input 
	unsigned char medianVal = median(residualByte,length,hist);
	// deviation array
	for (unsigned int i = 0; i < length; i++) {
		residualStdev[i] = abs(residualByte[i]-medianVal);
	}
	// median deviation
	unsigned char medianDev = median(residualStdev,length,hist);
	//printf("mad: %d\n",medianDev);
	if (medianDev < 1) medianDev = 1;
	float deviation = 1.4826f * float(medianDev)/2.0f; // div 2 improves results although not in standard formula!
	// generate tukey weights
	tukeyW(deviation,residualStdev, weights, length);
	//timer.StopCounter(); printf("generateWeightsCPU: %fms\n",1000.0f*timer.GetElapsedTime()); 	timer.StartCounter();
}

void MEstimator::generateWeightsDev(float *residualDev, float *weightsDev, unsigned int length) {
	PerformanceCounter timer;
	timer.StartCounter();
	convertResidualCuda(residualDev,residualByteDev,length);
	timer.StopCounter(); printf("convertResidualCuda: %fms\n",1000.0f*timer.GetElapsedTime()); 	timer.StartCounter();
	// median of input 
	medianCuda(residualByteDev,length,histDev,cumuHistDev,cumuHistPlusDev,hist,medianDev,tukeyThresholdDev);
	timer.StopCounter(); printf("medianCuda: %fms\n",1000.0f*timer.GetElapsedTime()); 	timer.StartCounter();

	absDiffCuda(residualByteDev,medianDev,length,residualStdevDev);
	timer.StopCounter(); printf("absDiffCuda: %fms\n",1000.0f*timer.GetElapsedTime()); 	timer.StartCounter();

	// median deviation
	medianCuda(residualStdevDev,length,histDev,cumuHistDev,cumuHistPlusDev,hist,medianDev,tukeyThresholdDev);
	timer.StopCounter(); printf("medianCuda: %fms\n",1000.0f*timer.GetElapsedTime()); 	timer.StartCounter();

	tukeyWCuda(tukeyThresholdDev,residualStdevDev,weightsDev,length);
	timer.StopCounter(); printf("tukeyWCuda: %fms\n",1000.0f*timer.GetElapsedTime()); 	timer.StartCounter();
}

MEstimator::MEstimator( unsigned int nMaxPoints )
{
	this->nMaxPoints = nMaxPoints;
	residualByte = new unsigned char[nMaxPoints+4];
	residualStdev = new unsigned char[nMaxPoints+4];
	hist = new unsigned int[256];
	/*cudaMalloc( (void **)&residualByteDev, nMaxPoints*sizeof(unsigned char)+4); 
	cudaMalloc( (void **)&residualStdevDev, nMaxPoints*sizeof(unsigned char)+4); 
	cudaMalloc( (void **)&histDev, 256*sizeof(unsigned int)); 
	cudaMalloc( (void **)&cumuHistDev, 256*sizeof(unsigned int)); 
	cudaMalloc( (void **)&cumuHistPlusDev, 257*sizeof(unsigned int)); 
	cudaMalloc( (void **)&medianDev, 2*sizeof(unsigned int)); 
	cudaMalloc( (void **)&tukeyThresholdDev, 2*sizeof(float)); */

//	initHistogram64();
//	initHistogram256();
//	initScan();

//	cudaMemset( residualByteDev, 255, sizeof(unsigned char)*nMaxPoints+4);
//	cudaMemset( residualStdevDev, 255, sizeof(unsigned char)*nMaxPoints+4);
//	cudaMemset( cumuHistPlusDev, 255, sizeof(unsigned int)*257);
}

MEstimator::~MEstimator()
{
	delete[] residualByte;
	delete[] residualStdev;
	delete[] hist;
/*	if (residualByteDev != NULL)    { cudaFree(residualByteDev);  residualByteDev = NULL; }
	if (residualStdevDev != NULL)   { cudaFree(residualStdevDev); residualStdevDev = NULL; }
	if (histDev != NULL)            { cudaFree(histDev);          histDev = NULL; }
	if (cumuHistDev != NULL)        { cudaFree(cumuHistDev);      cumuHistDev = NULL; }
	if (cumuHistPlusDev != NULL)    { cudaFree(cumuHistPlusDev);  cumuHistPlusDev = NULL; }
	if (medianDev != NULL)          { cudaFree(medianDev);        medianDev = NULL; }
	if (tukeyThresholdDev != NULL)  { cudaFree(tukeyThresholdDev);     tukeyThresholdDev = NULL; }*/

//	closeHistogram64();
//	closeHistogram256();
//	closeScan();
}