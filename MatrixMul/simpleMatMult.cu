#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define THREADS 16
#define BLOCKS 32
#define WIDTH (THREADS * BLOCKS)

int size = WIDTH * WIDTH * sizeof(float);
float *M, *N, *P;
float *gpuM, *gpuN, *gpuP;
time_t seed;

void initGPU(int devNum){
  cudaSetDevice(devNum);

  cudaMalloc((void**)&gpuM, size);
  cudaMalloc((void**)&gpuN, size);
  cudaMalloc((void**)&gpuP, size);
}

void randomInit(float* data, int size){
  for (int i=0; i<size; i++)
    data[i] = drand48();
}

void cpuMatrixMul(const float* M, const float* N, float* P){
  int i, j, k;
  float sum;

  for (i=0; i<WIDTH; i++)
    for (j=0; j<WIDTH; j++){
      sum = 0.0f;
      for (k=0; k<WIDTH; k++)
	sum += M[i * WIDTH + k] * N[k * WIDTH + j];
      P[i * WIDTH + j] = sum;
    }
}

__global__ void
matrixMultKernel(float* M, float* N, float* P){
  int tx = blockIdx.x * THREADS + threadIdx.x;
  int ty = blockIdx.y * THREADS + threadIdx.y;
  float tmp = 0.0f;

  for (int k=0; k<WIDTH; k++)
    tmp += M[ty * WIDTH + k] * N[k * WIDTH + tx];
  P[ty * WIDTH + tx] = tmp;
}

int main(void){
  float msecTotal;
  cudaEvent_t start;
  cudaEvent_t stop;

  initGPU(0);
  time(&seed);
  srand48(seed);
  M = (float*)malloc(size);
  N = (float*)malloc(size);
  P = (float*)malloc(size);
  randomInit(M, WIDTH*WIDTH);
  randomInit(N, WIDTH*WIDTH);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, NULL);
  cpuMatrixMul(M, N, P);
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);
  printf("cpu time: %.3f ms\n", msecTotal);

  cudaEventRecord(start, NULL);
  cudaMemcpy(gpuM, M, size, cudaMemcpyHostToDevice);
  cudaMemcpy(gpuN, N, size, cudaMemcpyHostToDevice);
  dim3 numThread(THREADS, THREADS);
  dim3 numBlock(BLOCKS, BLOCKS);
  matrixMultKernel<<<numBlock, numThread>>>(gpuM, gpuN, gpuP);
  cudaDeviceSynchronize();
  cudaMemcpy(P, gpuP, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);
  printf("gpu time: %.3f ms\n",msecTotal);

  free(M);
  free(N);
  free(P);
  cudaFree(gpuM);
  cudaFree(gpuN);
  cudaFree(gpuP);
  return 0;
}
