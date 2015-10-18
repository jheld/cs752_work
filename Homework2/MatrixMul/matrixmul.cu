#include <stdio.h>
#include <stdlib.h>
#include "matrixmul.h"

// include the kernel
#include "matrixmul_kernel.cu"

extern void computeGold(float*, const float*, const float*, unsigned int,
			unsigned int, unsigned int);

void gpuInit(void){
  cudaSetDevice(0);
}

void randomInit(float* data, int size){
  for (int i = 0; i < size; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

int main(int argc, char** argv){
  int WA = 4; // Matrix A width
  int HA = 6; // Matrix A height
  int WB = 4; // Matrix B width
  int HB; // Matrix B height
  int WC; // Matrix C width 
  int HC; // Matrix C height
  int i;

  i = 1;
  while (i<argc){
    if (!strcmp(argv[i],"wa")) sscanf(argv[i+1],"%d", &WA);
    else if (!strcmp(argv[i],"ha")) sscanf(argv[i+1],"%d", &HA);
    else if (!strcmp(argv[i],"wb")) sscanf(argv[i+1],"%d", &WB);
    i += 2;
  }
  WA = WA * BLOCK_SIZE;
  HA = HA * BLOCK_SIZE;
  WB = WB * BLOCK_SIZE;
  HB = WA;
  WC = WB;
  HC = HA;

  gpuInit();
  srand(2006);
  printf("A: %d by %d\nB: %d by %d\n", HA, WA, HB, WB);

  unsigned int size_A = WA * HA;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float* h_A = (float*)malloc(mem_size_A);
  unsigned int size_B = WB * HB;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float* h_B = (float*)malloc(mem_size_B);
  unsigned int size_C = WC * HC;
  unsigned int mem_size_C = sizeof(float) * size_C;
  float* h_C    = (float*) malloc(mem_size_C);
  float* h_Copy = (float*) malloc(mem_size_C);

  randomInit(h_A, size_A);
  randomInit(h_B, size_B);
  
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float msecTotal;

  cudaEventRecord(start, NULL);
  computeGold(h_Copy, h_A, h_B, HA, WA, WB);
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);
  printf("cpu time: %.3f ms\n", msecTotal);

  float* d_A, *d_B, *d_C;
  cudaMalloc((void**) &d_A, mem_size_A);
  cudaMalloc((void**) &d_B, mem_size_B);
  cudaMalloc((void**) &d_C, mem_size_C);

  cudaEventRecord(start, NULL);
  cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(WC/threads.x, HC/threads.y);
  matrixMul<<<grid, threads>>>(d_C, d_A, d_B, WA, WB);
  cudaDeviceSynchronize();
  cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);
  printf("gpu time: %.3f ms\n", msecTotal);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_Copy);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}
